"""
Attack runner for diffusion-based watermarks (Tree-Ring and PRC).
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

# We need to import code from the shai branch, which lives in the parent directory
# This adds that directory to Python's search path so we can import watermark detection code
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)


@dataclass
class DiffusionAttackResult:
    """This stores the results from one attack test.
    
    For example, if we compress an image with JPEG quality 80, this stores:
    - Whether the watermark was still detected (detected: True/False)
    - How confident we are in the detection (confidence: 0.0 to 1.0)
    - How much the image quality degraded (psnr, ssim metrics)
    - Where we saved the attacked image (output_path)
    """
    name: str  # e.g., "jpeg_q80"
    attack_type: str  # e.g., "compression"
    params: Dict[str, Any]  # e.g., {"quality": 80}
    confidence: float  # how confident we are the watermark is still there (0-1)
    p_value: Optional[float]  # statistical significance (lower = more confident)
    detected: bool  # did we detect the watermark? (True/False)
    psnr: float  # peak signal-to-noise ratio (higher = better quality)
    ssim: float  # structural similarity (higher = more similar to original)
    output_path: str  # where we saved the attacked image


@dataclass 
class DiffusionAttackConfig:
    """This holds all the settings for running our attack tests.
    
    Think of this as the recipe for our experiment:
    - Which watermarking method are we testing? (Tree-Ring or PRC)
    - Which image should we attack?
    - What attacks should we try? (compression, cropping, noise, etc.)
    - Where should we save the results?
    """
    watermark_method: str  # either 'treering' or 'prc'
    watermarked_image: str  # path to the watermarked image we'll attack
    original_image: Optional[str] = None  # path to clean image (to measure quality loss)
    output_dir: str = "results/diffusion_attacks"  # where to save attacked images
    results_file: Optional[str] = None  # where to save the JSON results
    attacks: List[Dict[str, Any]] = field(default_factory=list)  # list of attacks to apply
    
    # These settings tell us which Stable Diffusion model to use for detection
    model_id: str = "runwayml/stable-diffusion-v1-5"  # which model checkpoint to load
    num_inference_steps: int = 50  # how many steps for forward/reverse diffusion (more = slower but more accurate)
    device: str = "cuda"  # use GPU ("cuda") or CPU ("cpu")
    
    # Tree-Ring specific parameters
    treering_eps: float = 1e-9  # the epsilon value used when embedding Tree-Ring (tiny number like 0.000000001)
    
    # PRC specific parameters
    prc_key_path: Optional[str] = None  # path to the PRC encoding/decoding key file (a .pkl file)


class DiffusionWatermarkDetector:
    """
    This is the brain of the operation - it detects watermarks embedded during diffusion.
    
    Here's why this is different from normal watermark detection:
    - Normal watermarks: Look at the pixels directly
    - Diffusion watermarks: Need to reverse-engineer the image back to noise
    
    The detection process:
    1. Take your image (could be attacked/modified)
    2. Use VAE to encode it into latent space (compress 512x512 → 64x64)
    3. Run FORWARD diffusion to add noise back (this is the reverse of image generation!)
    4. Analyze the noise pattern - this is where the watermark lives
    5. Check if the watermark signature is still there
    
    Think of it like: the watermark is hidden in the "recipe" used to cook the image,
    not in the final dish itself. So we need to deconstruct the dish to find the recipe.
    """
    
    def __init__(self, config: DiffusionAttackConfig):
        self.config = config
        self.pipe = None  # we'll load the diffusion model lazily (only when needed)
        self.null_distribution = None  # for hypothesis testing (not implemented yet)
        
    def _load_pipe(self):
        """Load the Stable Diffusion model only when we need it (lazy loading).
        
        This is useful because loading the model takes time and memory,
        so we only do it once when we first need to detect something.
        """
        if self.pipe is None:
            try:
                # This imports the custom Stable Diffusion code from shai's branch
                from watermark_stable_diffusion import WatermarkStableDiffusion
                print(f"Loading diffusion model: {self.config.model_id}")
                self.pipe = WatermarkStableDiffusion.from_pretrained(self.config.model_id)
                self.pipe = self.pipe.to(self.config.device)  # move to GPU or CPU
            except ImportError:
                raise ImportError(
                    "Could not import WatermarkStableDiffusion. "
                    "Make sure watermark_stable_diffusion.py from shai branch is accessible."
                )
        return self.pipe
    
    def _image_to_latent_noise(self, image_path: str) -> torch.Tensor:
        """
        This is the CRITICAL step: convert an image back into latent noise.
        
        Think of it like reverse-engineering:
        - When creating an image: noise → (diffusion) → image
        - When detecting: image → (VAE + forward diffusion) → noise
        
        The steps:
        1. Load the image from disk
        2. Transform it to the format the model expects (normalization, etc.)
        3. Use VAE to encode: 512x512 RGB image → 64x64 latent representation
        4. Run FORWARD diffusion: gradually add noise over 50 steps
        5. Extract the final noise - this contains the watermark pattern!
        
        This is what makes diffusion watermarks special - the watermark lives
        in the noise space, not the pixel space.
        """
        from utils import transform_image
        
        pipe = self._load_pipe()
        
        # Step 1: Load and prepare the image
        image = Image.open(image_path)
        image_tensor = transform_image(image).unsqueeze(0).to(self.config.device)
        
        # Step 2: Encode to latent space using VAE (like compressing the image into a code)
        image_latents = pipe.get_image_latents(image=image_tensor, sample=False)
        
        # Step 3: Get text embeddings (we use empty prompt since we're just detecting, not generating)
        text_embeddings = pipe.get_text_embedding('')  # Empty prompt
        
        # Step 4: Run FORWARD diffusion to recover the original noise
        # This is the opposite of image generation!
        # Generation: noise → clean image (reverse diffusion)
        # Detection: clean image → noise (forward diffusion)
        pipe(
            latents=image_latents,
            prompt_embeds=text_embeddings,
            output_type='pil',
            guidance_scale=1,
            num_inference_steps=self.config.num_inference_steps,
            forward_process=True  # This is KEY - tells it to go backwards (forward in noise)
        )
        
        # Step 5: Extract the final noisy latents - this is where the watermark signature lives!
        noise_latents = pipe.get_last_latent()
        return noise_latents
    
    def detect_treering(self, image_path: str) -> Dict[str, Any]:
        """
        Detect Tree-Ring watermark by looking for a ring pattern in frequency space.
        
        Tree-Ring watermarking works by embedding a circular pattern (like tree rings!)
        in the Fourier transform of the latent noise. The idea:
        1. During generation: Set specific frequencies in a ring to a constant value (epsilon)
        2. During detection: Check if those frequencies still have unusual energy
        
        Detection process:
        1. Convert image → latent noise (using forward diffusion)
        2. Take FFT (Fast Fourier Transform) to see frequency patterns
        3. Create a ring mask at ~70% radius (where the watermark should be)
        4. Compare energy inside the ring vs outside
        5. If ring has significantly more energy → watermark detected!
        
        The ratio tells us: is the ring region "brighter" than the background?
        Higher ratio = stronger watermark signal
        """
        from utils import generate_watermark_mask, calc_watermark_dist
        
        # Step 1: Get the latent noise from the image (this is where the watermark lives)
        noise_latents = self._image_to_latent_noise(image_path)
        
        # Step 2: Compute FFT to convert from spatial domain → frequency domain
        # This lets us see repeating patterns at different frequencies
        fft_latents = torch.fft.fftshift(torch.fft.fft2(noise_latents))
        magnitudes = torch.abs(fft_latents).cpu().numpy()  # strength of each frequency
        
        # Step 3: Generate the ring mask - this marks where we expect the watermark
        # Uses shai's code to create a circular ring at 70% radius with width 2
        watermark_mask = generate_watermark_mask(noise_latents)[0, 0].cpu().numpy()
        
        # Step 4: Extract and compare energies
        # Split the FFT into two groups: inside the ring vs outside the ring
        ring_magnitudes = magnitudes[0, 0][watermark_mask]  # frequencies where watermark should be
        non_ring_magnitudes = magnitudes[0, 0][~watermark_mask]  # everywhere else
        
        # Step 5: Calculate the ring ratio - our main detection statistic
        # This tells us: is the ring region stronger than the background?
        ring_energy = float(np.mean(ring_magnitudes))  # average strength in the ring
        non_ring_energy = float(np.mean(non_ring_magnitudes))  # average strength elsewhere
        ring_ratio = ring_energy / (non_ring_energy + 1e-10)  # how much stronger? (1.0 = same, 2.0 = twice as strong)
        
        # Also compute distance metric from shai's code (measures deviation from epsilon)
        fft_numpy = fft_latents.detach().cpu().numpy()
        distance = calc_watermark_dist(fft_numpy, self.config.treering_eps)
        
        # Step 6: Make detection decision
        # Simple threshold: if ring is 10% stronger than background, we call it detected
        # (In a real system, you'd use statistical hypothesis testing with p-values)
        threshold = 1.1  # Should ideally be set from null distribution experiments
        detected = ring_ratio > threshold
        
        # Convert ratio to a 0-1 confidence score (higher = more confident)
        # Formula: how much above 1.0 is the ratio, scaled to [0, 1]
        confidence = min(max(ring_ratio - 1.0, 0.0) * 2.0, 1.0)
        
        return {
            'method': 'treering',
            'confidence': confidence,
            'detected': detected,
            'ring_ratio': ring_ratio,
            'ring_energy': ring_energy,
            'non_ring_energy': non_ring_energy,
            'distance': distance,
            'p_value': None,  # Would need null distribution for this
        }
    
    def detect_prc(self, image_path: str) -> Dict[str, Any]:
        """
        Detect PRC watermark by trying to decode a hidden binary message.
        
        PRC (Pseudorandom Code) is like hiding a secret password in the image:
        1. During generation: Embed a binary message by controlling +/- signs of noise
        2. During detection: Try to decode the message using error-correcting codes
        3. If we can successfully decode → watermark is there!
        
        The clever part: PRC uses error-correcting codes (like those used in WiFi/CDs)
        so even if some bits get flipped by attacks, we can still recover the message.
        
        Detection process:
        1. Convert image → latent noise
        2. Take FFT and look at real part (positive vs negative values)
        3. Extract binary pattern from signs (+1 → 1, -1 → 0)
        4. Run PRC decoder to try recovering the message
        5. Decoder gives us confidence: how sure are we this is the right message?
        """
        from prc import Decode
        
        # Step 1: Load the PRC decoding key (this is like the secret decoder ring!)
        if not self.config.prc_key_path or not os.path.exists(self.config.prc_key_path):
            raise ValueError(f"PRC key not found: {self.config.prc_key_path}")
        
        # The key contains encoding/decoding matrices for the error-correcting code
        import pickle
        with open(self.config.prc_key_path, 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        
        # Step 2: Get latent noise from the image
        noise_latents = self._image_to_latent_noise(image_path)
        
        # Step 3: Take FFT to get frequency representation
        fft_latents = torch.fft.fftshift(torch.fft.fft2(noise_latents))
        real_part = fft_latents.real  # PRC uses the real component (ignores imaginary part)
        
        # Step 4: Extract binary message from signs
        # The watermark is encoded in whether each value is positive (+1) or negative (-1)
        signs = (real_part > 0).float()  # convert to 1s and 0s
        
        # Step 5: Decode using PRC's error-correcting decoder
        # This tries to recover the original message, correcting any bit errors
        extracted_codeword = signs[0].flatten().cpu().numpy()  # flatten to 1D array of bits
        decoded_message, confidence = Decode(decoding_key, extracted_codeword[:len(extracted_codeword)])
        
        # Step 6: Make detection decision based on decoder confidence
        # The decoder tells us: how confident am I that this is a valid message?
        # High confidence (>0.8) → message decoded successfully → watermark detected!
        threshold = 0.8
        detected = confidence > threshold
        
        return {
            'method': 'prc',
            'confidence': float(confidence),
            'detected': detected,
            'message': decoded_message if decoded_message else None,
            'p_value': None,  # PRC provides confidence directly
        }
    
    def detect_watermark(self, image_path: str) -> Dict[str, Any]:
        """
        Detect watermark based on configured method.
        """
        if self.config.watermark_method.lower() == 'treering':
            return self.detect_treering(image_path)
        elif self.config.watermark_method.lower() == 'prc':
            return self.detect_prc(image_path)
        else:
            raise ValueError(f"Unknown watermark method: {self.config.watermark_method}")


class DiffusionAttackRunner:
    """
    Applies attacks to diffusion-watermarked images and evaluates robustness.
    
    Similar to attack_runner.py but adapted for watermarks embedded during
    reverse diffusion (requires forward diffusion for detection).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = DiffusionAttackConfig(**config)
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.detector = DiffusionWatermarkDetector(self.config)
    
    def _apply_attack(self, image: np.ndarray, attack: Dict[str, Any], name: str) -> str:
        """
        Apply one attack to the image and save the result.
        
        Each attack simulates a real-world scenario:
        - Compression: someone saves as JPEG (loses some information)
        - Rescaling: image gets resized (e.g., for thumbnails)
        - Cropping: part of image is cut off
        - Gaussian noise: camera sensor noise or transmission errors
        - Gaussian blur: out-of-focus camera or intentional blur
        - Rotation: image gets rotated by a few degrees
        
        Returns: path to the attacked image file
        """
        attack_type = attack.get("type")
        output_path = os.path.join(self.config.output_dir, f"{name}.png")
        
        # Attack 1: JPEG compression
        # This is the most common attack - what happens when someone saves as JPEG?
        if attack_type == "compression":
            quality = int(attack.get("quality", 50))  # 0-100, lower = more compression
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return output_path
        
        # Attack 2: Rescaling (downscale then upscale back)
        # Simulates making a thumbnail or low-res copy
        if attack_type == "rescaling":
            scale = float(attack.get("scale", 0.5))  # e.g., 0.5 = shrink to 50%
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # Scale back to original size for fair comparison (this loses detail)
            original_size = (image.shape[1], image.shape[0])
            resized = cv2.resize(resized, original_size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output_path, resized)
            return output_path
        
        # Attack 3: Cropping (cut center, then scale back up)
        # Simulates someone cropping to reframe the image
        if attack_type == "cropping":
            frac = float(attack.get("crop_fraction", 0.8))  # keep 80% of image
            h, w = image.shape[:2]
            new_h, new_w = int(h * frac), int(w * frac)
            start_y = max((h - new_h) // 2, 0)  # center crop
            start_x = max((w - new_w) // 2, 0)
            cropped = image[start_y : start_y + new_h, start_x : start_x + new_w]
            # Resize to original size (stretches the cropped region)
            cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output_path, cropped)
            return output_path
        
        # Attack 4: Gaussian noise (random pixel noise)
        # Simulates camera sensor noise or transmission errors
        if attack_type == "gaussian_noise":
            sigma = float(attack.get("sigma", 5.0))  # noise strength (typical: 5-15)
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(output_path, noisy)
            return output_path
        
        # Attack 5: Gaussian blur (smoothing filter)
        # Simulates out-of-focus camera or intentional blurring
        if attack_type == "gaussian_blur":
            ksize = int(attack.get("ksize", 5))  # kernel size (larger = more blur)
            if ksize % 2 == 0:  # OpenCV requires odd kernel size
                ksize += 1
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
            cv2.imwrite(output_path, blurred)
            return output_path
        
        # Attack 6: Rotation
        # Simulates rotating the image by a few degrees
        if attack_type == "rotation":
            angle = float(attack.get("angle", 15))  # degrees
            h, w = image.shape[:2]
            center = (w // 2, h // 2)  # rotate around center
            M = cv2.getRotationMatrix2D(center, angle, 1.0)  # create rotation matrix
            rotated = cv2.warpAffine(image, M, (w, h))  # apply rotation
            cv2.imwrite(output_path, rotated)
            return output_path
        
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    def _compute_psnr_ssim(self, reference_path: str, attacked_path: str) -> tuple[float, float]:
        """Calculate image quality metrics (how much did the attack degrade quality?).
        
        PSNR (Peak Signal-to-Noise Ratio):
        - Measures average pixel difference
        - Higher is better (30+ dB is good, 40+ is excellent)
        - Infinity means identical images
        
        SSIM (Structural Similarity Index):
        - Measures perceptual similarity (considers patterns, not just pixels)
        - Range: 0 to 1, where 1 = identical
        - 0.9+ means very similar, 0.7-0.9 is decent, <0.7 is noticeable degradation
        """
        ref = cv2.imread(reference_path)
        attacked = cv2.imread(attacked_path)
        
        if ref is None or attacked is None:
            return 0.0, 0.0
        
        if ref.shape != attacked.shape:
            attacked = cv2.resize(attacked, (ref.shape[1], ref.shape[0]), 
                                 interpolation=cv2.INTER_AREA)
        
        psnr = float(cv2.PSNR(ref, attacked))
        ssim = float(structural_similarity(ref, attacked, channel_axis=2))
        return psnr, ssim
    
    def run(self) -> List[Dict[str, Any]]:
        """
        This is the main evaluation pipeline - runs all attacks and tests watermark survival.
        
        The process for each attack:
        1. Apply the attack to the watermarked image (JPEG compress, crop, add noise, etc.)
        2. Save the attacked image
        3. Run watermark detection on the attacked image
           - This means: convert to noise via forward diffusion, analyze pattern
        4. Measure how much quality was lost (PSNR, SSIM)
        5. Record whether watermark survived and how confident we are
        
        At the end, we save all results to JSON and print a summary table.
        """
        # Load the watermarked image we'll be attacking
        watermarked_img = cv2.imread(self.config.watermarked_image)
        if watermarked_img is None:
            raise ValueError(f"Could not load watermarked image: {self.config.watermarked_image}")
        
        results: List[DiffusionAttackResult] = []
        
        # First, establish a baseline: can we detect the watermark in the original (no attack)?
        # This tells us if the watermark was successfully embedded in the first place
        print("Testing baseline (no attack)...")
        baseline_detection = self.detector.detect_watermark(self.config.watermarked_image)
        print(f"  Baseline detection: {baseline_detection['detected']}, "
              f"confidence: {baseline_detection['confidence']:.4f}")
        
        # Save baseline results (perfect image quality since no attack)
        results.append(
            DiffusionAttackResult(
                name="baseline_no_attack",
                attack_type="none",
                params={},
                confidence=baseline_detection['confidence'],
                p_value=baseline_detection.get('p_value'),
                detected=baseline_detection['detected'],
                psnr=float('inf'),  # Perfect match = infinite PSNR
                ssim=1.0,  # Perfect match = SSIM of 1.0
                output_path=self.config.watermarked_image,
            ).__dict__
        )
        
        # Now run each attack and test if the watermark survives
        for attack in self.config.attacks:
            name = attack.get("name") or attack.get("type")
            print(f"\nRunning attack: {name} ({attack.get('type')})...")
            
            # Step 1: Apply the attack (this modifies the image)
            output_path = self._apply_attack(watermarked_img, attack, name)
            print(f"  Attack applied, saved to: {output_path}")
            
            # Step 2: Try to detect the watermark in the attacked image
            # This is the expensive part - requires running forward diffusion!
            print(f"  Detecting watermark (running forward diffusion)...")
            detection_result = self.detector.detect_watermark(output_path)
            
            # Step 3: Measure how much quality was lost
            psnr, ssim = self._compute_psnr_ssim(self.config.watermarked_image, output_path)
            
            # Step 4: Package up the results
            result = DiffusionAttackResult(
                name=name,
                attack_type=attack.get("type", "unknown"),
                params={k: v for k, v in attack.items() if k not in {"name", "type"}},
                confidence=detection_result['confidence'],
                p_value=detection_result.get('p_value'),
                detected=detection_result['detected'],
                psnr=psnr,
                ssim=ssim,
                output_path=output_path,
            ).__dict__
            
            results.append(result)
            
            # Print immediate feedback
            print(f"  Detection: {'DETECTED ✓' if result['detected'] else 'NOT DETECTED ✗'}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        # Save all results to JSON file for later analysis
        if self.config.results_file:
            os.makedirs(os.path.dirname(self.config.results_file), exist_ok=True)
            with open(self.config.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {self.config.results_file}")
        
        # Print a nice summary table showing all results at a glance
        print("\n" + "="*70)
        print("ATTACK SUMMARY")
        print("="*70)
        print(f"{'Attack':<25} {'Detected':<12} {'Confidence':<12} {'PSNR':<10} {'SSIM':<10}")
        print("-"*70)
        for r in results:
            detected_str = '✓' if r['detected'] else '✗'
            print(f"{r['name']:<25} {detected_str:<12} {r['confidence']:>10.4f}  "
                  f"{r['psnr']:>8.2f}  {r['ssim']:>8.4f}")
        print("="*70)
        print("\nLegend:")
        print("  Detected: ✓ = watermark found, ✗ = watermark not found")
        print("  Confidence: 0.0 to 1.0 (higher = more confident the watermark is there)")
        print("  PSNR: Peak Signal-to-Noise Ratio in dB (higher = better image quality)")
        print("  SSIM: Structural Similarity 0-1 (higher = more similar to original)")
        
        return results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Run attacks on diffusion-watermarked images (Tree-Ring, PRC)"
    )
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    parser.add_argument("--suite", type=str, required=True,
                       help="Name of attack suite to run")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get suite config
    suite_config = config.get("diffusion_attack_suites", {}).get(args.suite)
    if suite_config is None:
        raise ValueError(f"Suite '{args.suite}' not found in {args.config}")
    
    # Run attacks
    print("="*70)
    print(f"DIFFUSION WATERMARK ATTACK EVALUATION: {args.suite}")
    print("="*70)
    
    runner = DiffusionAttackRunner(suite_config)
    results = runner.run()
    
    print(f"\n✓ Evaluation complete!")

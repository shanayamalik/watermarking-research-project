"""
Robustness Testing for LSB vs DCT Watermarking

This script tests how well each watermarking method survives various attacks:
- JPEG compression
- Gaussian noise
- Salt & pepper noise
- Cropping
- Scaling/resizing
- Rotation (small angles)
"""

import numpy as np
import cv2
import os
import sys
from io import BytesIO
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lsb_watermarking import LSBWatermarking
from dct_watermarking import DCTWatermarking


class RobustnessTest:
    """Test robustness of watermarking methods against various attacks."""
    
    def __init__(self):
        self.lsb_watermarker = LSBWatermarking()
        self.dct_watermarker = DCTWatermarking(alpha=0.1)
        self.test_message = "Test watermark for robustness evaluation!"
    
    def apply_jpeg_compression(self, image, quality=50):
        """Apply JPEG compression attack."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        return compressed_img
    
    def apply_gaussian_noise(self, image, std=10):
        """Apply Gaussian noise attack."""
        noise = np.random.normal(0, std, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def apply_salt_pepper_noise(self, image, prob=0.01):
        """Apply salt and pepper noise attack."""
        noisy_image = image.copy()
        # Salt noise
        salt_mask = np.random.random(image.shape[:2]) < prob/2
        noisy_image[salt_mask] = 255
        # Pepper noise
        pepper_mask = np.random.random(image.shape[:2]) < prob/2
        noisy_image[pepper_mask] = 0
        return noisy_image
    
    def apply_cropping(self, image, crop_ratio=0.1):
        """Apply cropping attack (remove borders)."""
        h, w = image.shape[:2]
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        cropped = image[crop_h:h-crop_h, crop_w:w-crop_w]
        # Resize back to original size
        resized = cv2.resize(cropped, (w, h))
        return resized
    
    def apply_scaling(self, image, scale_factor=0.5):
        """Apply scaling attack (resize down then up)."""
        h, w = image.shape[:2]
        # Scale down
        small = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
        # Scale back up
        restored = cv2.resize(small, (w, h))
        return restored
    
    def apply_rotation(self, image, angle=5):
        """Apply small rotation attack."""
        h, w = image.shape[:2]
        center = (w//2, h//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def test_lsb_robustness(self, original_path, watermarked_path):
        """Test LSB watermark robustness against various attacks."""
        results = {}
        watermarked_image = cv2.imread(watermarked_path)
        
        attacks = {
            'No Attack': lambda img: img,
            'JPEG Q=90': lambda img: self.apply_jpeg_compression(img, 90),
            'JPEG Q=70': lambda img: self.apply_jpeg_compression(img, 70),
            'JPEG Q=50': lambda img: self.apply_jpeg_compression(img, 50),
            'Gaussian Noise σ=5': lambda img: self.apply_gaussian_noise(img, 5),
            'Gaussian Noise σ=10': lambda img: self.apply_gaussian_noise(img, 10),
            'Salt&Pepper 1%': lambda img: self.apply_salt_pepper_noise(img, 0.01),
            'Cropping 10%': lambda img: self.apply_cropping(img, 0.1),
            'Scaling 0.5x': lambda img: self.apply_scaling(img, 0.5),
            'Rotation 2°': lambda img: self.apply_rotation(img, 2),
        }
        
        for attack_name, attack_func in attacks.items():
            try:
                # Apply attack
                attacked_image = attack_func(watermarked_image)
                
                # Save attacked image temporarily
                temp_path = f"results/temp_lsb_attacked.png"
                cv2.imwrite(temp_path, attacked_image)
                
                # Try to extract watermark
                try:
                    extracted_message = self.lsb_watermarker.extract_watermark(temp_path)
                    success = (extracted_message.strip() == self.test_message)
                except:
                    success = False
                
                results[attack_name] = success
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                results[attack_name] = False
        
        return results
    
    def test_dct_robustness(self, original_path, watermarked_path):
        """Test DCT watermark robustness using correlation detection."""
        results = {}
        watermarked_image = cv2.imread(watermarked_path)
        
        attacks = {
            'No Attack': lambda img: img,
            'JPEG Q=90': lambda img: self.apply_jpeg_compression(img, 90),
            'JPEG Q=70': lambda img: self.apply_jpeg_compression(img, 70),
            'JPEG Q=50': lambda img: self.apply_jpeg_compression(img, 50),
            'Gaussian Noise σ=5': lambda img: self.apply_gaussian_noise(img, 5),
            'Gaussian Noise σ=10': lambda img: self.apply_gaussian_noise(img, 10),
            'Salt&Pepper 1%': lambda img: self.apply_salt_pepper_noise(img, 0.01),
            'Cropping 10%': lambda img: self.apply_cropping(img, 0.1),
            'Scaling 0.5x': lambda img: self.apply_scaling(img, 0.5),
            'Rotation 2°': lambda img: self.apply_rotation(img, 2),
        }
        
        for attack_name, attack_func in attacks.items():
            try:
                # Apply attack
                attacked_image = attack_func(watermarked_image)
                
                # Save attacked image temporarily
                temp_path = f"results/temp_dct_attacked.png"
                cv2.imwrite(temp_path, attacked_image)
                
                # Detect watermark using correlation
                correlation = self.dct_watermarker.detect_watermark(original_path, temp_path)
                # Consider correlation > 0.3 as successful detection
                success = correlation > 0.3
                results[attack_name] = (success, correlation)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                results[attack_name] = (False, 0.0)
        
        return results
    
    def compare_methods(self, original_path):
        """Compare LSB and DCT methods across all metrics."""
        print("WATERMARKING METHODS COMPARISON")
        print("=" * 80)
        
        # Create watermarked images
        print("Creating watermarked images...")
        
        # LSB watermarking
        lsb_output = "results/mandrill_lsb_robustness_test.png"
        self.lsb_watermarker.embed_watermark(original_path, self.test_message, lsb_output)
        
        # DCT watermarking
        dct_output = "results/mandrill_dct_robustness_test.png"
        self.dct_watermarker.embed_watermark(original_path, dct_output)
        
        # Test imperceptibility
        original = cv2.imread(original_path)
        lsb_watermarked = cv2.imread(lsb_output)
        dct_watermarked = cv2.imread(dct_output)
        
        # Calculate PSNR for both
        lsb_psnr = self.calculate_psnr(original, lsb_watermarked)
        dct_psnr = self.calculate_psnr(original, dct_watermarked)
        
        print(f"\nIMPERCEPTIBILITY COMPARISON:")
        print(f"LSB PSNR: {lsb_psnr:.2f} dB")
        print(f"DCT PSNR: {dct_psnr:.2f} dB")
        
        # Test robustness
        print(f"\nROBUSTNESS TESTING:")
        print(f"Testing LSB robustness...")
        lsb_results = self.test_lsb_robustness(original_path, lsb_output)
        
        print(f"Testing DCT robustness...")
        dct_results = self.test_dct_robustness(original_path, dct_output)
        
        # Display results
        self.display_comparison_results(lsb_results, dct_results, lsb_psnr, dct_psnr)
    
    def calculate_psnr(self, original, watermarked):
        """Calculate PSNR between two images."""
        if original.shape != watermarked.shape:
            watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))
        
        mse = np.mean((original.astype(float) - watermarked.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def display_comparison_results(self, lsb_results, dct_results, lsb_psnr, dct_psnr):
        """Display detailed comparison results."""
        print(f"\nDETAILED ROBUSTNESS COMPARISON:")
        print("=" * 80)
        print(f"{'Attack Type':<20} {'LSB Success':<12} {'DCT Success':<12} {'DCT Correlation':<15}")
        print("-" * 80)
        
        lsb_success_count = 0
        dct_success_count = 0
        
        for attack in lsb_results.keys():
            lsb_success = lsb_results[attack]
            dct_success, dct_corr = dct_results[attack]
            
            if lsb_success:
                lsb_success_count += 1
            if dct_success:
                dct_success_count += 1
            
            lsb_status = "✅ PASS" if lsb_success else "❌ FAIL"
            dct_status = "✅ PASS" if dct_success else "❌ FAIL"
            
            print(f"{attack:<20} {lsb_status:<12} {dct_status:<12} {dct_corr:<15.3f}")
        
        print("-" * 80)
        print(f"SUMMARY:")
        print(f"LSB: {lsb_success_count}/{len(lsb_results)} attacks survived ({lsb_success_count/len(lsb_results)*100:.1f}%)")
        print(f"DCT: {dct_success_count}/{len(dct_results)} attacks survived ({dct_success_count/len(dct_results)*100:.1f}%)")
        print(f"\nIMPERCEPTIBILITY:")
        print(f"LSB PSNR: {lsb_psnr:.2f} dB")
        print(f"DCT PSNR: {dct_psnr:.2f} dB")
        
        print(f"\nCONCLUSION:")
        if dct_success_count > lsb_success_count:
            print("🏆 DCT watermarking shows better robustness against attacks")
        elif lsb_success_count > dct_success_count:
            print("🏆 LSB watermarking shows better robustness against attacks")
        else:
            print("🤝 Both methods show similar robustness")
        
        if lsb_psnr > dct_psnr:
            print("🏆 LSB watermarking shows better imperceptibility")
        elif dct_psnr > lsb_psnr:
            print("🏆 DCT watermarking shows better imperceptibility")
        else:
            print("🤝 Both methods show similar imperceptibility")


if __name__ == "__main__":
    tester = RobustnessTest()
    original_image = "images/4.2.03.tiff"
    
    if not os.path.exists(original_image):
        print(f"Original image not found: {original_image}")
        exit(1)
    
    # Run comprehensive comparison
    tester.compare_methods(original_image)
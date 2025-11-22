"""Configurable attack runner for watermark robustness experiments."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from dct_watermarking import DCTWatermarking
from fourier_watermarking import FourierWatermarking
from lsb_watermarking import LSBWatermarking
from stable_diffusion_watermarking import StableDiffusionWatermarking

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


@dataclass
class AttackResult:
    name: str
    attack_type: str
    params: Dict[str, Any]
    confidence: float
    psnr: float
    ssim: float
    output_path: str


@dataclass
class AttackRunnerConfig:
    watermark_method: str
    watermarked_image: str
    cover_image: str | None = None
    message: str | None = None
    output_dir: str = "results/attacks"
    results_file: str | None = None
    attacks: List[Dict[str, Any]] = field(default_factory=list)


class AttackRunner:
    def __init__(self, config: Dict[str, Any]):
        resolved = config.copy()
        resolved["watermarked_image"] = _resolve_path(config.get("watermarked_image"))
        resolved["cover_image"] = _resolve_path(config.get("cover_image"))
        resolved["output_dir"] = _resolve_path(config.get("output_dir") or "results/attacks")
        resolved["results_file"] = _resolve_path(config.get("results_file"))
        self.config = AttackRunnerConfig(**resolved)

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.watermarker = self._build_watermarker()
        self._ensure_watermarked_image()

    def _build_watermarker(self):
        method = self.config.watermark_method.lower()
        if method == "lsb":
            return LSBWatermarking()
        if method == "fourier":
            return FourierWatermarking()
        if method == "dct":
            return DCTWatermarking()
        if method in {"stable_diffusion", "stable-diffusion", "sd"}:
            return StableDiffusionWatermarking()
        raise ValueError(f"Unsupported watermark method: {self.config.watermark_method}")

    def _ensure_watermarked_image(self) -> None:
        if os.path.exists(self.config.watermarked_image):
            return

        if not self.config.cover_image:
            raise ValueError("cover_image must be provided when generating a watermarked image")

        os.makedirs(os.path.dirname(self.config.watermarked_image), exist_ok=True)

        if isinstance(self.watermarker, LSBWatermarking):
            message = self.config.message or "This is a secret watermark embedded using LSB technique!"
            self.watermarker.embed_watermark(self.config.cover_image, message, self.config.watermarked_image)
        elif isinstance(self.watermarker, DCTWatermarking):
            message = self.config.message or "This is a discrete cosine transform watermark!"
            self.watermarker.embed_watermark(self.config.cover_image, message, self.config.watermarked_image)
        else:
            self.watermarker.embed_watermark(self.config.cover_image, self.config.watermarked_image)

    def _apply_attack(self, image: np.ndarray, attack: Dict[str, Any], name: str) -> str:
        attack_type = attack.get("type")
        output_path = os.path.join(self.config.output_dir, f"{name}.png")

        if attack_type == "compression":
            quality = int(attack.get("quality", 50))
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return output_path

        if attack_type == "rescaling":
            scale = float(attack.get("scale", 0.5))
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, resized)
            return output_path

        if attack_type == "cropping":
            frac = float(attack.get("crop_fraction", 0.8))
            h, w = image.shape[:2]
            new_h, new_w = int(h * frac), int(w * frac)
            start_y = max((h - new_h) // 2, 0)
            start_x = max((w - new_w) // 2, 0)
            cropped = image[start_y : start_y + new_h, start_x : start_x + new_w]
            cv2.imwrite(output_path, cropped)
            return output_path

        if attack_type == "gaussian_noise":
            sigma = float(attack.get("sigma", 5.0))
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(output_path, noisy)
            return output_path

        if attack_type == "gaussian_blur":
            ksize = int(attack.get("ksize", 5))
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
            cv2.imwrite(output_path, blurred)
            return output_path

        raise ValueError(f"Unsupported attack type: {attack_type}")

    def _compute_confidence(self, image_path: str) -> float:
        if isinstance(self.watermarker, FourierWatermarking):
            return self.watermarker.calculate_confidence_score(image_path)

        if isinstance(self.watermarker, LSBWatermarking):
            message = self.config.message or "This is a secret watermark embedded using LSB technique!"
            return self.watermarker.calculate_confidence_score(image_path, message)

        if isinstance(self.watermarker, DCTWatermarking):
            message = self.config.message or "This is a discrete cosine transform watermark!"
            return self.watermarker.calculate_confidence_score(image_path, message)

        if isinstance(self.watermarker, StableDiffusionWatermarking):
            return self.watermarker.calculate_confidence_score(image_path)

        return 0.0

    def _compute_psnr_ssim(self, reference_path: str, attacked_path: str) -> tuple[float, float]:
        ref = cv2.imread(reference_path)
        attacked = cv2.imread(attacked_path)
        if ref is None or attacked is None:
            return 0.0, 0.0

        if ref.shape != attacked.shape:
            attacked = cv2.resize(attacked, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

        psnr = float(cv2.PSNR(ref, attacked))
        ssim = float(structural_similarity(ref, attacked, channel_axis=2))
        return psnr, ssim

    def run(self) -> List[Dict[str, Any]]:
        base_image = cv2.imread(self.config.watermarked_image)
        if base_image is None:
            raise ValueError(f"Could not load watermarked image: {self.config.watermarked_image}")

        results: List[AttackResult] = []
        for attack in self.config.attacks:
            name = attack.get("name") or attack.get("type")
            print(f"Running attack {name} ({attack.get('type')})...")
            output_path = self._apply_attack(base_image, attack, name)
            confidence = float(self._compute_confidence(output_path))
            psnr, ssim = self._compute_psnr_ssim(self.config.watermarked_image, output_path)

            results.append(
                AttackResult(
                    name=name,
                    attack_type=attack.get("type", "unknown"),
                    params={k: v for k, v in attack.items() if k not in {"name", "type"}},
                    confidence=confidence,
                    psnr=psnr,
                    ssim=ssim,
                    output_path=output_path,
                ).__dict__
            )

        if self.config.results_file:
            os.makedirs(os.path.dirname(self.config.results_file), exist_ok=True)
            with open(self.config.results_file, "w") as f:
                json.dump(results, f, indent=2)

        print(f"Completed {len(results)} attacks. Results saved to {self.config.results_file}.")

        return results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run configured watermark attacks")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--suite", type=str, default="lsb_attack_suite")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    suite_config = config["base_config"].get("attack_suites", {}).get(args.suite)
    if suite_config is None:
        raise ValueError(f"Suite {args.suite} not found in config file {args.config}")

    runner = AttackRunner(suite_config)
    runner.run()
    print(f"Finished running attacks for {args.suite}. Results written to {suite_config.get('results_file')}")


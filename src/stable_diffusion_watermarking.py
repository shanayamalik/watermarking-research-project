"""Stable Diffusion-inspired latent watermarking in image space.

This module adapts the ring-mask latent watermarking approach from the
`shai/` experiments to a lightweight image-space routine so it can slot into
`shanaya/` attack suites without requiring a full Stable Diffusion pipeline.
Watermarks are injected in the Fourier domain using a circular ring mask that
mimics the latent-space pattern, then recovered by measuring the boosted energy
in that ring after attacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SD_IMAGE = os.path.abspath(
    os.path.join(BASE_DIR, "..", "alex", "outputs", "sd2.1", "000000006818.png")
)


@dataclass
class StableDiffusionWatermarkConfig:
    ring_ratio: float = 0.7
    ring_width: int = 6
    boost_factor: float = 0.35
    channel: int = 1  # apply watermark to the green channel by default


class StableDiffusionWatermarking:
    """Embed and detect ring-mask Fourier watermarks inspired by SD latents."""

    def __init__(self, config: StableDiffusionWatermarkConfig | None = None):
        self.config = config or StableDiffusionWatermarkConfig()

    def _generate_mask(self, shape: tuple[int, int]) -> np.ndarray:
        rows, cols = shape
        row_indices_c2 = (np.arange(rows) - rows / 2) ** 2
        col_indices_c2 = (np.arange(cols) - cols / 2) ** 2
        dist_grid = row_indices_c2.reshape(-1, 1) + col_indices_c2.reshape(1, -1)
        r = self.config.ring_ratio * rows / 2
        mask = (dist_grid >= r**2) & (dist_grid <= (r + self.config.ring_width) ** 2)
        return mask

    def _load_channel(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        channel = image[:, :, self.config.channel].astype(np.float64)
        return image, channel

    def embed_watermark(self, image_path: str, output_path: str | None = None) -> np.ndarray:
        image, channel = self._load_channel(image_path)
        freq = np.fft.fftshift(np.fft.fft2(channel))
        magnitudes, phases = np.abs(freq), np.angle(freq)

        mask = self._generate_mask(channel.shape)
        baseline = float(np.mean(magnitudes[~mask])) if np.any(~mask) else float(np.mean(magnitudes))
        target_energy = baseline * (1.0 + self.config.boost_factor)
        magnitudes[mask] = target_energy

        watermarked_freq = magnitudes * np.exp(1j * phases)
        watermarked_channel = np.real(np.fft.ifft2(np.fft.ifftshift(watermarked_freq)))
        watermarked_channel = np.clip(watermarked_channel, 0, 255)

        watermarked_image = image.copy().astype(np.float64)
        watermarked_image[:, :, self.config.channel] = watermarked_channel
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, watermarked_image)
            print(f"Stable Diffusion-style watermarked image saved to: {output_path}")

        return watermarked_image

    def calculate_confidence_score(self, watermarked_image_path: str) -> float:
        _, channel = self._load_channel(watermarked_image_path)
        freq = np.fft.fftshift(np.fft.fft2(channel))
        magnitudes = np.abs(freq)
        mask = self._generate_mask(channel.shape)

        masked_mean = float(np.mean(magnitudes[mask]))
        unmasked_mean = float(np.mean(magnitudes[~mask])) if np.any(~mask) else 0.0
        boost_ratio = masked_mean / (unmasked_mean + 1e-6)
        confidence = max(boost_ratio - 1.0, 0.0) / max(self.config.boost_factor, 1e-6)
        return float(min(confidence, 1.0))


if __name__ == "__main__":
    watermarker = StableDiffusionWatermarking()
    cover = DEFAULT_SD_IMAGE
    output = os.path.join(BASE_DIR, "results", "stable_diffusion_watermarked.png")

    if not os.path.exists(cover):
        raise FileNotFoundError(
            f"Default Stable Diffusion cover image not found at {cover}. "
            "Run alex/generate_sd.py or update the config paths."
        )

    print("Embedding Stable Diffusion-style watermark...")
    watermarker.embed_watermark(cover, output)
    score = watermarker.calculate_confidence_score(output)
    print(f"Confidence score on freshly watermarked image: {score:.3f}")

"""
DCT (Discrete Cosine Transform) watermarking implementation.

Embeds watermark bits into mid-frequency DCT coefficients of the luminance
channel to balance invisibility and robustness. Uses a simple pairwise
coefficient comparison for each block to encode bits and can reconstruct the
message using a delimiter sentinel.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_IMAGE = os.path.join(BASE_DIR, "images", "4.2.03.tiff")


@dataclass
class DCTConfig:
    block_size: int = 8
    alpha: float = 8.0  # minimum magnitude separation when enforcing bits
    coeff_pos_one: Tuple[int, int] = (3, 4)
    coeff_pos_zero: Tuple[int, int] = (4, 3)
    delimiter: str = "###END###"


class DCTWatermarking:
    """Block-wise DCT watermarking for single images."""

    def __init__(self, config: DCTConfig | None = None):
        self.config = config or DCTConfig()

    def text_to_binary(self, text: str) -> str:
        binary = "".join(format(ord(char), "08b") for char in text)
        delimiter_binary = "".join(format(ord(char), "08b") for char in self.config.delimiter)
        return binary + delimiter_binary

    def binary_to_text(self, binary: str) -> str:
        delimiter_binary = "".join(format(ord(char), "08b") for char in self.config.delimiter)
        if delimiter_binary in binary:
            binary = binary[: binary.index(delimiter_binary)]

        text = ""
        for i in range(0, len(binary), 8):
            byte = binary[i : i + 8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text

    def _embed_bit(self, block: np.ndarray, bit: str) -> np.ndarray:
        dct_block = cv2.dct(block.astype(np.float32))
        pos_one = self.config.coeff_pos_one
        pos_zero = self.config.coeff_pos_zero
        alpha = self.config.alpha

        diff = dct_block[pos_one] - dct_block[pos_zero]
        if bit == "1":
            if diff <= alpha:
                adjust = (alpha - diff) + 1.0
                dct_block[pos_one] += adjust / 2
                dct_block[pos_zero] -= adjust / 2
        else:
            if diff >= -alpha:
                adjust = (diff + alpha) + 1.0
                dct_block[pos_one] -= adjust / 2
                dct_block[pos_zero] += adjust / 2

        watermarked_block = cv2.idct(dct_block)
        return watermarked_block

    def embed_watermark(self, image_path: str, message: str, output_path: str | None = None) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]

        block_size = self.config.block_size
        h, w = y_channel.shape
        max_blocks = (h // block_size) * (w // block_size)

        binary_message = self.text_to_binary(message)
        if len(binary_message) > max_blocks:
            raise ValueError(
                f"Message too long for image. Capacity {max_blocks} bits, message {len(binary_message)} bits."
            )

        watermarked_y = y_channel.astype(np.float32).copy()
        bit_idx = 0
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                if bit_idx >= len(binary_message):
                    break
                block = watermarked_y[row : row + block_size, col : col + block_size]
                watermarked_block = self._embed_bit(block, binary_message[bit_idx])
                watermarked_y[row : row + block_size, col : col + block_size] = watermarked_block
                bit_idx += 1
            if bit_idx >= len(binary_message):
                break

        ycrcb[:, :, 0] = np.clip(watermarked_y, 0, 255)
        watermarked_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, watermarked_bgr)
            print(f"DCT watermarked image saved to: {output_path}")

        return watermarked_bgr

    def extract_watermark(self, watermarked_image_path: str) -> str:
        image = cv2.imread(watermarked_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {watermarked_image_path}")

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]

        block_size = self.config.block_size
        h, w = y_channel.shape
        delimiter_binary = "".join(format(ord(char), "08b") for char in self.config.delimiter)
        delimiter_length = len(delimiter_binary)
        bits = []

        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = y_channel[row : row + block_size, col : col + block_size]
                dct_block = cv2.dct(block.astype(np.float32))
                diff = dct_block[self.config.coeff_pos_one] - dct_block[self.config.coeff_pos_zero]
                bits.append("1" if diff > 0 else "0")

                if len(bits) >= delimiter_length and "".join(bits[-delimiter_length:]) == delimiter_binary:
                    return self.binary_to_text("".join(bits))

        return self.binary_to_text("".join(bits))

    def calculate_confidence_score(self, watermarked_image_path: str, original_message: str) -> float:
        try:
            extracted = self.extract_watermark(watermarked_image_path)
            if extracted == original_message:
                return 1.0
            if not extracted:
                return 0.0

            min_len = min(len(extracted), len(original_message))
            if min_len == 0:
                return 0.0
            matches = sum(1 for i in range(min_len) if extracted[i] == original_message[i])
            return matches / len(original_message)
        except Exception:
            return 0.0


def load_test_image(image_path: str) -> np.ndarray | None:
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from: {image_path}")
        return None

    height, width, channels = image.shape
    print(f"Loaded image: {os.path.basename(image_path)}")
    print(f"Dimensions: {width}x{height}x{channels}")
    print(f"Total blocks: {(height // 8) * (width // 8)} (capacity in bits)")
    return image


if __name__ == "__main__":
    watermarker = DCTWatermarking()
    test_image = DEFAULT_IMAGE
    message = "This is a discrete cosine transform watermark!"

    original = load_test_image(test_image)
    if original is not None:
        print(f"\nEmbedding message: '{message}'")
        try:
            output_path = "results/mandrill_watermarked_dct.png"
            watermarker.embed_watermark(test_image, message, output_path)

            extracted = watermarker.extract_watermark(output_path)
            print(f"Extracted message: '{extracted}'")

            confidence = watermarker.calculate_confidence_score(output_path, message)
            print(f"DCT watermark confidence: {confidence:.3f}")
        except Exception as exc:
            print(f"Error during DCT watermarking: {exc}")


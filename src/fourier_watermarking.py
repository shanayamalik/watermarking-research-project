"""
Fourier Domain Watermarking Implementation

Implements watermarking in the frequency domain using FFT.
More robust against compression compared to spatial domain methods like LSB.
"""

import numpy as np
import cv2
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_IMAGE = os.path.join(BASE_DIR, "images", "4.2.03.tiff")


class FourierWatermarking:
    """Fourier domain watermarking implementation."""
    
    def __init__(self):
        pass
    
    def embed_watermark(self, image_path, output_path=None):
        """
        Embed a watermark in the Fourier domain while preserving color (fast version).
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Work only on the green channel (most sensitive to human vision)
        # but preserve all color channels in output
        watermarked_image = image.copy().astype(np.float64)
        
        # Apply FFT only to green channel
        green_channel = watermarked_image[:, :, 1]
        fourier_transform = np.fft.fftshift(np.fft.fft2(green_channel))
        
        # Add small watermark in center region
        h, w = fourier_transform.shape
        center_h, center_w = h // 2, w // 2
        watermark_size = 10  # Smaller for speed
        
        # Add very subtle watermark
        fourier_transform[center_h-watermark_size:center_h+watermark_size, 
                         center_w-watermark_size:center_w+watermark_size] += 100  # Much smaller
        
        # Convert back and update only green channel
        watermarked_image[:, :, 1] = np.abs(np.fft.ifft2(fourier_transform))
        
        # Convert back to uint8 and ensure proper range
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        
        # Save watermarked image
        if output_path:
            cv2.imwrite(output_path, watermarked_image)
            print(f"Fourier watermarked image saved to: {output_path}")
        
        return watermarked_image
    
    def calculate_confidence_score(self, watermarked_image_path):
        """
        Calculate a normalized confidence score (0-1) - fast version.
        """
        try:
            # Load image and get green channel
            image = cv2.imread(watermarked_image_path)
            if image is None:
                return 0.0
            
            green_channel = image[:, :, 1]
            
            # Quick FFT on green channel only
            fourier_transform = np.fft.fftshift(np.fft.fft2(green_channel))
            
            # Check small watermark region
            h, w = fourier_transform.shape
            center_h, center_w = h // 2, w // 2
            watermark_size = 10
            
            # Calculate average energy in small region
            watermark_region = fourier_transform[center_h-watermark_size:center_h+watermark_size, 
                                               center_w-watermark_size:center_w+watermark_size]
            average_energy = np.mean(np.abs(watermark_region))
            
            # Simple normalization
            confidence_score = min(average_energy / 10000, 1.0)
            return confidence_score
            
        except Exception as e:
            print(f"Error calculating confidence score: {e}")
            return 0.0


if __name__ == "__main__":
    # Test Fourier watermarking
    watermarker = FourierWatermarking()
    test_image = DEFAULT_IMAGE
    
    if os.path.exists(test_image):
        print("Testing Fourier domain watermarking...")
        
        # Embed watermark
        watermarker.embed_watermark(test_image, "results/mandrill_watermarked_fourier.png")
        
        # Test confidence score
        score = watermarker.calculate_confidence_score("results/mandrill_watermarked_fourier.png")
        print(f"Confidence score: {score:.3f}")
        
    else:
        print(f"Test image not found: {test_image}")

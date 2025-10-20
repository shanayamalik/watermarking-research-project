"""
DCT-based Watermarking Implementation

This module implements invisible watermarking using DCT (Discrete Cosine Transform)
in the frequency domain. DCT watermarking embeds a pseudorandom signal into 
mid-frequency coefficients, making it more robust to compression and noise
while maintaining good imperceptibility with proper alpha tuning.
"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os


class DCTWatermarking:
    """DCT watermarking implementation for frequency domain embedding."""
    
    def __init__(self, alpha=0.1, block_size=8):
        """
        Initialize DCT watermarking.
        
        Args:
            alpha (float): Watermark strength parameter (0.01-1.0)
            block_size (int): Size of DCT blocks (typically 8x8)
        """
        self.alpha = alpha
        self.block_size = block_size
        self.seed = 42  # For reproducible pseudorandom sequences
    
    def generate_watermark_pattern(self, height, width):
        """Generate pseudorandom watermark pattern."""
        np.random.seed(self.seed)
        # Generate pattern with values in [-1, 1]
        pattern = np.random.randn(height, width)
        # Normalize to [-1, 1] range
        pattern = pattern / np.std(pattern)
        return pattern
    
    def dct2(self, block):
        """2D DCT transform."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def idct2(self, block):
        """2D inverse DCT transform."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def embed_watermark_block(self, dct_block, watermark_block):
        """
        Embed watermark in a single DCT block using mid-frequency coefficients.
        
        Args:
            dct_block: 8x8 DCT coefficients
            watermark_block: 8x8 watermark pattern
            
        Returns:
            Watermarked DCT block
        """
        watermarked_block = dct_block.copy()
        
        # Define mid-frequency coefficients (avoid DC and high frequencies)
        # Using zigzag pattern coefficients 10-50 (mid-frequency range)
        mid_freq_positions = [
            (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0),
            (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2),
            (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5),
            (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1),
            (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7)
        ]
        
        # Embed watermark in mid-frequency coefficients
        for i, (row, col) in enumerate(mid_freq_positions[:20]):  # Use first 20 coefficients
            if row < self.block_size and col < self.block_size:
                # Additive embedding: modify coefficient by alpha * watermark_value
                watermarked_block[row, col] += self.alpha * watermark_block[row, col]
        
        return watermarked_block
    
    def extract_watermark_block(self, original_dct, watermarked_dct):
        """Extract watermark from a single DCT block."""
        # Same mid-frequency positions as embedding
        mid_freq_positions = [
            (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0),
            (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2),
            (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5),
            (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1),
            (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7)
        ]
        
        extracted_block = np.zeros((self.block_size, self.block_size))
        
        for i, (row, col) in enumerate(mid_freq_positions[:20]):
            if row < self.block_size and col < self.block_size:
                # Extract watermark: (watermarked - original) / alpha
                extracted_block[row, col] = (watermarked_dct[row, col] - original_dct[row, col]) / self.alpha
        
        return extracted_block
    
    def embed_watermark(self, image_path, output_path=None):
        """
        Embed watermark into image using DCT.
        
        Args:
            image_path (str): Path to original image
            output_path (str): Path to save watermarked image
            
        Returns:
            tuple: (watermarked_image, watermark_pattern)
        """
        # Load image
        if image_path.lower().endswith(('.tiff', '.tif')):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to YUV color space (embed in Y channel - luminance)
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv_image[:, :, 0].astype(np.float32)
        
        # Get image dimensions
        height, width = y_channel.shape
        
        # Ensure dimensions are multiples of block_size
        new_height = (height // self.block_size) * self.block_size
        new_width = (width // self.block_size) * self.block_size
        y_channel = y_channel[:new_height, :new_width]
        
        # Generate watermark pattern
        watermark_pattern = self.generate_watermark_pattern(new_height, new_width)
        
        # Process image in blocks
        watermarked_y = y_channel.copy()
        
        for i in range(0, new_height, self.block_size):
            for j in range(0, new_width, self.block_size):
                # Extract blocks
                image_block = y_channel[i:i+self.block_size, j:j+self.block_size]
                watermark_block = watermark_pattern[i:i+self.block_size, j:j+self.block_size]
                
                # Apply DCT
                dct_block = self.dct2(image_block)
                
                # Embed watermark
                watermarked_dct = self.embed_watermark_block(dct_block, watermark_block)
                
                # Apply inverse DCT
                watermarked_block = self.idct2(watermarked_dct)
                
                # Clip values to valid range
                watermarked_block = np.clip(watermarked_block, 0, 255)
                
                # Store watermarked block
                watermarked_y[i:i+self.block_size, j:j+self.block_size] = watermarked_block
        
        # Reconstruct full image
        watermarked_yuv = yuv_image.copy()
        watermarked_yuv[:new_height, :new_width, 0] = watermarked_y.astype(np.uint8)
        
        # Convert back to BGR
        watermarked_image = cv2.cvtColor(watermarked_yuv, cv2.COLOR_YUV2BGR)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, watermarked_image)
            print(f"DCT watermarked image saved to: {output_path}")
        
        return watermarked_image, watermark_pattern
    
    def detect_watermark(self, original_path, watermarked_path):
        """
        Detect and measure watermark strength using correlation.
        
        Args:
            original_path (str): Path to original image
            watermarked_path (str): Path to watermarked image
            
        Returns:
            float: Correlation coefficient (higher = stronger watermark detection)
        """
        # Load images
        original = cv2.imread(original_path)
        watermarked = cv2.imread(watermarked_path)
        
        if original is None or watermarked is None:
            raise ValueError("Could not load images")
        
        # Convert to YUV and extract Y channels
        orig_yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
        wm_yuv = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YUV)
        
        orig_y = orig_yuv[:, :, 0].astype(np.float32)
        wm_y = wm_yuv[:, :, 0].astype(np.float32)
        
        # Ensure same dimensions
        height, width = min(orig_y.shape[0], wm_y.shape[0]), min(orig_y.shape[1], wm_y.shape[1])
        new_height = (height // self.block_size) * self.block_size
        new_width = (width // self.block_size) * self.block_size
        
        orig_y = orig_y[:new_height, :new_width]
        wm_y = wm_y[:new_height, :new_width]
        
        # Generate expected watermark pattern
        expected_pattern = self.generate_watermark_pattern(new_height, new_width)
        
        # Extract watermark from DCT coefficients
        extracted_watermark = np.zeros((new_height, new_width))
        
        for i in range(0, new_height, self.block_size):
            for j in range(0, new_width, self.block_size):
                # Extract blocks
                orig_block = orig_y[i:i+self.block_size, j:j+self.block_size]
                wm_block = wm_y[i:i+self.block_size, j:j+self.block_size]
                
                # Apply DCT
                orig_dct = self.dct2(orig_block)
                wm_dct = self.dct2(wm_block)
                
                # Extract watermark
                extracted_block = self.extract_watermark_block(orig_dct, wm_dct)
                extracted_watermark[i:i+self.block_size, j:j+self.block_size] = extracted_block
        
        # Calculate correlation with expected pattern
        correlation = np.corrcoef(expected_pattern.flatten(), extracted_watermark.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0


def calculate_metrics(original, watermarked):
    """Calculate PSNR and MSE."""
    mse = np.mean((original.astype(float) - watermarked.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr, mse


if __name__ == "__main__":
    # Test DCT watermarking
    print("DCT Watermarking Test")
    print("=" * 50)
    
    # Initialize watermarker with different alpha values to test
    alpha_values = [0.05, 0.1, 0.2]
    
    original_path = "images/4.2.03.tiff"
    
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        exit(1)
    
    original = cv2.imread(original_path)
    
    for alpha in alpha_values:
        print(f"\nTesting with alpha = {alpha}")
        print("-" * 30)
        
        # Create watermarker
        watermarker = DCTWatermarking(alpha=alpha)
        
        # Embed watermark
        output_path = f"results/mandrill_dct_alpha_{alpha}.png"
        watermarked, pattern = watermarker.embed_watermark(original_path, output_path)
        
        # Calculate imperceptibility metrics
        psnr, mse = calculate_metrics(original, watermarked)
        print(f"PSNR: {psnr:.2f} dB")
        print(f"MSE: {mse:.6f}")
        
        # Test watermark detection
        correlation = watermarker.detect_watermark(original_path, output_path)
        print(f"Watermark correlation: {correlation:.4f}")
        
        # Interpretation
        if correlation > 0.5:
            print("✅ Strong watermark detected")
        elif correlation > 0.3:
            print("⚠️  Moderate watermark detected")
        elif correlation > 0.1:
            print("⚠️  Weak watermark detected")
        else:
            print("❌ No clear watermark detected")
        
        if psnr > 40:
            print("✅ Excellent imperceptibility")
        elif psnr > 30:
            print("✅ Good imperceptibility")
        else:
            print("⚠️  Noticeable quality degradation")
    
    print(f"\n✅ DCT watermarking test complete!")
    print(f"Check results/ folder for watermarked images with different alpha values.")
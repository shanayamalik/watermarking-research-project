# TODO: update documentation & inline comments

"""
LSB Watermarking Visualization and Analysis Tool

This script helps visualize the differences between original and watermarked images,
and proves that the watermark exists even though it's invisible to the human eye.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lsb_watermarking import LSBWatermarking


def calculate_metrics(original, watermarked):
    """Calculate PSNR and MSE between original and watermarked images."""
    
    # Ensure images are the same size
    if original.shape != watermarked.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original.astype(float) - watermarked.astype(float)) ** 2)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr, mse


def visualize_differences(original, watermarked, save_path="results/"):
    """Create visualizations showing the differences between images."""
    
    # Calculate absolute difference
    diff = np.abs(original.astype(int) - watermarked.astype(int))
    
    # Amplify differences for visualization (multiply by 50)
    diff_amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('LSB Watermarking Analysis', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Watermarked image
    axes[0, 1].imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Watermarked Image')
    axes[0, 1].axis('off')
    
    # Difference (amplified)
    axes[0, 2].imshow(diff_amplified, cmap='hot')
    axes[0, 2].set_title('Differences (50x amplified)')
    axes[0, 2].axis('off')
    
    # LSB visualization for each channel
    channels = ['Blue', 'Green', 'Red']
    for i in range(3):
        lsb_original = original[:, :, i] & 1  # Extract LSB
        lsb_watermarked = watermarked[:, :, i] & 1  # Extract LSB
        lsb_diff = lsb_original ^ lsb_watermarked  # XOR to show changes
        
        axes[1, i].imshow(lsb_diff * 255, cmap='gray')
        axes[1, i].set_title(f'{channels[i]} Channel LSB Changes')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lsb_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


def analyze_lsb_changes(original, watermarked):
    """Analyze and report statistics about LSB changes."""
    
    print("\n" + "="*50)
    print("LSB WATERMARKING ANALYSIS")
    print("="*50)
    
    # Calculate metrics
    psnr, mse = calculate_metrics(original, watermarked)
    
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    
    # Analyze pixel differences
    diff = np.abs(original.astype(int) - watermarked.astype(int))
    changed_pixels = np.sum(diff > 0)
    total_pixels = original.shape[0] * original.shape[1] * original.shape[2]
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Changed pixels: {changed_pixels:,}")
    print(f"Percentage changed: {(changed_pixels/total_pixels)*100:.2f}%")
    
    # Analyze magnitude of changes
    max_change = np.max(diff)
    avg_change = np.mean(diff[diff > 0]) if changed_pixels > 0 else 0
    
    print(f"Maximum pixel change: {max_change}")
    print(f"Average change (non-zero): {avg_change:.2f}")
    
    # PSNR interpretation
    print(f"\nPSNR Interpretation:")
    if psnr > 40:
        print("✅ Excellent quality (imperceptible)")
    elif psnr > 30:
        print("✅ Good quality (barely perceptible)")
    elif psnr > 20:
        print("⚠️  Fair quality (perceptible but acceptable)")
    else:
        print("❌ Poor quality (clearly visible)")


def extract_and_verify(watermarked_path):
    """Extract message and verify watermark exists."""
    
    print("\n" + "="*50)
    print("WATERMARK EXTRACTION VERIFICATION")
    print("="*50)
    
    watermarker = LSBWatermarking()
    
    try:
        extracted_message = watermarker.extract_watermark(watermarked_path)
        print(f"Extracted message: '{extracted_message}'")
        
        if extracted_message.strip():
            print("✅ Watermark successfully extracted!")
            print("✅ This proves the watermark exists in the image")
        else:
            print("❌ No watermark found")
            
    except Exception as e:
        print(f"❌ Error extracting watermark: {e}")


if __name__ == "__main__":
    # Paths
    original_path = "images/4.2.03.tiff"
    watermarked_path = "results/mandrill_watermarked_lsb.png"
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        exit(1)
        
    if not os.path.exists(watermarked_path):
        print(f"Watermarked image not found: {watermarked_path}")
        print("Please run lsb_watermarking.py first to create the watermarked image")
        exit(1)
    
    # Load images
    original = cv2.imread(original_path)
    watermarked = cv2.imread(watermarked_path)
    
    if original is None or watermarked is None:
        print("Error loading images")
        exit(1)
    
    # Resize watermarked to match original if needed
    if original.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))
    
    print("Analyzing LSB watermarking...")
    
    # Perform analysis
    analyze_lsb_changes(original, watermarked)
    
    # Extract and verify watermark
    extract_and_verify(watermarked_path)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    visualize_differences(original, watermarked)
    
    print(f"\n✅ Analysis complete! Check 'results/lsb_analysis.png' for visual comparison.")
"""
Visual Comparison of LSB vs DCT Watermarking
Shows both methods are invisible but work differently
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lsb_watermarking import LSBWatermarking
from dct_watermarking import DCTWatermarking


def create_comparison_visualization():
    """Create side-by-side comparison showing both methods are invisible."""
    
    original_path = "images/4.2.03.tiff"
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        return
    
    # Load original
    original = cv2.imread(original_path)
    
    # Create watermarked versions
    print("Creating watermarked images...")
    
    # LSB watermarking
    lsb_watermarker = LSBWatermarking()
    lsb_watermarked = lsb_watermarker.embed_watermark(
        original_path, 
        "Secret LSB message!", 
        "results/comparison_lsb.png"
    )
    
    # DCT watermarking
    dct_watermarker = DCTWatermarking(alpha=0.1)
    dct_watermarked, _ = dct_watermarker.embed_watermark(
        original_path, 
        "results/comparison_dct.png"
    )
    
    # Calculate differences (amplified for visualization)
    lsb_diff = np.abs(original.astype(int) - lsb_watermarked.astype(int))
    dct_diff = np.abs(original.astype(int) - dct_watermarked.astype(int))
    
    # Amplify differences
    lsb_diff_amp = np.clip(lsb_diff * 50, 0, 255).astype(np.uint8)
    dct_diff_amp = np.clip(dct_diff * 50, 0, 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('LSB vs DCT Watermarking - Both Are Invisible!', fontsize=16)
    
    # Row 1: LSB
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(lsb_watermarked, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('LSB Watermarked\n(Looks identical!)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(lsb_diff_amp, cmap='hot')
    axes[0, 2].set_title('LSB Differences\n(50x amplified)')
    axes[0, 2].axis('off')
    
    # Row 2: DCT
    axes[1, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(dct_watermarked, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('DCT Watermarked\n(Also looks identical!)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(dct_diff_amp, cmap='hot')
    axes[1, 2].set_title('DCT Differences\n(50x amplified)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/lsb_vs_dct_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test detection methods
    print("\n" + "="*60)
    print("WATERMARK DETECTION COMPARISON")
    print("="*60)
    
    # LSB Detection (extract message)
    print("\nLSB Detection (Message Extraction):")
    try:
        extracted_lsb = lsb_watermarker.extract_watermark("results/comparison_lsb.png")
        print(f"✅ Extracted message: '{extracted_lsb}'")
        print("✅ LSB watermark confirmed by message extraction")
    except Exception as e:
        print(f"❌ LSB extraction failed: {e}")
    
    # DCT Detection (correlation)
    print("\nDCT Detection (Statistical Correlation):")
    try:
        correlation = dct_watermarker.detect_watermark(original_path, "results/comparison_dct.png")
        print(f"Correlation coefficient: {correlation:.4f}")
        if correlation > 0.3:
            print("✅ DCT watermark confirmed by high correlation")
        else:
            print("❌ DCT watermark not detected (low correlation)")
    except Exception as e:
        print(f"❌ DCT detection failed: {e}")
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"Both LSB and DCT watermarks are INVISIBLE to human eyes!")
    print(f"The difference is in HOW we detect them:")
    print(f"• LSB: Extract hidden text message")
    print(f"• DCT: Measure statistical correlation with known pattern")


if __name__ == "__main__":
    create_comparison_visualization()
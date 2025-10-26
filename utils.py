import cv2
import os
import sys
sys.path.append('src')
from lsb_watermarking import LSBWatermarking
from fourier_watermarking import FourierWatermarking

def run_fourier_compression_test(config):
    """Run compression robustness test for Fourier watermarking."""
    original_image = config.get("original_image", "images/4.2.03.tiff")
    fourier_watermarked_image = config.get("fourier_watermarked_image", "results/mandrill_watermarked_fourier.png")
    quality = config.get("compression_quality", 50)
    
    if not os.path.exists(original_image):
        print(f"Original image not found: {original_image}")
        return 0.0
    
    watermarker = FourierWatermarking()
    
    # Create watermarked image if it doesn't exist
    if not os.path.exists(fourier_watermarked_image):
        watermarker.embed_watermark(original_image, fourier_watermarked_image)
    
    # Apply compression to watermarked image
    image = cv2.imread(fourier_watermarked_image)
    compressed_path = "results/temp_fourier_compressed.jpg"
    cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    # Test watermark detection from compressed image
    confidence_score = watermarker.calculate_confidence_score(compressed_path)    # Clean up
    if os.path.exists(compressed_path):
        os.remove(compressed_path)
    
    # Return confidence score (higher = better survival)
    return confidence_score

def run_compression_test(config):
    """Run compression robustness test for LSB watermarking."""
    watermarked_image = config.get("watermarked_image", "results/mandrill_watermarked_lsb.png")
    original_message = config.get("original_message", "This is a secret watermark embedded using LSB technique!")
    quality = config.get("compression_quality", 50)
    
    if not os.path.exists(watermarked_image):
        print(f"Watermarked image not found: {watermarked_image}")
        return 0.0
    
    watermarker = LSBWatermarking()
    
    # Apply compression
    image = cv2.imread(watermarked_image)
    compressed_path = "results/temp_compressed.jpg"
    cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    # Calculate confidence score for extraction
    confidence = watermarker.calculate_confidence_score(compressed_path, original_message)
    
    # Clean up
    if os.path.exists(compressed_path):
        os.remove(compressed_path)
    
    return confidence

def run_metric(metric, config):
    """Run a specific metric and return the score."""
    
    # Image quality metrics
    if metric == "fid":
        # TODO: Install cleanfid package
        # return fid.compute_fid(config["dir1"], config["dir2"])
        return 0.0
    
    elif metric == "clip":
        # TODO: Implement CLIP score
        return 0.0
    
    elif metric == "IS":
        # TODO: Implement Inception Score
        return 0.0
    
    # Detectability metrics
    elif metric == "binary_classifier":
        # TODO: Implement binary classifier
        return 0.0
    
    # Perceptual variability metrics
    elif metric == "lpips":
        # TODO: Implement LPIPS
        return 0.0
    
    # Robustness tests
    elif metric == "cropping":
        # TODO: Implement cropping robustness
        return 0.0
    
    elif metric == "rescaling":
        # TODO: Implement rescaling robustness
        return 0.0
    
    elif metric == "compression":
        # Our LSB compression test
        return run_compression_test(config)
    
    elif metric == "fourier_compression":
        # Our Fourier compression test
        return run_fourier_compression_test(config)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
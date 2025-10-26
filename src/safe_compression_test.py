"""
Safe Compression Test - with timeout protection
"""

import cv2
from lsb_watermarking import LSBWatermarking

def safe_extract_watermark(watermarker, image_path, max_pixels=50000):
    """Extract watermark with a pixel limit to prevent infinite loops."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        flat_image = image.flatten()
        binary_message = ''
        delimiter_binary = ''.join(format(ord(char), '08b') for char in watermarker.delimiter)
        
        # Limit the search to prevent hanging
        search_limit = min(len(flat_image), max_pixels)
        
        for i in range(search_limit):
            binary_message += str(flat_image[i] % 2)
            if delimiter_binary in binary_message:
                return watermarker.binary_to_text(binary_message)
        
        # If delimiter not found in the limit, return partial result
        return watermarker.binary_to_text(binary_message)
        
    except Exception as e:
        return None

# Test
watermarker = LSBWatermarking()
message = "This is a secret watermark embedded using LSB technique!"

print("Testing compression attack (with safety limits)...")

# Compress the watermarked image
image = cv2.imread("results/mandrill_watermarked_lsb.png")
cv2.imwrite("results/test_compressed.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])
print("Image compressed.")

# Try to extract with safety limit
extracted = safe_extract_watermark(watermarker, "results/test_compressed.jpg")

if extracted:
    success = extracted == message
    print(f"Extracted: '{extracted[:50]}{'...' if len(extracted) > 50 else ''}'")
    print(f"Result: {'PASSED' if success else 'FAILED'}")
else:
    print("FAILED - Could not extract watermark")

print("Test complete - as expected, LSB is fragile against compression.")
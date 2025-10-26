"""
LSB (Least Significant Bit) Watermarking Implementation

Implements invisible watermarking using LSB technique in the spatial domain.
Embeds messages by replacing the least significant bits of pixel values.
"""

import numpy as np
import cv2
from PIL import Image
import os


class LSBWatermarking:
    """LSB watermarking for spatial domain embedding."""
    
    def __init__(self):
        self.delimiter = "###END###"
    
    def text_to_binary(self, text):
        """Convert text to binary string with delimiter."""
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary + ''.join(format(ord(char), '08b') for char in self.delimiter)
    
    def binary_to_text(self, binary):
        """Convert binary string back to text."""
        delimiter_binary = ''.join(format(ord(char), '08b') for char in self.delimiter)
        if delimiter_binary in binary:
            binary = binary[:binary.index(delimiter_binary)]
        
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text
    
    def embed_watermark(self, image_path, message, output_path=None):
        """Embed a text message into an image using LSB technique."""
        # Load image
        if image_path.lower().endswith(('.tiff', '.tif')):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        binary_message = self.text_to_binary(message)
        message_length = len(binary_message)
        
        # Check capacity
        total_pixels = image.shape[0] * image.shape[1] * image.shape[2]
        if message_length > total_pixels:
            raise ValueError(f"Message too long. Image can hold {total_pixels} bits, message is {message_length} bits.")
        
        watermarked_image = image.copy()
        flat_image = watermarked_image.flatten()
        
        # Embed message bits
        for i in range(message_length):
            pixel_value = flat_image[i]
            message_bit = int(binary_message[i])
            
            # Modify LSB with boundary checks
            if pixel_value % 2 != message_bit:
                if pixel_value == 255 and message_bit == 1:
                    flat_image[i] = 254
                elif pixel_value == 0 and message_bit == 1:
                    flat_image[i] = 1
                else:
                    flat_image[i] = pixel_value + (1 if message_bit == 1 else -1)
        
        watermarked_image = flat_image.reshape(image.shape)
        
        if output_path:
            cv2.imwrite(output_path, watermarked_image)
            print(f"Watermarked image saved to: {output_path}")
        
        return watermarked_image
    
    def extract_watermark(self, watermarked_image_path):
        """Extract embedded message from watermarked image."""
        # Load watermarked image
        if watermarked_image_path.lower().endswith(('.tiff', '.tif')):
            image = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(watermarked_image_path)
            
        if image is None:
            raise ValueError(f"Could not load image from {watermarked_image_path}")
        
        flat_image = image.flatten()
        binary_message = ''
        delimiter_binary = ''.join(format(ord(char), '08b') for char in self.delimiter)
        
        for pixel_value in flat_image:
            binary_message += str(pixel_value % 2)
            if delimiter_binary in binary_message:
                break
        
        return self.binary_to_text(binary_message)


def load_test_image(image_path):
    """Load and display test image information."""
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        print("Please download mandrill.tiff from USC SIPI database and place it in the images/ folder")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from: {image_path}")
        return None
    
    height, width, channels = image.shape
    print(f"Loaded image: {os.path.basename(image_path)}")
    print(f"Dimensions: {width}x{height}x{channels}")
    print(f"Total pixels: {width * height}")
    print(f"Max message capacity: {width * height * channels} bits")
    
    return image


if __name__ == "__main__":
    watermarker = LSBWatermarking()
    test_image = "images/4.2.03.tiff"
    
    original = load_test_image(test_image)
    if original is not None:
        message = "This is a secret watermark embedded using LSB technique!"
        print(f"\nEmbedding message: '{message}'")
        
        try:
            watermarked = watermarker.embed_watermark(
                test_image, 
                message, 
                "results/mandrill_watermarked_lsb.png"
            )
            
            extracted = watermarker.extract_watermark("results/mandrill_watermarked_lsb.png")
            print(f"Extracted message: '{extracted}'")
            
            if extracted == message:
                print("LSB watermarking successful.")
            else:
                print("LSB watermarking unsuccessful.")
                
        except Exception as e:
            print(f"Error: {e}")
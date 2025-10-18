# TODO: update documentation & inline comments

"""
LSB (Least Significant Bit) Watermarking Implementation

This module implements invisible watermarking using the LSB technique in the spatial domain.
LSB watermarking embeds a message by replacing the least significant bits of pixel values,
making it visually imperceptible but fragile to compression and modifications.
"""

import numpy as np
import cv2
from PIL import Image
import os


class LSBWatermarking:
    """LSB watermarking implementation for spatial domain embedding."""
    
    def __init__(self):
        self.delimiter = "###END###"  # Delimiter to mark end of message
    
    def text_to_binary(self, text):
        """Convert text to binary string."""
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary + ''.join(format(ord(char), '08b') for char in self.delimiter)
    
    def binary_to_text(self, binary):
        """Convert binary string back to text."""
        # Remove delimiter and everything after it
        delimiter_binary = ''.join(format(ord(char), '08b') for char in self.delimiter)
        if delimiter_binary in binary:
            binary = binary[:binary.index(delimiter_binary)]
        
        # Convert binary to text
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text
    
    def embed_watermark(self, image_path, message, output_path=None):
        """
        Embed a text message into an image using LSB technique.
        
        Args:
            image_path (str): Path to the original image
            message (str): Text message to embed
            output_path (str): Path to save watermarked image (optional)
            
        Returns:
            numpy.ndarray: Watermarked image array
        """
        # Load image
        if image_path.lower().endswith('.tiff') or image_path.lower().endswith('.tif'):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert message to binary
        binary_message = self.text_to_binary(message)
        message_length = len(binary_message)
        
        # Check if image can hold the message
        total_pixels = image.shape[0] * image.shape[1] * image.shape[2]
        if message_length > total_pixels:
            raise ValueError(f"Message too long. Image can hold {total_pixels} bits, message is {message_length} bits.")
        
        # Create a copy of the image
        watermarked_image = image.copy()
        
        # Flatten image for easier bit manipulation
        flat_image = watermarked_image.flatten()
        
        # Embed message bits
        for i in range(message_length):
            # Get the current pixel value
            pixel_value = flat_image[i]
            
            # Get the message bit
            message_bit = int(binary_message[i])
            
            # Modify LSB
            if pixel_value % 2 != message_bit:
                if pixel_value == 255 and message_bit == 1:
                    flat_image[i] = 254  # Avoid overflow
                elif pixel_value == 0 and message_bit == 1:
                    flat_image[i] = 1
                else:
                    flat_image[i] = pixel_value + (1 if message_bit == 1 else -1)
            
        # Reshape back to original image shape
        watermarked_image = flat_image.reshape(image.shape)
        
        # Save watermarked image if output path provided
        if output_path:
            cv2.imwrite(output_path, watermarked_image)
            print(f"Watermarked image saved to: {output_path}")
        
        return watermarked_image
    
    def extract_watermark(self, watermarked_image_path):
        """
        Extract embedded message from watermarked image.
        
        Args:
            watermarked_image_path (str): Path to the watermarked image
            
        Returns:
            str: Extracted message
        """
        # Load watermarked image
        if watermarked_image_path.lower().endswith('.tiff') or watermarked_image_path.lower().endswith('.tif'):
            image = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(watermarked_image_path)
            
        if image is None:
            raise ValueError(f"Could not load image from {watermarked_image_path}")
        
        # Flatten image
        flat_image = image.flatten()
        
        # Extract bits
        binary_message = ''
        delimiter_binary = ''.join(format(ord(char), '08b') for char in self.delimiter)
        
        for pixel_value in flat_image:
            # Extract LSB
            binary_message += str(pixel_value % 2)
            
            # Check if we've found the delimiter
            if delimiter_binary in binary_message:
                break
        
        # Convert binary to text
        extracted_message = self.binary_to_text(binary_message)
        return extracted_message


def load_test_image(image_path):
    """Load and display information about test image."""
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
    # Test the LSB watermarking
    watermarker = LSBWatermarking()
    
    # Test image path
    test_image = "images/4.2.03.tiff" # TODO: remove hardcoded value
    
    # Load test image
    original = load_test_image(test_image)
    if original is not None:
        # Test message
        message = "This is a secret watermark embedded using LSB technique!"
        print(f"\nEmbedding message: '{message}'")
        
        # Embed watermark
        try:
            watermarked = watermarker.embed_watermark(
                test_image, 
                message, 
                "results/mandrill_watermarked_lsb.png"
            )
            
            # Extract watermark
            extracted = watermarker.extract_watermark("results/mandrill_watermarked_lsb.png")
            print(f"Extracted message: '{extracted}'")
            
            # Check if extraction was successful
            if extracted == message:
                print("✅ LSB watermarking successful!")
            else:
                print("❌ LSB watermarking failed!")
                
        except Exception as e:
            print(f"Error: {e}")
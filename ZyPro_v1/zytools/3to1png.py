import cv2
import numpy as np
import os

def process_image(image_path, output_dir):
    # Read image
    image = cv2.imread(image_path)

    # Extract red channel
    red_channel = image[:,:,2]

    # Binarization: set all values greater than 0 to 255
    binary_red = np.where(red_channel > 0, 255, 0).astype(np.uint8)

    # Build output file path
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{base_name}.png")

    # Save as single channel png format
    cv2.imwrite(output_path, binary_red)

def batch_process(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of supported file extensions
    extensions = ['.jpg', '.jpeg', '.png']

    # Process all files in input directory
    for filename in os.listdir(input_dir):
        if os.path.splitext(filename)[1].lower() in extensions:
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)

input_dir = 'D:/***'  # Replace with your input image directory path
output_dir = 'D:/***' # Replace with your output image directory path
batch_process(input_dir, output_dir)

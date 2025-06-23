import os
from PIL import Image

def tif_to_jpg(source_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                # Open tif file
                image_path = os.path.join(root, file)
                image = Image.open(image_path)

                # Save as jpg format
                jpg_filename = os.path.splitext(file)[0] + ".jpg"
                jpg_path = os.path.join(root, jpg_filename)
                image.save(jpg_path, "JPEG")

source_directory = r'D:/***'
tif_to_jpg(source_directory)
print("Conversion completed!")

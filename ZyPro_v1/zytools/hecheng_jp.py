from PIL import Image
import os


def stitch_images_in_folder(folder_path, output_path):
    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter files that match the naming format
    image_files = [f for f in all_files if f.startswith('t02_') and f.endswith('.jpg')]

    # Get max x, y values to determine output image size
    max_x, max_y = 0, 0
    for image_file in image_files:
        x, y = map(int, image_file.split('.')[0].split('_')[1:])
        max_x = max(max_x, x)
        max_y = max(max_y, y)


    # Use first image size to calculate output image size
    sample_image = Image.open(os.path.join(folder_path, 't02_0_0.jpg'))
    width, height = sample_image.size
    total_width = width * (max_x + 1)
    total_height = height * (max_y + 1)

    # Create a blank image for stitching
    stitched_image = Image.new('L', (total_width, total_height))

    for image_file in image_files:
        x, y = map(int, image_file.split('.')[0].split('_')[1:])
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        stitched_image.paste(image, (x * width, y * height))

    stitched_image.save(output_path)


# Example usage
folder_path = r'D:/***'
output_path = r'D:/***'
stitch_images_in_folder(folder_path, output_path)

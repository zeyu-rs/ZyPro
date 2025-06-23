from PIL import Image
import os

def augment_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # You can add other image formats as needed
            image_path = os.path.join(folder_path, filename)

            # Open original image
            image = Image.open(image_path)

            # Vertical flip
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_name = "flipped_" + filename
            flipped_image.save(os.path.join(folder_path, flipped_name))

            # Rotate 90 degrees
            rotated_image = image.rotate(-90)  # Rotate 90 degrees counterclockwise
            rotated_name = "rotated_" + filename
            rotated_image.save(os.path.join(folder_path, rotated_name))

# Process 3-channel image folder
three_channel_folder = 'D:/***/'
augment_images(three_channel_folder)

# Process single-channel image folder
single_channel_folder = 'D:/***/'
augment_images(single_channel_folder)

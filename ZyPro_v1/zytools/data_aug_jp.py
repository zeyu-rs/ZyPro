from PIL import Image
import os

def augment_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 您可以根据需要增加其他图像格式
            image_path = os.path.join(folder_path, filename)

            # 打开原始图像
            image = Image.open(image_path)

            # 垂直镜像
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_name = "flipped_" + filename
            flipped_image.save(os.path.join(folder_path, flipped_name))

            # 旋转90度
            rotated_image = image.rotate(-90)  # 逆时针旋转90度
            rotated_name = "rotated_" + filename
            rotated_image.save(os.path.join(folder_path, rotated_name))

# 处理3通道的图像文件夹
three_channel_folder = 'E:/aafa/zypro/data/image01/'
augment_images(three_channel_folder)

# 处理单通道的图像文件夹
single_channel_folder =  'E:/aafa/zypro/data/label01/'
augment_images(single_channel_folder)

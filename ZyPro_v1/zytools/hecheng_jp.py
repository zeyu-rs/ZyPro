from PIL import Image
import os


def stitch_images_in_folder(folder_path, output_path):
    # 获取文件夹中的所有文件
    all_files = os.listdir(folder_path)

    # 过滤出符合命名格式的文件
    image_files = [f for f in all_files if f.startswith('t02_') and f.endswith('.jpg')]

    # 获取x, y的最大值来确定输出图像的大小
    max_x, max_y = 0, 0
    for image_file in image_files:
        x, y = map(int, image_file.split('.')[0].split('_')[1:])
        max_x = max(max_x, x)
        max_y = max(max_y, y)


    # 使用第一张图像的大小来计算输出图像的大小
    sample_image = Image.open(os.path.join(folder_path, 't02_0_0.jpg'))
    width, height = sample_image.size
    total_width = width * (max_x + 1)
    total_height = height * (max_y + 1)

    # 创建一个空白的图像用于拼接
    stitched_image = Image.new('L', (total_width, total_height))

    for image_file in image_files:
        x, y = map(int, image_file.split('.')[0].split('_')[1:])
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        stitched_image.paste(image, (x * width, y * height))

    stitched_image.save(output_path)


# 使用示例
folder_path = r'E:\aafa\zypro\data\outimk2075'
output_path = r'E:\aafa\zypro\data\pinjiek2075.jpg'
stitch_images_in_folder(folder_path, output_path)

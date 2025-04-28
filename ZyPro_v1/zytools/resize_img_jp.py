import cv2
import os

def resize_image(input_path, output_path, scale_factor):
    """
    根据指定的缩放因子调整图像大小并保存。

    :param input_path: 输入图像的路径
    :param output_path: 输出图像的路径
    :param scale_factor: 缩放因子，如0.5表示将图像缩放为原始大小的一半
    """
    # 读取图像
    img = cv2.imread(input_path)

    # 检查图像是否正确加载
    if img is None:
        print(f"Error loading image {input_path}")
        return

    # 根据缩放因子调整图像大小
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    resized_img = cv2.resize(img, (new_width, new_height))

    # 保存调整大小的图像
    cv2.imwrite(output_path, resized_img)
    print(f"Image saved to {output_path}")

def process_folder(input_folder, output_folder, scale_factor):
    """
    处理文件夹内的所有图像。

    :param input_folder: 输入图像文件夹
    :param output_folder: 输出图像文件夹
    :param scale_factor: 缩放因子
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, scale_factor)

# 示例使用
input_directory = "D:\picele"
output_directory = "D:\picele03"
scale_factor = 0.3

process_folder(input_directory, output_directory, scale_factor)

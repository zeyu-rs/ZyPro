import cv2
import numpy as np
import os

def process_image(image_path, output_dir):
    # 读取图片
    image = cv2.imread(image_path)

    # 提取红色通道
    red_channel = image[:,:,2]

    # 二值化：所有大于0的值设置为255
    binary_red = np.where(red_channel > 0, 255, 0).astype(np.uint8)

    # 构建输出文件路径
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{base_name}.png")

    # 保存为单通道png格式
    cv2.imwrite(output_path, binary_red)

def batch_process(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的文件扩展名列表
    extensions = ['.jpg', '.jpeg', '.png']

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if os.path.splitext(filename)[1].lower() in extensions:
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)

input_dir = 'E:/aafa/zypro/data/labels'  # 替换为您的输入图片目录路径
output_dir = 'E:/aafa/zypro/data/label01' # 替换为您想要保存处理后的图片的目录路径
batch_process(input_dir, output_dir)

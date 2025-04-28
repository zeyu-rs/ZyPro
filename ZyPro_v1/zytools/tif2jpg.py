import os
from PIL import Image

def tif_to_jpg(source_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                # 打开tif文件
                image_path = os.path.join(root, file)
                image = Image.open(image_path)

                # 保存为jpg格式
                jpg_filename = os.path.splitext(file)[0] + ".jpg"
                jpg_path = os.path.join(root, jpg_filename)
                image.save(jpg_path, "JPEG")

source_directory = r'D:\picele2'
tif_to_jpg(source_directory)
print("转换完成!")

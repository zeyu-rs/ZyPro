import numpy as np
from osgeo import gdal
import torch
from torchvision import transforms

# 加载模型
model_path = "path_to_your_saved_model.pth"
model = torch.load(model_path)
model.eval()

# 1. 读取遥感影像
input_file = "path_to_your_raster.tif"
dataset = gdal.Open(input_file, gdal.GA_ReadOnly)

width = dataset.RasterXSize
height = dataset.RasterYSize
bands = dataset.RasterCount

data = np.zeros((height, width, bands), dtype=np.float32)
for i in range(bands):
    band = dataset.GetRasterBand(i + 1)
    data[:, :, i] = band.ReadAsArray()

# 归一化
data = data / 255.0


# 2. 定义滑动窗口
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # 当窗口移到图像边缘时进行调整
            x_end = x + windowSize[0]
            y_end = y + windowSize[1]
            if x_end > image.shape[1]:
                x_end = image.shape[1]
                x = x_end - windowSize[0]
            if y_end > image.shape[0]:
                y_end = image.shape[0]
                y = y_end - windowSize[1]

            yield (x, y, image[y:y_end, x:x_end])


# 3. 模型预测
output = np.zeros((height, width), dtype=np.float32)
overlap_count = np.zeros((height, width), dtype=np.float32)

transform = transforms.ToTensor()

for (x, y, window) in sliding_window(data, stepSize=256, windowSize=(512, 512, bands)):
    window = transform(window).unsqueeze(0)
    with torch.no_grad():
        pred = model(window)
        pred = pred.squeeze().numpy()

    output[y:y + 512, x:x + 512] += pred
    overlap_count[y:y + 512, x:x + 512] += 1

# 避免零除
overlap_count[overlap_count == 0] = 1
output /= overlap_count

# 4. 保存结果
output_path = "path_for_output.tif"
driver = gdal.GetDriverByName('GTiff')
outDataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
outBand = outDataset.GetRasterBand(1)
outBand.WriteArray(output)
outDataset.SetGeoTransform(dataset.GetGeoTransform())
outDataset.SetProjection(dataset.GetProjection())
outBand.FlushCache()
outDataset = None

import os
import gdal
import numpy as np
import torch
import numpy as np
from osgeo import gdal
import torch
from torchvision import transforms
from skimage.transform import resize


def get_files(directory, extension='.txt'):

    all_files = os.listdir(directory)
    specific_files = [os.path.join(directory, file) for file in all_files if file.endswith(extension)]

    return specific_files


def pre_tif_band1(input_channel,model,device,img_path,out_path,img_type):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_files = get_files(img_path, img_type)

    for img_file in img_files:
        filename = os.path.basename(img_file)
        out_filename = os.path.join(out_path, filename)
        rs_data = gdal.Open(img_file)
        images = (np.stack([rs_data.GetRasterBand(i).ReadAsArray() for i in range(1, input_channel+1)], axis=0))
        test_images = torch.tensor(images).float().unsqueeze(0)
        test_images = test_images.to(device)
        outputs = model(test_images)
        outputs = (outputs > 0.75).float().cpu().numpy()*150


        output = gdal.GetDriverByName('GTiff').Create(out_filename, rs_data.RasterXSize, rs_data.RasterYSize, 1,
                                                      gdal.GDT_Float32)
        output.SetProjection(rs_data.GetProjection())
        output.SetGeoTransform(rs_data.GetGeoTransform())
        output.GetRasterBand(1).WriteArray(outputs)
        output.GetRasterBand(1).SetNoDataValue(-999)

        output = None
        rs_data = None

def pre_tif_band2(input_channel, model, device, img_path, out_path, img_type):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_files = get_files(img_path, img_type)

    for img_file in img_files:
        filename = os.path.basename(img_file)
        out_filename = os.path.join(out_path, filename)

        rs_data = gdal.Open(img_file)
        ori_w = rs_data.RasterXSize
        ori_h = rs_data.RasterYSize

        images = np.stack([rs_data.GetRasterBand(i).ReadAsArray() for i in range(1, input_channel + 1)], axis=0)

        # resize to 512x512 using skimage
        images_resized = np.stack([
            resize(images[c], (512, 512), order=1, preserve_range=True, anti_aliasing=True)
            for c in range(input_channel)
        ])
        test_images = torch.tensor(images_resized).float().unsqueeze(0).to(device)

        outputs = model(test_images)
        outputs = (outputs > 0.75).float().squeeze().cpu().numpy() * 150

        # resize output back to original size using skimage
        output_resized = resize(outputs, (ori_h, ori_w), order=0, preserve_range=True, anti_aliasing=False)

        output = gdal.GetDriverByName('GTiff').Create(out_filename, ori_w, ori_h, 1, gdal.GDT_Float32)
        output.SetProjection(rs_data.GetProjection())
        output.SetGeoTransform(rs_data.GetGeoTransform())
        output.GetRasterBand(1).WriteArray(output_resized)
        output.GetRasterBand(1).SetNoDataValue(-999)

        output = None
        rs_data = None


class RemoteSensingPredictor:

    def __init__(self, model, input_file, output_path):
        self.model = model

        self.dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
        self.output_path = output_path

        self.transform = transforms.ToTensor()

    def read_data(self):
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize
        bands = self.dataset.RasterCount

        data = np.zeros((height, width, bands), dtype=np.float32)
        for i in range(bands):
            band = self.dataset.GetRasterBand(i + 1)
            data[:, :, i] = band.ReadAsArray()

        return data / 255.0

    def sliding_window(self, image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                x_end = x + windowSize[0]
                y_end = y + windowSize[1]
                if x_end > image.shape[1]:
                    x_end = image.shape[1]
                    x = x_end - windowSize[0]
                if y_end > image.shape[0]:
                    y_end = image.shape[0]
                    y = y_end - windowSize[1]
                yield (x, y, image[y:y_end, x:x_end])

    def predict(self, stepSize=256, windowSize=(512, 512, 4)):
        data = self.read_data()

        height, width, _ = data.shape
        output = np.zeros((height, width), dtype=np.float32)
        overlap_count = np.zeros((height, width), dtype=np.float32)

        for (x, y, window) in self.sliding_window(data, stepSize, windowSize):
            window = self.transform(window).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(window)
                pred = pred.squeeze().numpy()

            output[y:y + 512, x:x + 512] += pred
            overlap_count[y:y + 512, x:x + 512] += 1

        overlap_count[overlap_count == 0] = 1
        output /= overlap_count

        self.save_result(output)

    def save_result(self, output):
        driver = gdal.GetDriverByName('GTiff')
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize
        outDataset = driver.Create(self.output_path, width, height, 1, gdal.GDT_Float32)
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray(output)
        outDataset.SetGeoTransform(self.dataset.GetGeoTransform())
        outDataset.SetProjection(self.dataset.GetProjection())
        outBand.FlushCache()
        outDataset = None



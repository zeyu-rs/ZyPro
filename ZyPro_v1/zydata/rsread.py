import gdal
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class RSDataset_SegBasic(Dataset):
    def __init__(self, images_dir, labels_dir, input_channel):
        self.input_channel = input_channel
        self.target_size = (512, 512)  # resize to (H, W)
        self.images = self.read_multiband_images(images_dir)
        self.labels = self.read_singleband_labels(labels_dir)

    def read_multiband_images(self, images_dir):
        images = []
        for image_path in images_dir:
            ds = gdal.Open(image_path)
            input_channels = self.input_channel + 1
            img = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, input_channels)], axis=0)
            # resize each channel independently
            resized_img = np.stack([
                resize(img[c], self.target_size, order=1, preserve_range=True, anti_aliasing=True)
                for c in range(img.shape[0])
            ])
            images.append(resized_img)
        return images

    def read_singleband_labels(self, labels_dir):
        labels = []
        for label_path in labels_dir:
            ds = gdal.Open(label_path)
            label = ds.GetRasterBand(1).ReadAsArray()
            # Use nearest neighbor interpolation for labels
            label_resized = resize(label, self.target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            labels.append(label_resized)
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class RSDataset_YOLOBasic(Dataset):
    def __init__(self, images_dir, labels_dir, input_channel):
        self.input_channel = input_channel
        self.target_size = (512, 512)
        self.images = self.read_multiband_images(images_dir)
        self.labels = self.read_singleband_labels(labels_dir)

    def read_multiband_images(self, images_dir):
        images = []
        for image_path in images_dir:
            ds = gdal.Open(image_path)
            input_channels = self.input_channel + 1
            img = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, input_channels)], axis=0)
            resized_img = np.stack([
                resize(img[c], self.target_size, order=1, preserve_range=True, anti_aliasing=True)
                for c in range(img.shape[0])
            ])
            images.append(resized_img)
        return images

    def read_singleband_labels(self, labels_dir):
        labels = []
        for label_path in labels_dir:
            ds = gdal.Open(label_path)
            label = ds.GetRasterBand(1).ReadAsArray()
            label_resized = resize(label, self.target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            labels.append(label_resized)
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

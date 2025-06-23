"""
Author: Zeyu Xu
Date Created: September 4, 2023
Last Modified: Feb 12, 2024
Description: Training
"""

import torch
import os
from torch.utils.data import DataLoader
from zycontrol.train_set import train_model_seg
from zycontrol.lr import zy_RONP,zy_CosLR
from zynets.unet import UNet,UNet_Res34,UNetZ,UNetZ_CT2D,UNet256
from zydata.rsread import RSDataset_SegBasic
from zyloss.basic_loss import bceloss
from zyloss.loss_seg import TverskyLoss,ComboLoss

#Dataset
images_folder_path = r'D:/***/'
labels_folder_path = r'D:/***/'
log_path='D:/***'
task_name='unettrees'
input_channel=3
#Dataset_end

images_dir = [os.path.join(images_folder_path, filename) for filename in os.listdir(images_folder_path) if filename.endswith('.jpg')]
labels_dir = [os.path.join(labels_folder_path, filename) for filename in os.listdir(labels_folder_path) if filename.endswith('.png')]
print(images_dir)

#Set
dataset = RSDataset_SegBasic(images_dir, labels_dir , input_channel)
trainloader = DataLoader(dataset, batch_size=5, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = UNetZ(input_channel, 1)
model = UNetZ(3,1)
#criterion = bceloss()
criterion = ComboLoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = zy_RONP(optimizer)
#Set_end

#Train
train_model_seg(model, trainloader, criterion, optimizer, scheduler,log_path,task_name, num_epochs=500, device=device)




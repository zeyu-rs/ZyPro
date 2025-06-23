"""
Author: Zeyu Xu
Date Created: September 4, 2023
Last Modified: September 4, 2023
Description: Test from folder
"""

import torch
import os
from zynets.unet import UNet,UNet_Res34,UNetZ_CT2D,UNetZ
from zydata.rsseg import pre_tif_band2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = r"D:/***/"
out_path = r"D:/***/"
img_type= ".jpg"
input_channel=3

model = UNetZ(input_channel, 1)
model.load_state_dict(torch.load(r'D:/***/*.pth'))
#logs\2025-03-28-06-28-51\unettrees_final_2025-03-28-08-15-59.pth
#logs\2025-03-24-04-53-54\unettrees_final_2025-03-24-06-41-17.pth
model = model.to(device)
model.eval()
if not os.path.exists(out_path):
    os.makedirs(out_path)

pre_tif_band2(input_channel,model,device,img_path,out_path,img_type)
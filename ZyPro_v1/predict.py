
"""
Author: Zeyu Xu
Date Created: September 4, 2023
Last Modified: September 4, 2023
Description: Test from folder
"""

import torch
from zynets.unet import UNet,UNet_Res34,UNetZ_CT2D,UNetZ
from zydata.rsseg import pre_tif_band1
from zydata.rsseg import RemoteSensingPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = "data/testim/"
out_path = "data/outimk2075/"
img_type= ".jpg"

input_channel=3
model = UNetZ(input_channel, 1)
model.load_state_dict(torch.load('logs/*.pth'))
model = model.to(device)
model.eval()

predictor = RemoteSensingPredictor(model, "*.tif", "*.tif")
predictor.predict()
import torch
import __main__
import torch.nn as nn
import torch.nn.functional as F
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)

        center = self.center(pool4)

        dec4 = self.decoder4(center)

        k=torch.cat([center, dec4], 1)

        up4 = self.up4(center)

        dec3 = self.decoder3(up4)
        up3 = self.up3(torch.cat([pool3, dec3], 1))

        dec2 = self.decoder2(up3)
        up2 = self.up2(torch.cat([pool2, dec2], 1))

        dec1 = self.decoder1(up2)
        up1 = self.up1(torch.cat([pool1, dec1], 1))

        final = self.final(up1).squeeze()

        return F.sigmoid(final)


net = UNet(3, 1)
pthfile1 = r'rs_models_state_dict.pth'  # faster_rcnn_ckpt.pth
pthfile = r'rs_model.pth'  # faster_rcnn_ckpt.pth
net = torch.load(pthfile1)  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上
nets = torch.load(pthfile)
print(nets)
#print(type(net))  # 类型是 dict
#print(len(net))  # 长度为 4，即存在四个 key-value 键值对

for k in nets.keys():
    print(k)  # 查看四个键，分别是 model,optimizer,scheduler,iteration
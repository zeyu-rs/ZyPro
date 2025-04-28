import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNetZ_CT2D(nn.Module):
    def __init__(self, input_channels, out_channels, drop_out=0, regularizers=None):
        super(UNetZ_CT2D, self).__init__()

        # Define the layers for the UNet
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.center = self.conv_block(512, 1024)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.drop_out = drop_out
        self.regularizers = regularizers
        self.pool = nn.MaxPool2d(2, 2)

        # ConvTranspose2d layers
        self.up1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        if self.drop_out:
            enc4 = F.dropout(enc4, p=self.drop_out, training=self.training)

        center = self.center(self.pool(enc4))

        if self.drop_out:
            center = F.dropout(center, p=self.drop_out, training=self.training)

        dec4 = self.dec4(torch.cat([enc4, self.up1(center)], 1))
        dec3 = self.dec3(torch.cat([enc3, self.up2(dec4)], 1))
        dec2 = self.dec2(torch.cat([enc2, self.up3(dec3)], 1))
        dec1 = self.dec1(torch.cat([enc1, self.up4(dec2)], 1))
        final = self.final(dec1).squeeze()

        return F.sigmoid(final)

class UNetZ(nn.Module):
    def __init__(self, input_channels, out_channels, drop_out=0, regularizers=None):
        super(UNetZ, self).__init__()

        # Define the layers for the UNet
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.center = self.conv_block(512, 1024)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64,out_channels, kernel_size=1)
        self.drop_out = drop_out
        self.regularizers = regularizers
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        if self.drop_out:
            enc4 = F.dropout(enc4, p=self.drop_out, training=self.training)

        center = self.center(self.pool(enc4))

        if self.drop_out:
            center = F.dropout(center, p=self.drop_out, training=self.training)

        dec4 = self.dec4(torch.cat([enc4, self.up(center)], 1))
        dec3 = self.dec3(torch.cat([enc3, self.up(dec4)], 1))
        dec2 = self.dec2(torch.cat([enc2, self.up(dec3)], 1))
        dec1 = self.dec1(torch.cat([enc1, self.up(dec2)], 1))
        final = self.final(dec1).squeeze()


        return F.sigmoid(final)

class UNet_Res34(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Res34, self).__init__()

        # Pre-trained ResNet34
        self.resnet = models.resnet34(pretrained=True)

        # Modify the first layer of ResNet to accept various input channels
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Encoder layers (ResNet34 layers without fully connected)
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.center = self.resnet.layer4

        # Decoder layers
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder4 = self.conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = self.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = self.conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x) #64
        enc2 = self.encoder2(enc1) #64
        enc3 = self.encoder3(enc2) #128
        enc4 = self.encoder4(enc3)  #256
        center = self.center(enc4)  #512

        dec4 = self.up4(center)
        dec4 = torch.cat([dec4, enc4], 1)
        dec4 = self.decoder4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], 1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], 1)
        dec2 = self.decoder2(dec2)

        dec1 = torch.cat([dec2, enc1], 1)
        dec1 = self.decoder1(dec1)
        dec1 = self.up1(dec1)

        final = self.final(dec1).squeeze()
        return F.sigmoid(final)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet256(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet256, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = F.max_pool2d(x1, 2)
        x3 = self.down1(x2)
        x4 = F.max_pool2d(x3, 2)
        x5 = self.down2(x4)
        x6 = F.max_pool2d(x5, 2)
        x7 = self.down3(x6)

        x = self.up3(x7)
        x = torch.cat([x, x5], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        logits = self.outc(x)
        return logits


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


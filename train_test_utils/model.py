# Model for RadarHD

# Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_test_utils.unet_parts import *

from train_test_utils.model import DoubleConv, Down, Up, Up_nocat, OutConv

#REMARK: All the defined UNet architectures use final sigmoid activation to be compatible with BCE + Dice loss
# The output is therefore in [0, 1] range and can be thresholded to obtain binary occupancy maps    
# this is coherent with the use of BCEloss without any sigmoid inside the main training script
class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        conv_out = self.outc(x)
        logits = self.final_sigmoid(conv_out)

        return logits

class UNet2(nn.Module): #adapted version to SLAM dataset with one pooling added at the beginning to reduce the azimuthal direction from 180 to 64
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.prepool = nn.AdaptiveMaxPool2d((256,64)) # added
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prepool(x) #added
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        conv_out = self.outc(x)
        logits = self.final_sigmoid(conv_out)

        return logits

class UNet3(nn.Module): #adapted version to SLAM dataset with DoubleConv + pooling added at the beginning
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.double_conv = DoubleConv(n_channels, n_channels) #added
        self.prepool = nn.AdaptiveMaxPool2d((256,64)) # added
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.double_conv(x) #added
        x = self.prepool(x) #added
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        conv_out = self.outc(x)
        logits = self.final_sigmoid(conv_out)

        return logits

#UNet4 was used during first stages of sperimentation by this group but later became obsolete, all our scripts and helpers still account for 6 models
#it is left as a future developement opportunity
class UNet4(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("UNet4 is not implemented yet.")


class UNet5(nn.Module): #azimuth adapter version
    """
    UNet5 = UNet1 + azimuth adapter (Option A)

    Input:
        (B, n_channels, 256, 180)

    Internal UNet grid:
        (256, 64)

    Output:
        (B, n_classes, 256, 512)
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet5, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # -------------------------------------------------
        # Azimuth feature extractor (NO resize)
        # -------------------------------------------------
        self.azimuth_adapter = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(1, 7),
                stride=(1, 1),
                padding=(0, 3),
                bias=False
            ),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Original UNet1 backbone (UNCHANGED)
        # -------------------------------------------------
        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Extra upsampling stages (already present in UNet1)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, n_channels, 256, 180)
        """

        # ---------------------------------------------
        # Azimuth feature extraction (learned)
        # ---------------------------------------------
        x = self.azimuth_adapter(x)

        # ---------------------------------------------
        # HARD geometry normalization for UNet
        # ---------------------------------------------
        x = F.interpolate(
            x,
            size=(256, 64),
            mode="bilinear",
            align_corners=False
        )

        # ---------------------------------------------
        # UNet1 forward (unchanged)
        # ---------------------------------------------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)

        x = self.outc(x)
        x = self.final_sigmoid(x)

        return x

class UNet6(nn.Module):  # deeper azimuth stem
    """
    UNet6 = UNet5 + deeper azimuth stem

    Input:
        (B, n_channels, 256, 180)

    Internal UNet grid:
        (256, 64)

    Output:
        (B, n_classes, 256, 512)
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet6, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # -------------------------------------------------
        # Deeper azimuth stem (anisotropic, NO resize)
        # -------------------------------------------------
        self.azimuth_stem = nn.Sequential(
            # strong azimuth aggregation
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=64,
                kernel_size=(1, 9),
                stride=(1, 1),
                padding=(0, 4),
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 9),
                stride=(1, 1),
                padding=(0, 4),
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # mild range–azimuth coupling
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------
        # UNet backbone (same geometry as UNet1/5)
        # -------------------------------------------------
        self.inc = DoubleConv(64, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Extra upsampling stages (unchanged)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, n_channels, 256, 180)
        """

        # ---------------------------------------------
        # Learned azimuth processing (NO resampling)
        # ---------------------------------------------
        x = self.azimuth_stem(x)      # (B, 64, 256, 180)

        # ---------------------------------------------
        # HARD geometry normalization for UNet
        # ---------------------------------------------
        x = F.interpolate(
            x,
            size=(256, 64),
            mode="bilinear",
            align_corners=False
        )

        # ---------------------------------------------
        # UNet forward
        # ---------------------------------------------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)

        x = self.outc(x)
        x = self.final_sigmoid(x)

        return x



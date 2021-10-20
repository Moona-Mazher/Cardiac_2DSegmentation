# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:27:02 2021

@author: moona
"""

#%% preprocessing the dataset
import os
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = '/raid/Home/Users/aqayyum/EZProj/HeartMandM/MnM-2/'
os.listdir(path)

patients = os.listdir(f'{path}/training')
len(patients)

os.listdir(f'{path}/training/{patients[0]}')

import nibabel as nib
import random

##%% M&M2 dataset prepartion for segmentation
#
############################ dataloader for training and validation the model
import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, path = '/raid/Home/Users/aqayyum/EZProj/HeartMandM/MnM-2/training', trans=None):
        self.path = path
        self.data = data
        self.trans = trans
        self.num_classes = 4
        self.max_val = {
            'LA_ED': 4104.,
            'LA_ES': 7875.,
            'SA_ED': 11510.,
            'SA_ES': 9182.
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        patient = self.data.iloc[ix].patient
        image = self.data.iloc[ix].image
        channel = self.data.iloc[ix].channel

        img = nib.load(f'{self.path}/{patient}/{patient}_{image}.nii.gz').get_fdata()[...,channel] / self.max_val[image]
        mask = nib.load(f'{self.path}/{patient}/{patient}_{image}_gt.nii.gz').get_fdata()[...,channel].astype(np.int)
        if self.trans:
            t = self.trans(image=img, mask=mask)
            img = t['image']
            mask = t['mask'] 
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        # mask encoding
        mask_oh = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), self.num_classes).permute(2,0,1).float()
        return img_t, mask_oh

######################################load the data list ##############
import pandas as pd

data = pd.read_csv('training_data1.csv')

################## cross validation for training and testing the model ##########
folds = [(1, 32), (33, 64), (65, 96), (97, 128), (129, 160)]
trainfold=[]
testfold=[]
for f, val_split in enumerate(folds):
    
    train = data[(data.patient < val_split[0]) | (data.patient > val_split[1])]
    val = data[(data.patient >= val_split[0]) & (data.patient <= val_split[1])]
    trainfold.append(train)
    testfold.append(val)

######################## setting different foldes #################  
data_train=trainfold[1]
data_val=testfold[1]
data_train['patient'] = data_train['patient'].astype(str).str.zfill(3)
data_val['patient']= data_val['patient'].astype(str).str.zfill(3)    
###################### data augemnetation function ###################
import albumentations as A

trans = A.Compose([A.Resize(224, 224)])
import nibabel as nib
import random
import numpy as np
import matplotlib.pyplot as plt

ds = Dataset(data_train, trans=trans)

img, mask = ds[0]
img.shape, mask.shape

############################## data loader ################
data={'train':Dataset(data_train,trans=trans),
      'val':Dataset(data_val,trans=trans)}
## check dataset image shape and mask
imgs, masks = next(iter(data['train']))
imgs.shape, masks.shape
#################### take the batch size and prepare dataloader ######
batch_size=64
dataloader = {
    'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, pin_memory=True),
    'val': torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=False, pin_memory=True),
}
################## hyper parameters #####################

################################ define the model #################
import torch
import torch.nn as nn
import torch.nn.functional as F


#class DoubleConv(nn.Module):
#    """(convolution => [BN] => ReLU) * 2"""
#
#    def __init__(self, in_channels, out_channels, mid_channels=None):
#        super().__init__()
#        if not mid_channels:
#            mid_channels = out_channels
#        self.double_conv = nn.Sequential(
#            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(mid_channels),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True)
#        )
#
#    def forward(self, x):
#        return self.double_conv(x)
#
#
#class Down(nn.Module):
#    """Downscaling with maxpool then double conv"""
#
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.maxpool_conv = nn.Sequential(
#            nn.MaxPool2d(2),
#            DoubleConv(in_channels, out_channels)
#        )
#
#    def forward(self, x):
#        return self.maxpool_conv(x)
#
#
#class Up(nn.Module):
#    """Upscaling then double conv"""
#
#    def __init__(self, in_channels, out_channels, bilinear=True):
#        super().__init__()
#
#        # if bilinear, use the normal convolutions to reduce the number of channels
#        if bilinear:
#            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#        else:
#            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#            self.conv = DoubleConv(in_channels, out_channels)
#            ###self.resblock= ResBlock(in_channels, out_channels)
#
#
#    def forward(self, x1, x2):
#        x1 = self.up(x1)
#        # input is CHW
#        diffY = x2.size()[2] - x1.size()[2]
#        diffX = x2.size()[3] - x1.size()[3]
#
#        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                        diffY // 2, diffY - diffY // 2])
#        # if you have padding issues, see
#        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bRV1w6eCQFVVW3Q9RZeanm2bC9hjAT7d
#        x = torch.cat([x2, x1], dim=1)
#        return self.conv(x)
#
#
#class OutConv(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super(OutConv, self).__init__()
#        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#    def forward(self, x):
#        return self.conv(x)
#    
## New Residule Block    
#class ResBlock(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super(ResBlock, self).__init__()
#        self.downsample = nn.Sequential(
#            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#            nn.BatchNorm2d(out_channels))
#        self.double_conv = nn.Sequential(
#            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True),
#        )
#        self.down_sample = nn.MaxPool2d(2)
#        self.relu = nn.ReLU()
#
#    def forward(self, x):
#        identity = self.downsample(x)
#        out = self.double_conv(x)
#        out = self.relu(out + identity)
#        return out
#
#class ResUNet(nn.Module):
#    """ Full assembly of the parts to form the complete network """
#    def __init__(self, n_channels, n_classes, bilinear=True):
#        super(ResUNet, self).__init__()
#        self.n_channels = n_channels
#        self.n_classes = n_classes
#        self.bilinear = bilinear
#
#        self.inc = DoubleConv(n_channels, 64)
#        self.res1= ResBlock(64,64)
#        self.down1 = Down(64, 128)
#        self.res2= ResBlock(128, 128)
#        self.down2 = Down(128, 256)
#        self.res3= ResBlock(256, 256)
#        self.down3 = Down(256, 512)
#        self.res4= ResBlock(512, 512)
#        factor = 2 if bilinear else 1
#        self.down4 = Down(512, 1024 // factor)
#        self.up1 = Up(1024, 512 // factor, bilinear)
#        self.up2 = Up(512, 256 // factor, bilinear)
#        self.up3 = Up(256, 128 // factor, bilinear)
#        self.up4 = Up(128, 64, bilinear)
#        self.outc = OutConv(64, n_classes)
#
#    def forward(self, x):
#        x1 = self.inc(x)
#        res1= self.res1(x1) 
#        #print("1st conv block", x1.shape)
#        #print("1st res block", res1.shape)
#        x2 = self.down1(x1)
#        res2= self.res2(x2)
#        #print("sec conv block", x2.shape)
#        #print("sec res block", res2.shape)
#        x3 = self.down2(x2)
#        res3= self.res3(x3)
#        #print("3rd conv block", x3.shape)
#        #print("3rd res block", res3.shape)
#        x4 = self.down3(x3)
#        res4= self.res4(x4)
#        #print("4 conv block", x4.shape)
#        #print("4 res block", res4.shape)
#        x5 = self.down4(x4)
#        #print("Base down ", x5.shape)
#        x = self.up1(x5, res4)
#        #print("1st up block", x.shape)
#        x = self.up2(x, res3)
#        #print(" sec up block", x.shape)
#        x = self.up3(x, res2)
#        #print("3rd up block", x.shape)
#        x = self.up4(x, res1)
#   
#        logits = self.outc(x)
#
#        return logits
#
## generate random input (batch size, channel, height, width)
#inp=torch.rand(1,4,224,224)
#inp.shape
#    
## Giving Classes & Channels
#n_classes=4
#n_channels=1
#
##Creating Class Instance of Model Inf_Net_UNet Class
#model=ResUNet(n_channels, n_classes)

## Giving random input (inp) to the model
#out=model(inp)
#
#print(out.shape)
##################################################################### ResUnet model ###################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

"""
 RESNET34 UNET PRETRAINED ENCODER
"""


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ELU(True)
    )


class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.interpolate(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


class FPAv3(nn.Module):  # Custom  FPA for 224x224 input img sizes
    def __init__(self, input_dim, output_dim):
        super(FPAv3, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.interpolate(x_glob, scale_factor=x.shape[2], mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        # d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class Decoderv2(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class ResUnetv4(nn.Module):
    def __init__(self, model_version, pretrained=True, num_classes=1, classification=False, in_channels=1):
        super(ResUnetv4, self).__init__()

        self.classification = classification

        if "resnet18" in model_version:
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif "resnet34" in model_version:
            self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        else:
            assert False, "Unknown model: {}".format(model_version)

        if pretrained:
            assert False, "No pretrained models allowed!"
            """
            print("\n--- Frosted pretrained backbone! ---")
            for param in self.resnet.parameters():  # Frost model
                param.requires_grad = False
            """

        self.resnet.conv1 = torch.nn.Conv1d(in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1,
                                     SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2,
                                     SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3,
                                     SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4,
                                     SCse(512))

        self.center = nn.Sequential(FPAv3(512, 256),
                                    nn.MaxPool2d(2, 2))

        if classification:
            self.linear = nn.Linear(256 * 1 * 1, num_classes)

        else:
            self.decode5 = Decoderv2(256, 512, 64)
            self.decode4 = Decoderv2(64, 256, 64)
            self.decode3 = Decoderv2(64, 128, 64)
            self.decode2 = Decoderv2(64, 64, 64)
            self.decode1 = Decoder(64, 32, 64)

            self.logit = nn.Sequential(nn.Conv2d(320, num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        x = self.conv1(x)  # 64, 112, 112
        e2 = self.encode2(x)  # 64, 112, 112
        e3 = self.encode3(e2)  # 128, 56, 56
        e4 = self.encode4(e3)  # 256, 28, 28
        e5 = self.encode5(e4)  # 512, 14, 14

        f = self.center(e5)  # 256, 7, 7

        if self.classification:
            out = F.avg_pool2d(f, 7)  # (batch, 256, 8, 7) -> (batch, 256, 1, 1)
            out = out.view(out.size(0), -1)  # (batch, 256, 1, 1) -> (batch, 256)
            out = self.linear(out)
            return out

        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128
        d1 = self.decode1(d2)  # 64, 256, 256

        f = torch.cat((d1,
                       F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 256, 256

        logit = self.logit(f)  # 1, 256, 256

        return logit
        
#model=ResUnetv4("resnet34", pretrained=False, num_classes=4,classification=False, in_channels=1)

############################################## efficinetNet model ###########################################
#
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="efficientnet-b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
)


########### define the training and testing function ###########
import os
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# modelUdense=ResUNet(n_channels, n_classes)
# print(modelUdense)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% second training function for optimizing the model

def IoU(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    gt = gt > th
    intersection = torch.sum(gt * pr, axis=(-2,-1))
    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
    ious = (intersection + eps) / union
    return torch.mean(ious).item()


def iou(outputs, labels):
    # check output mask and labels
    outputs, labels = torch.sigmoid(outputs) > 0.5, labels > 0.5
    SMOOTH = 1e-6
    # BATCH x num_classes x H x W
    B, N, H, W = outputs.shape
    ious = []
    for i in range(N-1): # we skip the background
        _out, _labs = outputs[:,i,:,:], labels[:,i,:,:]
        intersection = (_out & _labs).float().sum((1, 2))  
        union = (_out | _labs).float().sum((1, 2))         
        iou = (intersection + SMOOTH) / (union + SMOOTH)  
        ious.append(iou.mean().item())
    return np.mean(ious)

from tqdm import tqdm

def fit(model,dataloader, epochs=100, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.to(device)
    hist = {'loss': [], 'iou': [], 'test_loss': [], 'test_iou': []}
    for epoch in range(1, epochs+1):
      bar = tqdm(dataloader['train'])
      train_loss, train_iou = [], []
      model.train()
      for imgs, masks in bar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        y_hat = model(imgs)
        loss = criterion(y_hat, masks)
        loss.backward()
        optimizer.step()
        ious = IoU(y_hat, masks)
        train_loss.append(loss.item())
        train_iou.append(ious)
        bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
      hist['loss'].append(np.mean(train_loss))
      hist['iou'].append(np.mean(train_iou))
      bar = tqdm(dataloader['val'])
      test_loss, test_iou = [], []
      model.eval()
      with torch.no_grad():
        for imgs, masks in bar:
          imgs, masks = imgs.to(device), masks.to(device)
          y_hat = model(imgs)
          loss = criterion(y_hat, masks)
          ious = IoU(y_hat, masks)
          test_loss.append(loss.item())
          test_iou.append(ious)
          bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
      hist['test_loss'].append(np.mean(test_loss))
      hist['test_iou'].append(np.mean(test_iou))
      print(f"\nEpoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f} test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    return hist
    
n_classes=4
n_channels=1

#Creating Class Instance of Model Inf_Net_UNet Class
#model=ResUNet(n_channels, n_classes)

hist=fit(model, dataloader, epochs=100, lr=3e-4)

#import pandas as pd
#df = pd.DataFrame(hist)
#df.to_csv("predcitioneffold2.csv",index=False)
#df.plot(grid=True)
#plt.show()
#################### save the model and load the model ############
#torch.save(model.state_dict(), 'checkpoint_foldefficnet.pt')
#pathl="/raid/Home/Users/aqayyum/EZProj/HeartMandM/checkpoint_foldnew.pt/"
#model.load_state_dict(torch.load("checkpoint_foldnew.pt"))
import os

path = '/raid/Home/Users/aqayyum/EZProj/HeartMandM/MnM-2/validation/'
patients = os.listdir(path)
len(patients)
import shutil

dest = '/raid/Home/Users/aqayyum/EZProj/HeartMandM/predfold2/'
#shutil.rmtree(dest)
#os.makedirs(dest)

max_val = {
    'LA_ED': 4104.,
    'LA_ES': 7875.,
    'SA_ED': 11510.,
    'SA_ES': 9182.
}
import albumentations as A

resize = A.Resize(224, 224)

def pred_la(patient, file):
    img = nib.load(f'{path}/{patient}/{patient}_{file}.nii.gz')
    img_data = img.get_fdata() / max_val[file]
    resized = resize(image=img_data[...,0])['image']
    img_t = torch.from_numpy(resized).float().unsqueeze(0)
    with torch.no_grad():
        output = model(img_t.unsqueeze(0).cuda())
        output = torch.sigmoid(output)
    mask = torch.argmax(output[0,...], axis=0).float().cpu().numpy()
    mask_resized = np.rint(A.Resize(*img.shape)(image=mask)['image'])[...,None]
    nib.save(nib.Nifti1Image(mask_resized, img.affine), f'{dest}/{patient}/{patient}_{file}_pred.nii.gz')  
    
def pred_sa(patient, file):
    img = nib.load(f'{path}/{patient}/{patient}_{file}.nii.gz')
    img_data = img.get_fdata() / max_val[file]
    resized = resize(image=img_data)['image']
    img_t = torch.from_numpy(resized).float().permute(2,0,1).unsqueeze(1)
    with torch.no_grad():
        output = model(img_t.cuda())
        output = torch.sigmoid(output)
    masks = torch.argmax(output, axis=1).float().cpu().permute(1,2,0).numpy()
    masks_resized = np.rint(A.Resize(*img.shape[:2])(image=masks)['image'])
    nib.save(nib.Nifti1Image(masks_resized, img.affine), f'{dest}/{patient}/{patient}_{file}_pred.nii.gz')
    
import nibabel as nib
from tqdm import tqdm
import albumentations as A
import torch

resize = A.Resize(224, 224)
model.eval()
model.cuda()
for patient in tqdm(patients):
    os.makedirs(f'{dest}/{patient}', exist_ok=True)
    pred_la(patient, 'LA_ED')
    pred_la(patient, 'LA_ES')
    pred_sa(patient, 'SA_ED')
    pred_sa(patient, 'SA_ES')

import matplotlib.pyplot as plt
import random
torch.save(model.state_dict(), 'checkpoint_foldefficnetfold2.pt')
# #%%############################## model for evluation #####################
# import random

# model.eval()
# with torch.no_grad():
#     ix = random.randint(0, len(dataset['val'])-1)
#     img, mask = dataset['val'][ix]
#     output = model(img.unsqueeze(0).to(device))[0]
#     pred_mask = torch.argmax(output, axis=0)
    
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
# ax1.imshow(img.squeeze(0))
# ax2.imshow(torch.argmax(mask, axis=0))
# ax3.imshow(pred_mask.squeeze().cpu().numpy())
# plt.show()


# #%% Validation and prediciton
# # validation on test images
# import os

# path = 'data/MnM-2/training'
# patients = os.listdir(path)[-40:]
# len(patients)

# ############ load the model ##################
# # from src.models import SMP

# # model = SMP.load_from_checkpoint('Unet-resnet18-val_iou=0.8279.ckpt')
# # model.hparams

# import nibabel as nib
# from tqdm import tqdm
# import albumentations as A
# import torch
# import random

# resize = A.Resize(224, 224)
# model.eval()
# model.cuda()

# ix = random.randint(0, len(patients))
# patient = patients[ix]
# max_val = {
#     'LA_ED': 4104.,
#     'LA_ES': 7875.,
#     'SA_ED': 11510.0,
#     'SA_ES': 9182.0
# }
# files = ['LA_ED', 'LA_ES', 'SA_ED', 'SA_ES']
# imgs, masks = [], []
# for f in files:
#     img = nib.load(f'{path}/{patient}/{patient}_{f}.nii.gz')
#     gt = nib.load(f'{path}/{patient}/{patient}_{f}_gt.nii.gz').get_fdata()
#     img_data = img.get_fdata() / max_val[f]
#     channels = img_data.shape[-1]
#     for channel in range(channels):
#         resized = resize(image=img_data[...,channel])['image']
#         img_t = torch.from_numpy(resized).float().unsqueeze(0)
#         with torch.no_grad():
#             output = model(img_t.unsqueeze(0).cuda())
#             output = torch.sigmoid(output)
#         mask = torch.argmax(output[0,...], axis=0).float().cpu().numpy()
#         mask_resized = np.rint(A.Resize(*img.shape[:2])(image=mask)['image'])[...,None]
#         imgs.append(img_data[...,channel])
#         masks.append((mask_resized, gt[...,channel]))

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10, 5*len(imgs)))
# for i, (img, mask) in enumerate(zip(imgs, masks)):
#     ax = plt.subplot(len(imgs), 2, i*2 + 1)
#     ax.imshow(img, cmap='gray')
#     pred, gt = mask
#     gt[gt == 0] = np.nan
#     ax.imshow(gt, alpha=0.5)
#     ax.axis('off')
#     ax.set_title(img.shape)
#     ax = plt.subplot(len(imgs), 2, i*2 + 2)
#     ax.imshow(img, cmap='gray')
#     pred[pred == 0] = np.nan
#     ax.imshow(pred, alpha=0.5)
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
# #%% https://github.com/juansensio/kaggle/blob/master/MnMs-2/00_submission.ipynb
# ############################ test on validation or out samples ########
# import os
#
# path = 'data/MnM-2/validation'
# patients = os.listdir(path)
# len(patients)
# import shutil
#
# dest = 'preds'
# #shutil.rmtree(dest)
# os.makedirs(dest)
#
# max_val = {
#     'LA_ED': 4104.,
#     'LA_ES': 7875.,
#     'SA_ED': 11510.,
#     'SA_ES': 9182.
# }
# import albumentations as A
#
# resize = A.Resize(224, 224)
#
# def pred_la(patient, file):
#     img = nib.load(f'{path}/{patient}/{patient}_{file}.nii.gz')
#     img_data = img.get_fdata() / max_val[file]
#     resized = resize(image=img_data[...,0])['image']
#     img_t = torch.from_numpy(resized).float().unsqueeze(0)
#     with torch.no_grad():
#         output = model(img_t.unsqueeze(0).cuda())
#         output = torch.sigmoid(output)
#     mask = torch.argmax(output[0,...], axis=0).float().cpu().numpy()
#     mask_resized = np.rint(A.Resize(*img.shape)(image=mask)['image'])[...,None]
#     nib.save(nib.Nifti1Image(mask_resized, img.affine), f'{dest}/{patient}/{patient}_{file}_pred.nii.gz')  
#    
# def pred_sa(patient, file):
#     img = nib.load(f'{path}/{patient}/{patient}_{file}.nii.gz')
#     img_data = img.get_fdata() / max_val[file]
#     resized = resize(image=img_data)['image']
#     img_t = torch.from_numpy(resized).float().permute(2,0,1).unsqueeze(1)
#     with torch.no_grad():
#         output = model(img_t.cuda())
#         output = torch.sigmoid(output)
#     masks = torch.argmax(output, axis=1).float().cpu().permute(1,2,0).numpy()
#     masks_resized = np.rint(A.Resize(*img.shape[:2])(image=masks)['image'])
#     nib.save(nib.Nifti1Image(masks_resized, img.affine), f'{dest}/{patient}/{patient}_{file}_pred.nii.gz')
#    
# import nibabel as nib
# from tqdm import tqdm
# import albumentations as A
# import torch
#
# resize = A.Resize(224, 224)
# model.eval()
# model.cuda()
# for patient in tqdm(patients):
#     os.makedirs(f'{dest}/{patient}', exist_ok=True)
#     pred_la(patient, 'LA_ED')
#     pred_la(patient, 'LA_ES')
#     pred_sa(patient, 'SA_ED')
#     pred_sa(patient, 'SA_ES')

# import matplotlib.pyplot as plt
# import random

# ix = random.randint(0, len(patients))
# sample = patients[ix]
# fig = plt.figure(figsize=(20,5))
# ax = plt.subplot(1, 4, 1)
# image_path = f'{path}/{sample}/{sample}_LA_ED.nii.gz'
# img = nib.load(image_path).get_fdata()
# ax.imshow(img,cmap='gray')
# ax.axis('off')
# ax.set_title(img.shape)
# ax = plt.subplot(1, 4, 2)
# image_path = f'{dest}/{sample}/{sample}_LA_ED_pred.nii.gz'
# mask = nib.load(image_path).get_fdata()
# ax.imshow(img, cmap='gray')
# ax.set_title(np.unique(mask))
# mask[mask == 0] = np.nan
# ax.imshow(mask, alpha=0.5)
# ax.axis('off')
# ax = plt.subplot(1, 4, 3)
# image_path = f'{path}/{sample}/{sample}_LA_ES.nii.gz'
# img = nib.load(image_path).get_fdata()
# ax.imshow(img,cmap='gray')
# ax.axis('off')
# ax.set_title(img.shape)
# ax = plt.subplot(1, 4, 4)
# image_path = f'{dest}/{sample}/{sample}_LA_ES_pred.nii.gz'
# mask = nib.load(image_path).get_fdata()
# ax.imshow(img, cmap='gray')
# ax.set_title(np.unique(mask))
# mask[mask == 0] = np.nan
# ax.imshow(mask, alpha=0.5)
# ax.axis('off')
# plt.tight_layout()
# plt.show()

# ed_img = nib.load(f'{path}/{sample}/{sample}_SA_ED.nii.gz').get_fdata()
# channels = ed_img.shape[-1]

# ed_pred = nib.load(f'{dest}/{sample}/{sample}_SA_ED_pred.nii.gz').get_fdata()
# es_img = nib.load(f'{path}/{sample}/{sample}_SA_ES.nii.gz').get_fdata()
# es_pred = nib.load(f'{dest}/{sample}/{sample}_SA_ES_pred.nii.gz').get_fdata()

# fig = plt.figure(figsize=(20, channels*5))
# for c in range(channels):
#     ax = plt.subplot(channels, 4, 4*c + 1)
#     ax.imshow(ed_img[...,c],cmap='gray')
#     ax.axis('off')
#     ax = plt.subplot(channels, 4, 4*c + 2)
#     ax.imshow(ed_img[...,c], cmap='gray')
#     mask = ed_pred[...,c]
#     mask[mask == 0] = np.nan
#     ax.imshow(mask, alpha=0.5)
#     ax.axis('off')
#     ax = plt.subplot(channels, 4, 4*c + 3)
#     ax.imshow(es_img[...,c],cmap='gray')
#     ax.axis('off')
#     ax = plt.subplot(channels, 4, 4*c + 4)
#     ax.imshow(es_img[...,c],cmap='gray')
#     mask = es_pred[...,c]
#     ax.set_title(np.unique(mask))
#     mask[mask == 0] = np.nan
#     ax.imshow(mask, alpha=0.5)
#     ax.axis('off')
# plt.tight_layout()
# plt.show()




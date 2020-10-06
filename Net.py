# coding: utf-8
# Team : IIPLAB Medical Group
# Author：Bro Yuan
# Date ：2020/10/2 下午5:52
# Tool ：
import torch.nn as nn
from torch.nn import functional as F
import torch


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后两层
        submodel1 = list(model.children())[0]
        submodel2 = list(model.children())[1]
        self.fea_extract = nn.Sequential(*(list(submodel1.children()) + list(submodel2.children())))
        self.Con2d = nn.Conv2d(512, 9 , kernel_size=(3,3), stride=(1,1))
        self.Up = nn.UpsamplingBilinear2d(scale_factor=8.0)

        # self.pool_layer = nn.MaxPool2d(32)
        # self.Linear_layer = nn.Linear(2048, 8)

    def forward(self, x):
        x = self.fea_extract(x)

        x = self.Con2d(x)

        x = self.Up(x)



        x = F.softmax(x, dim=1)

        return x
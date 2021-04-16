import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from .config import config


class SiameseFeature1(nn.Module):
    def __init__(self):
        super(SiameseFeature1, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

    def forward(self, x):
        out = self.feature1(x)
        return out


class SiameseFeature2(nn.Module):
    def __init__(self):
        super(SiameseFeature2, self).__init__()
        self.feature2 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )

    def forward(self, x):
        out = self.feature2(x)
        return out


class ClusterWeight(nn.Module):
    def __init__(self):
        super(ClusterWeight, self).__init__()
        self.cluster_weight = nn.Sequential(
            nn.Conv2d(256, 384, 3, 2, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 1024, 3, 1, 0),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 6, False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cluster_weight(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x














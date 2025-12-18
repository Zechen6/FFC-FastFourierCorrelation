import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

class ResNet34_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        # 适配 32×32 小图：改第一层
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # 去掉 maxpool
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
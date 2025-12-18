import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import torchvision.models as md
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import ResNet152_Weights, DenseNet201_Weights, Inception_V3_Weights, ViT_B_32_Weights,ResNet50_Weights
from torchvision.models import resnet152, densenet201, inception_v3, vit_b_32,resnet50
To_tensor = transforms.ToTensor()
To_image = transforms.ToPILImage()
device = "cuda:3"

data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

val_dataset = datasets.ImageFolder(
    root='/data01/lzc/img_net_dataset/val',
    transform=data_transform)

img_loader = DataLoader(val_dataset,batch_size=16)


def heat_map(data:torch.Tensor, fig_name, form):
    """
    该函数以R,G,B为顺序横着画三幅热力图
    """
    if data.dim() < 3:
        data = data.unsqueeze(-1)
    data1 = data[0]  # 第一组数据
    data2 = data[1]  # 第二组数据
    data3 = data[2]  # 第三组数据

    # 创建一个图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1行3列的子图

    # 绘制第一幅热力图
    
    sns.heatmap(data1, ax=axes[0], annot=False, cmap='viridis', cbar=True)
    axes[0].set_title("Heatmap R")

    # 绘制第二幅热力图
    sns.heatmap(data2, ax=axes[1], annot=False, cmap='viridis', cbar=True)
    axes[1].set_title("Heatmap G")

    # 绘制第三幅热力图
    sns.heatmap(data3, ax=axes[2], annot=False, cmap='viridis', cbar=True)
    axes[2].set_title("Heatmap B")

    # 调整布局
    plt.tight_layout()
    plt.savefig('fft_score_pics/'+fig_name+'.'+form, format=form)
    plt.close('all')


def view_fftshift_pics(pics:torch.Tensor):
    shifted_pics = torch.fft.fft2(pics, dim=(-2, -1))
    mags = torch.abs(shifted_pics)
    mags = torch.fft.fftshift(mags, dim=(-2, -1))
    mags = torch.fft.ifftshift(mags, dim=(-2, -1))
    heat_map(mags[0].cpu(), "fftshift", "png")


for batch,(X, y) in enumerate(img_loader):
    X = X.to(device)
    view_fftshift_pics(X)
    break
"""
本函数用于计算unet的deletion-insertion-game指标曲线
"""
import sys
cwd = '/data01/lzc/Experiments/FFC/UNet/'
sys.path.insert(0, cwd)

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import COCOSegmentationDataset
from predict import load_model, preprocess_image, predict_mask
import numpy as np
from attri_utils.unet_ffc import unet_ffc, select_bottom_element, select_top_element
from draw_tools.tensor2img import *
from train import combined_loss, UNet, dice_loss
import random as rd
from PIL import Image, ImageDraw, ImageFont
import os
import math
import warnings
warnings.filterwarnings('ignore')


def mask_dice(mask1, mask2):
    inter = ((mask1 == 1) & (mask2 == 1)).sum()
    total = (mask1 == 1).sum() + (mask2 == 1).sum()
    return (2 * inter.float()) / (total.float() + 1e-6)


def mask_dice_batch(mask1, mask2):
    """
    支持 batch 的 Dice 计算
    mask1, mask2: 任意形状 (..., H, W)
    """
    # 计算交集与总数（在空间维度求和）
    inter = ((mask1 == 1) & (mask2 == 1)).sum(dim=(-1, -2))
    total = (mask1 == 1).sum(dim=(-1, -2)) + (mask2 == 1).sum(dim=(-1, -2))

    # 避免除零
    dice = (2 * inter.float()) / (total.float() + 1e-6)
    return dice


def load_model(model_path='best_model.pth', device='cuda'):
    """加载训练好的模型"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = UNet(n_filters=32).to(device)
        # 添加weights_only=True来避免警告
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def deletion_insertion_curve(net:nn.Module, samples:torch.Tensor, 
                             scores:torch.Tensor,
                             step:int=0.05):
    """
    dig指标
    """
    dice_list = []
    all_steps = int(1/step)
    original_confidence = net(samples)
    original_pred_mask = original_confidence.round()

    for e in range(-1,all_steps-1,1):
        masks = select_top_element(scores, 1-(e+1)*step)
        #print(masks.sum().item())
        freq_sample = torch.fft.fft2(samples)
        masked_sample = freq_sample*masks
        filtered_sample = torch.fft.ifft2(masked_sample).real
        new_pred = net(filtered_sample)
        pred_mask = new_pred.round()
        
        dice_list.append(mask_dice_batch(pred_mask, original_pred_mask).mean().item())
        
    return dice_list


def baseline_main():
    # 数据路径设置
    train_dir = './dataset/train'
    val_dir = './dataset/valid'
    test_dir = './dataset/test'

    train_annotation_file = './dataset/train/_annotations.coco.json'
    test_annotation_file = './dataset/test/_annotations.coco.json'
    val_annotation_file = './dataset/valid/_annotations.coco.json'

    # 加载COCO数据集
    train_coco = COCO(train_annotation_file)
    val_coco = COCO(val_annotation_file)
    test_coco = COCO(test_annotation_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)
    
    # 创建数据加载器
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = load_model(model_path='best_model.pth', device=device)
    net.eval()
    step = 0.05
    dice_list = [0 for _ in range(int(1/step))]
    cn = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            if cn != 6:
                cn += 1
                continue
            scores = torch.fft.fft2(images).abs()
            dices = deletion_insertion_curve(net, images, scores, step=step)
            dice_list = [dice_list[i]+dices[i] for i in range(len(dices))]
            cn += 1
            print(f'Processed {cn} batches')
            print(dices)
            temp = [dice_list[i]/cn for i in range(len(dice_list))]
            print('Current Deletion-Insertion Game Dice Scores:')
            print(temp)
    dice_list = [dice_list[i]/cn for i in range(len(dice_list))]
    """print('Deletion-Insertion Game Dice Scores:')
    for i, d in enumerate(dice_list):
        print(f'Step {i+1}: {d:.4f}')"""



def interpret_main():
    # 数据路径设置
    train_dir = './dataset/train'
    val_dir = './dataset/valid'
    test_dir = './dataset/test'

    train_annotation_file = './dataset/train/_annotations.coco.json'
    test_annotation_file = './dataset/test/_annotations.coco.json'
    val_annotation_file = './dataset/valid/_annotations.coco.json'

    # 加载COCO数据集
    train_coco = COCO(train_annotation_file)
    val_coco = COCO(val_annotation_file)
    test_coco = COCO(test_annotation_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)
    
    # 创建数据加载器
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = load_model(model_path='best_model.pth', device=device)
    net.eval()
    step = 0.05
    dice_list = [0 for _ in range(int(1/step))]
    cn = 0
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            # best param: lr=8, echo=60, value:17.816857389041356
            scores = unet_ffc(net, images, lr=8, echo=60, loss_fn=combined_loss)
            dices = deletion_insertion_curve(net, images, scores, step=step)
            dice_list = [dice_list[i]+dices[i] for i in range(len(dices))]
            cn += 1
            print(f'Processed {cn} batches')
            print(dices)
            temp = [dice_list[i]/cn for i in range(len(dice_list))]
            print('Current Deletion-Insertion Game Dice Scores:')
            print(temp)

    dice_list = [dice_list[i]/cn for i in range(len(dice_list))]
    print('Deletion-Insertion Game Dice Scores:')
    print(dice_list)
    print(sum(dice_list))


if __name__ == '__main__':
    interpret_main()
    #baseline_main()
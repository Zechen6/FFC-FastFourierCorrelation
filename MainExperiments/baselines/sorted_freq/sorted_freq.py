"""
Implementation of Sorted Frequency-based Analysis for Neural Network Interpretability.
This script implements a method to analyze model behavior by manipulating image frequencies
in a sorted manner and observing the impact on model predictions.
"""

##################### Packages ###################
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.models import vit_b_32,resnet50
from torchvision.models import ViT_B_32_Weights, ResNet50_Weights
import time
import random as rd
from math import floor
import shutil
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import json
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
########################################################
#################### Global Params #####################

# Number of steps for frequency manipulation
step_num = 10
# Set device for computation (CPU or GPU)
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda:2"

# Enable logging
log_en = True


class NormalizeInverse(transforms.Normalize):
    """
    Custom transform to undo normalization on images.
    Useful for converting normalized tensors back to viewable images.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())

# Image normalization parameters (ImageNet standards)
normalize_fun = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

# Data transformation pipeline for input images
data_transform = transforms.Compose([
        transforms.Resize(224),          # Resize images to 224x224
        transforms.CenterCrop(224),      # Center crop to ensure consistent size
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(            # Normalize using ImageNet means and stds
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

# Transform for converting normalized tensors back to original scale
unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])

# Load ImageNet validation dataset
val_dataset = datasets.ImageFolder(
    root='/data01/img_net_dataset/val',
    transform=data_transform)

# Root path for saving interpretation results
root_path = "/data01/InterpretRes/"
###################################################################



def save_as_pic(tensor:torch.Tensor, fig_name):
    """
    Save a tensor as an image file.
    Args:
        tensor: Input tensor representing an image
        fig_name: Name for the output file
    """
    tensor4show = unnormalize(tensor)
    tensor4show = tensor4show.cpu().numpy()
    tensor4show = (tensor4show * 255).astype(np.uint8)

    # Transpose to shape (224, 224, 3)
    tensor4show = np.transpose(tensor4show, (1, 2, 0))

    # Save the image
    cv2.imwrite('pics/'+fig_name+'.png', tensor4show)


def heat_map(data:torch.Tensor, fig_name, form):
    """
    Generate and save heatmaps for R,G,B channels of the input tensor.
    Creates three side-by-side heatmaps, one for each color channel.
    
    Args:
        data: Input tensor of shape [3, H, W]
        fig_name: Name for the output file
        form: Output file format (e.g., 'png', 'jpg')
    """
    if data.dim() < 3:
        data = data.unsqueeze(-1)
    data1 = data[0]  # First channel data
    data2 = data[1]  # Second channel data
    data3 = data[2]  # Third channel data

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  

    
    sns.heatmap(data1, ax=axes[0], annot=False, cmap='viridis', cbar=True)
    axes[0].set_title("Heatmap R")

    sns.heatmap(data2, ax=axes[1], annot=False, cmap='viridis', cbar=True)
    axes[1].set_title("Heatmap G")

    sns.heatmap(data3, ax=axes[2], annot=False, cmap='viridis', cbar=True)
    axes[2].set_title("Heatmap B")

    # lay out
    plt.tight_layout()
    plt.savefig('fft_score_pics/'+fig_name+'.'+form, format=form)
    plt.close('all')


def generate_spiral_matrix(n):
    """
    Generate a spiral pattern matrix of size n x n.
    Used for creating structured frequency manipulation patterns.
    
    Args:
        n: Size of the square matrix
    Returns:
        Tensor containing spiral pattern values
    """
    matrix = [[0] * n for _ in range(n)]

    number = 1
    left, right, up, down = 0, n - 1, 0, n - 1
    while left < right and up < down:
        # Fill from left to right
        for i in range(left, right):
            matrix[up][i] = number
            number += 1

        # Fill from top to bottom
        for i in range(up, down):
            matrix[i][right] = number
            number += 1

        # Fill from right to left
        for i in range(right, left, -1):
            matrix[down][i] = number
            number += 1

        for i in range(down, up, -1):
            matrix[i][left] = number
            number += 1
        left += 1
        right -= 1
        up += 1
        down -= 1
    # When n is odd, fill the single cell in the middle of the square
    if n % 2 != 0: 
        matrix[n // 2][n // 2] = number
    return torch.tensor(matrix, dtype=torch.float32, device=device)

def generate_squre_matrix(N):
    """
    Generate a square matrix with values increasing from center to edges.
    Creates a pattern for radial frequency analysis.
    
    Args:
        N: Size of the square matrix
    Returns:
        Tensor containing the square pattern values
    """
    # Initialize matrix
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    # Center point coordinates
    center = (N-1) // 2
    # Current value to fill
    value = 1
    # Current square layer radius
    radius = 0
    while radius <= center:
        # Fill current square layer
        for i in range(center - radius, center + radius + 1):
            matrix[center - radius][i] = value
            matrix[center + radius][i] = value
            matrix[i][center - radius] = value
            matrix[i][center + radius] = value
        radius += 1
        value += 1
    return torch.tensor(matrix).to(device).float()


def expand_dim(tensor:torch.Tensor, order:int):
    """
    Expand tensor dimensions by adding specified number of dimensions at the end.
    
    Args:
        tensor: Input tensor
        order: Number of dimensions to add
    Returns:
        Tensor with expanded dimensions
    """
    res = tensor.clone()
    for i in range(order):
        res = res.unsqueeze(-1)
    return res


def construct_masks(scores:torch.Tensor, maintain_rate, imp:bool):
    """
    Construct masks for frequency manipulation based on importance scores.
    
    Args:
        scores: Importance scores tensor [B,C,H,W]
        maintain_rate: Proportion of frequencies to maintain
        imp: If True, delete important values; if False, delete unimportant values
    Returns:
        Binary mask tensor
    """
    flatten_score = scores.view(scores.shape[0],-1)
    if maintain_rate == 1:
        mask = torch.zeros_like(scores).to(device).float()
        return mask
    thred_ind = math.floor(flatten_score.shape[-1]*maintain_rate)
    sorted_score, inds = torch.sort(flatten_score, dim=-1, descending=imp)
    if imp:
        thred_value = sorted_score[:,thred_ind]
        mask = expand_dim(thred_value, 3)
        mask = mask.repeat(1,3,224,224)
        mask = torch.where(scores >= mask, 0, 1)
        return mask
    else:
        thred_value = sorted_score[:,thred_ind]
        mask = expand_dim(thred_value, 3)
        mask = mask.repeat(1,3,224,224)
        mask = torch.where(scores <= mask, 0, 1)
        return mask


def del_elements_sort_freq(score:torch.Tensor,
                           pics:torch.Tensor,
                           imp:bool):
    steps = list(range(step_num))
    res_pic = pics.clone()
    for i in steps:
        ratio = i/10+0.1
        #scores = torch.abs(torch.fft.ifft2(scores))
        mask = construct_masks(score, ratio,imp)
        freqs = torch.fft.fftshift(pics)
        masked_pic = torch.fft.ifftshift(freqs*mask).real
        res_pic = torch.concat([res_pic, masked_pic], dim=0)
    return res_pic


def del_elements(scores:torch.Tensor, 
                 pics:torch.Tensor, 
                 imp:bool,
                 Freq:bool,
                 rand:bool):
    """
    Delete elements from images based on frequency scores.
    
    Args:
        scores: Importance scores for frequencies
        pics: Input images
        imp: Whether to delete important (True) or unimportant (False) frequencies
        Freq: Whether to operate in frequency domain
        rand: Whether to use random selection
    Returns:
        Modified images
    """
    steps = list(range(step_num))
    res_pic = pics.clone()
    if Freq:
        for i in steps:
            if imp:
                ratio = i/10+0.1
            else:
                ratio = i/10+0.1
            #scores = torch.abs(torch.fft.ifft2(scores))
            mask = construct_masks(scores, ratio,imp).to(device)
            freqs = torch.fft.fftshift(torch.fft.fft2(pics))
            masked_pic = torch.fft.ifft2(torch.fft.ifftshift(freqs*mask)).real
            res_pic = torch.concat([res_pic, masked_pic], dim=0)
    
    return res_pic


def get_change_rate(conf, bs):
    """
    Calculate the rate of confidence change after frequency manipulation.
    
    Args:
        conf: Model confidence scores [B,1000*step]
        bs: Batch size
    Returns:
        Tensor containing change rates for each step
    """
    rate = bs*torch.ones(step_num).float().to(device)
    ind = torch.arange(bs).to(device).long()
    base = torch.softmax(conf[:bs,:], dim=-1)
    pred_label = base.argmax(-1)
    base = base[ind,pred_label]
    for i in range(1,step_num):
        conf_step = torch.softmax(conf[bs*i:bs*(i+1),:], dim=-1)
        conf_step = conf_step[ind,pred_label]
        rate[i] = torch.sum(conf_step/base)
    
    return rate

def get_conf(pics:torch.Tensor, 
             model:nn.Module, 
             scores:torch.Tensor, 
             imp:bool,
             bs:int,
             Freq:bool,
             rand:bool):
    #rate = torch.zeros(step_num).float().to(device)
    new_pics = del_elements(scores, pics, imp, Freq,rand)
    #new_pics = del_elements_sort_freq(scores, pics, imp)
    model.eval()
    conf = model(new_pics)
    return get_change_rate(conf, bs)


def print_info(msg, file_name = 'eval_ch_rate_sort_freq.log'):
    file_name = 'CIFAR/Task/FourierAttribution/Cache/'+file_name
    if log_en:
        f = open(file_name,'a')
        print(msg, file=f)
        f.close()
    else:
        print(msg)


def evaluate(model_name, method_name, imp):
    """
    Evaluate the impact of frequency manipulation on model predictions.
    
    Args:
        model_name: Name of the model to evaluate ('ViT' or 'ResNet50')
        method_name: Name of the evaluation method
        imp: Whether to manipulate important frequencies
    Returns:
        List of confidence change rates
    """
    bs = 500
    #############
    #############
    val_loader = DataLoader(val_dataset,batch_size=bs,shuffle=False)
    rate = torch.zeros(step_num).float().to(device)
    rand = False
    cn = 0
    scores = sort_score().unsqueeze(0).unsqueeze(0).repeat(bs,3,1,1)
    for batch,(X,y) in enumerate(val_loader):
        cn += 1
        if batch % 10 == 0:
            msg = "Method "+method_name+" "+str(batch) + " evaluating"
            print_info(msg)
        X = X.to(device)
        
        if X.shape[0] != scores.shape[0]:
            msg = "Error in "+str(batch)
            print_info(msg)
            continue
        if model_name == 'ViT':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) 

        Freq = True

        model.to(device)
        model.eval()
        rate += get_conf(X,model,scores,imp,X.shape[0],Freq,rand)
        if batch % 10 == 0:
            msg = str(rate/((batch+1)*bs))
            print_info(msg)
    rate /= 50000
    rate = rate.cpu()
    rate = rate.numpy().tolist()
    msg = method_name+":"+model_name+":"+str(rate)
    print_info(msg, 'temp_sorted_freq.log')
    return rate


def sort_score(rows=224, cols=224):
    """
    Generate a matrix of scores based on distance from center.
    Used for frequency importance scoring.
    
    Args:
        rows: Number of rows in the matrix
        cols: Number of columns in the matrix
    Returns:
        Tensor containing distance-based scores
    """
    y = torch.arange(rows).reshape(-1, 1)
    x = torch.arange(cols).reshape(1, -1)

    center_y = (rows) / 2
    center_x = (cols) / 2

    dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
    return dist



def run():
    """
    Main function to run the evaluation across different models and methods.
    Evaluates both ResNet50 and ViT models using various frequency manipulation strategies.
    """
    methods = ['sorted_score']
    models = ['ResNet50','ViT']
    res_dict = {}
    for method in methods:
        res_dict[method] = {}
        for m in models:
            """if m == 'ViT':
                continue"""
            if m == 'ViT' and method == 'fullgrad':
                continue
            #if method != 'sort_freq':
            #    continue
            res_dict[method][m] = {}
            for imp in [True,False]:
                evaluate(m, method, imp)

def main():
    """
    Entry point of the script.
    Runs the evaluation process with gradient disabled for efficiency.
    """
    with torch.no_grad():
        run()
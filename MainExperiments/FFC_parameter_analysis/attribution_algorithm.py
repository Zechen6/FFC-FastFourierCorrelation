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

step_num = 10
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda:3"

log_en = True


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())

normalize_fun = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(
    root='/data01/img_net_dataset/val',
    transform=data_transform)

root_path = "/data01/InterpretRes/"
###################################################################


def save_as_pic(tensor: torch.Tensor, fig_name):
    """
    Save a PyTorch tensor as an image file.

    This function takes a tensor, unnormalizes it, converts it to a numpy array,
    and then saves it as a PNG image.

    Args:
        tensor (torch.Tensor): The tensor to be saved as an image.
        fig_name (str): The filename for the saved image.
    """
    tensor4show = unnormalize(tensor)
    tensor4show = tensor4show.cpu().numpy()
    tensor4show = (tensor4show * 255).astype(np.uint8)

    # Transpose the tensor to (224, 224, 3) shape
    tensor4show = np.transpose(tensor4show, (1, 2, 0))

    # Save the image using OpenCV
    cv2.imwrite('pics/' + fig_name + '.png', tensor4show)


def heat_map(data: torch.Tensor, fig_name, form):
    """
    Plot three heatmaps for the R, G, and B channels of the input tensor.

    This function takes a tensor, splits it into three channels, and plots
    a heatmap for each channel using Seaborn.

    Args:
        data (torch.Tensor): The input tensor.
        fig_name (str): The filename for the saved heatmap.
        form (str): The file format for the saved heatmap.
    """
    if data.dim() < 3:
        data = data.unsqueeze(-1)
    data1 = data[0]  # First channel
    data2 = data[1]  # Second channel
    data3 = data[2]  # Third channel

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Plot the heatmap for the first channel
    sns.heatmap(data1, ax=axes[0], annot=False, cmap='viridis', cbar=True)
    axes[0].set_title("Heatmap R")

    # Plot the heatmap for the second channel
    sns.heatmap(data2, ax=axes[1], annot=False, cmap='viridis', cbar=True)
    axes[1].set_title("Heatmap G")

    # Plot the heatmap for the third channel
    sns.heatmap(data3, ax=axes[2], annot=False, cmap='viridis', cbar=True)
    axes[2].set_title("Heatmap B")

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig('fft_score_pics/' + fig_name + '.' + form, format=form)
    plt.close('all')


def generate_spiral_matrix(n):
    """
    Generate a spiral matrix of size n x n.

    This function creates a matrix where the numbers are arranged in a spiral
    pattern, starting from the top-left corner and moving clockwise.

    Args:
        n (int): The size of the matrix.

    Returns:
        torch.Tensor: The generated spiral matrix.
    """
    matrix = [[0] * n for _ in range(n)]

    number = 1
    left, right, up, down = 0, n - 1, 0, n - 1
    while left < right and up < down:
        # Move from left to right
        for i in range(left, right):
            matrix[up][i] = number
            number += 1

        # Move from up to down
        for i in range(up, down):
            matrix[i][right] = number
            number += 1

        # Move from right to left
        for i in range(right, left, -1):
            matrix[down][i] = number
            number += 1

        # Move from down to up
        for i in range(down, up, -1):
            matrix[i][left] = number
            number += 1
        left += 1
        right -= 1
        up += 1
        down -= 1
    # If n is odd, fill the center of the matrix
    if n % 2 != 0: 
        matrix[n // 2][n // 2] = number
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def generate_squre_matrix(N):
    """
    Generate a square matrix of size N x N with increasing values from the center.

    This function creates a matrix where the values increase from the center
    outward, forming a square pattern.

    Args:
        N (int): The size of the matrix.

    Returns:
        torch.Tensor: The generated square matrix.
    """
    # Initialize the matrix
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    # Center coordinates
    center = (N-1) // 2
    # Current value to fill
    value = 1
    # Current square layer radius
    radius = 0
    while radius <= center:
        # Fill the current square layer
        for i in range(center - radius, center + radius + 1):
            matrix[center - radius][i] = value
            matrix[center + radius][i] = value
            matrix[i][center - radius] = value
            matrix[i][center + radius] = value
        # Increase radius and value
        radius += 1
        value += 1
    return torch.tensor(matrix).to(device).float()


def expand_dim(tensor: torch.Tensor, order: int):
    """
    Expand the dimensions of a tensor by adding new dimensions at the end.

    This function takes a tensor and adds a specified number of new dimensions
    to the end of the tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        order (int): The number of new dimensions to add.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    res = tensor.clone()
    for i in range(order):
        res = res.unsqueeze(-1)
    return res


def construct_masks(scores:torch.Tensor, maintain_rate, imp:bool):
    """
    scores:[B,C,H,W]
    imp == True means it is deleting important values
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
    steps = list(range(step_num))
    res_pic = pics.clone()
    if Freq:
        for i in steps:
            if imp:
                ratio = i/10+0.1
            else:
                ratio = i/10+0.1
            #scores = torch.abs(torch.fft.ifft2(scores))
            mask = construct_masks(scores, ratio,imp)
            freqs = torch.fft.fft2(pics)
            masked_pic = torch.fft.ifft2(freqs*mask).real
            res_pic = torch.concat([res_pic, masked_pic], dim=0)
    else:
        scores = torch.abs(torch.fft.fft2(scores)).real
        freqs = torch.fft.fft2(pics)
        for i in steps:
            if imp:
                ratio = i/100+0.01
            else:
                ratio = i/100+0.9
            #mask = construct_masks(scores, ratio, imp)
            if rand is False:
                mask = construct_masks(scores, ratio,imp)
            else:
                mask = construct_masks(torch.rand_like(scores.real).to(device), ratio, imp)
            masked_pic = torch.fft.ifft2(freqs*mask).real
            #pic4show = masked_pic[0]
            #save_as_pic(pic4show, 'inputgrad-'+str(i)+'-'+str(imp))
            #heat_map((torch.abs(scores)*mask).cpu()[0],'inputgrad-'+str(i)+'-'+str(imp),form='png')
            res_pic = torch.concat([res_pic, masked_pic], dim=0)
    
    return res_pic


def get_change_rate(conf, bs):
    """
    conf:[B,1000*step]
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


def print_info(msg, file_name = 'eval_ch_rate_local.log'):
    file_name = 'CIFAR/Task/FourierAttribution/Cache/'+file_name
    if log_en:
        f = open(file_name,'a')
        print(msg, file=f)
        f.close()
    else:
        print(msg)


def evaluate(model_name, method_name, imp, e, lr):
    res_path = root_path+method_name+"/"+model_name+"/"
    rep_time = 3
    bs = 400
    #############
    #############
    # shuffle变成True了
    val_loader = DataLoader(val_dataset,batch_size=bs,shuffle=False)
    rate = torch.zeros(rep_time,step_num).float().to(device)
    rand = False
    loss_final = [0,0,0]
    cn = 0
    score_list = [None, None, None]
    for batch,(X,y) in enumerate(val_loader):
        cn += 1
        if batch % 10 == 0:
            msg = "Method "+method_name+" "+str(batch) + " evaluating"
            print_info(msg)
        X = X.to(device)
        if method_name in ['fullgrad','inputgrad','smoothgrad','gradcam','smooth_grad','input_grad']:
            scores = torch.load(res_path+str(batch+1), map_location=device)
        elif method_name == 'random':
            scores = torch.rand_like(X).to(device)
            rand = True
        elif method_name == 'energy':
            scores = torch.abs(torch.fft.fft2(X))
        elif method_name == 'sort_freq':
            scores = generate_squre_matrix(224)
            scores = scores.unsqueeze(0).unsqueeze(0).repeat(X.shape[0],3,1,1)
        elif method_name =='test_method':
            with torch.enable_grad():
                for _ in range(rep_time):
                    scores, loss_t  = test_method(model_name, X, e, lr)
                    score_list[_] = scores.clone()
                    loss_final[_] += loss_t
        else:
            scores= torch.load(res_path+str(batch), map_location=device)
        
        if X.shape[0] != scores.shape[0]:
            msg = "Error in "+str(batch)
            print_info(msg)
            continue
        

        Freq = True
        if method_name == 'Freq' or method_name == 'energy':
            Freq = True
        for _ in range(rep_time):
            if model_name == 'ViT':
                model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            else:
                model = resnet50(weights=ResNet50_Weights.DEFAULT) 
            model.to(device)
            model.eval()
            rate[_] += get_conf(X,model,scores,imp,X.shape[0],Freq,rand)
        if batch % 10 == 0:
            msg = str(rate/((batch+1)*bs))
            print_info(msg)
        torch.cuda.empty_cache()
    rate /= 50000
    rate = rate.cpu()
    rate = rate.numpy().tolist()
    for _ in range(rep_time):
        loss_final[_] /= cn
    msg = method_name+":"+model_name+":"+str(rate)+'\t Iter:'+str(e)+'\tLr:'+str(lr)+'\t Loss:'+str(loss_final)
    print_info(msg,'temp_ev_res.log')
    return rate


def test_method(model_name, 
                X:torch.Tensor, 
                e, 
                lr):
    X_ = X.clone()
    if model_name == 'ViT':
        model_test = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    else:
        model_test = resnet50(weights=ResNet50_Weights.DEFAULT) 
    model_test.eval()
    X_.requires_grad = True
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_final = 0
    with open('CIFAR/Task/FourierAttribution/Cache/loss.log','a') as f:
        print('-------',file=f)
        for i in range(e):
            X_.requires_grad = True
            model_test.to(device)
            pred = model_test(X_)
            pred_label = pred.argmax(-1)
            loss = loss_fn(pred, pred_label)
            loss.backward()
            print(loss.item(), e, file=f)
            X_grad = X_.grad.clone()
            X_.grad.zero_()
            with torch.no_grad():
                X_ = X_ - lr*X_grad
        loss_final = loss.item()
        with torch.no_grad():
            freqs = torch.fft.fft2(X)
            freq_after = torch.fft.fft2(X_)
            mag_ori = torch.abs(freqs)
            ori_after_mutual_energy = 2*(torch.conj(freq_after)*freqs).real
            scores = (ori_after_mutual_energy/(mag_ori)-mag_ori)
    del model_test, X_, X_grad
    torch.cuda.empty_cache()
    return scores, loss_final


def run():
    methods = ['test_method']
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
            for imp in [False, True]:
                for e in [1,10,20,30,40,50]:
                    for lr in [0.1,1,10,100,1000]:
                        evaluate(m, method, imp, e, lr)

def main():
    with torch.no_grad():
        run()
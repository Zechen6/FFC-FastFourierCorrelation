##################### Packages ###################
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet152, densenet201, inception_v3, vit_b_32,resnet50
from torchvision.models import ResNet152_Weights, DenseNet201_Weights, Inception_V3_Weights, ViT_B_32_Weights, ResNet50_Weights
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
    device = "cuda:0"

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

methods_bs_dict = {'fullgrad':8,'Freq':128,
                   'random':16,'inputgrad':100,
                   'gradcam':100,'IG':4,
                   'smoothgrad':100}#,'energy':100,'sort_freq':128,'input_grad':128,'smooth_grad':128}
special_methods = {'input_grad':128,'smooth_grad':128}
focus_method = {'test_method':2}
done_method = ['Freq','inputgrad','gradcam','fullgrad','random']
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
            """pic4show = masked_pic[0]
            print(255*torch.max(torch.abs(unnormalize(pic4show)-unnormalize(pics[0]))))
            save_as_pic(pic4show, 'inputgrad/test-'+str(i)+'-'+str(imp))
            heat_map((scores*mask).cpu()[0],'inputgrad/test-'+str(i)+'-'+str(imp),form='png')"""
            res_pic = torch.concat([res_pic, masked_pic], dim=0)
    else:
        #scores = torch.abs(torch.fft.fft2(scores)).real
        #freqs = torch.fft.fft2(pics)
        for i in steps:
            if imp:
                ratio = i/10+0.1
            else:
                ratio = i/10+0.1
            #mask = construct_masks(scores, ratio, imp)
            if rand is False:
                mask = construct_masks(scores, ratio,imp)
            else:
                mask = construct_masks(torch.rand_like(scores.real).to(device), ratio, imp)
            #masked_pic = torch.fft.ifft2(freqs*mask).real
            masked_pic = pics*mask
            if i < 1:
                pic4show = masked_pic[1]
                save_as_pic(pic4show, list(focus_method.keys())[0]+'-'+str(i)+'-'+str(imp))
            #heat_map((torch.abs(scores)*mask).cpu()[0],'fullgrad/test-'+str(i)+'-'+str(imp),form='png')
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
    #conf_temp = conf.argmax(-1)
    #print(conf_temp)
    return get_change_rate(conf, bs)


def print_info(msg, file_name = 'eval_ch_rate_local.log'):
    if log_en:
        f = open(file_name,'a')
        print(msg, file=f)
        f.close()
    else:
        print(msg)


def evaluate(model_name, method_name, imp, e):
    res_path = root_path+method_name+"/"+model_name+"/"
    
    bs = focus_method[method_name]
    #############
    #############
    val_loader = DataLoader(val_dataset,batch_size=bs,shuffle=False)
    rate = torch.zeros(step_num).float().to(device)
    rand = False
    time_sum = 0
    for batch,(X,y) in enumerate(val_loader):
        if batch % 10 == 0:
            msg = "Method "+method_name+" "+str(batch) + " evaluating"
            print_info(msg)
        X = X.to(device)
        if method_name in ['fullgrad','inputgrad','grad_cam','smoothgrad','gradcam','smooth_grad','input_grad']:
            scores = torch.load(res_path+str(batch+1), map_location=device)
        elif method_name == 'random':
            scores = torch.rand_like(X).to(device)
            rand = True
        elif method_name == 'energy':
            scores = torch.abs(torch.fft.fft2(X))
            scores = torch.log(1+scores-scores.min())
        elif method_name == 'sort_freq':
            scores = generate_squre_matrix(224)
            scores = scores.unsqueeze(0).unsqueeze(0).repeat(X.shape[0],3,1,1)
        elif method_name =='test_method':
            
            with torch.enable_grad():
                scores, time_cost = test_method(model_name, X, e)
                time_sum += time_cost
                print("Time:",time_sum/(batch+1), e)
            #torch.save(scores, res_path+str(batch)+'.pt')
            #continue
        else:
            scores = torch.load(res_path+str(batch), map_location=device)
        
        if X.shape[0] != scores.shape[0]:
            msg = "Error in "+str(batch)+"X:"+str(X.shape[0])+" scores:"+str(scores.shape[0])
            print_info(msg)
            continue
        if model_name == 'ViT':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) 

        Freq = False
        if method_name == 'Freq' or method_name == 'energy' or method_name == 'test_method':
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
    print_info(msg,'temp_ev_res.log')
    return rate


def run():
    methods = focus_method.keys()
    models = ['ResNet50']
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
                for e in [1]:
                    res = evaluate(m, method, imp, e)
                    res_dict[method][m][str(imp)] = res
    #with open('results/Evaluation-ifftFFT-ResultRes.json','w', encoding='UTF-8') as f:
    #    json.dump(res_dict, f, ensure_ascii=False, indent=4)

def test_method(model_name,X:torch.Tensor, e):
    X_ = X.clone()
    if model_name == 'ViT':
        model_test = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    else:
        model_test = resnet50(weights=ResNet50_Weights.DEFAULT) 
    model_test.to(device)
    model_test.eval()
    
    #grad_shape = e+list(X_.shape)
    #grad_list = torch.zeros(grad_shape).to(device).float()
    loss_fn = torch.nn.CrossEntropyLoss()
    st_time = time.time()
    
    X_.requires_grad = True
    lr = 1000

    for i in range(e):
        X_.requires_grad = True
        pred = model_test(X_)
        if i == 0:
            pred_label = pred.argmax(-1).detach()
        loss = loss_fn(pred, pred_label)
        loss.backward()
        X_grad = X_.grad.clone()
        
        #grad_list[i] = X_grad
        X_.grad.zero_()
        with torch.no_grad():
            X_ = X_ - lr*X_grad
    with torch.no_grad():
        freqs = torch.fft.fft2(X)
        
        #freq_grad = torch.fft.fft2(grad_list.mean(dim=0))
        freq_after = torch.fft.fft2(X_)
        #mag_change = torch.abs(freq_after)-torch.abs(freqs)
        #mag_grad = torch.abs(freq_grad)
        mag_ori = torch.abs(freqs)
        #mag_after = torch.abs(freq_after)
        ori_after_mutual_energy = 2*(torch.conj(freq_after)*freqs).real
        #ori_grad_mutual_energy = 2*(torch.conj(-freq_grad)*freqs).real
        #grad_aft_mutual_energy = (torch.conj(freq_grad)*freq_after).real
        

        scores = (ori_after_mutual_energy/(mag_ori)-mag_ori)
        scores = torch.log(1-scores.min()+scores)
        #torch.log(1+torch.abs(freqs))*(1-torch.abs(torch.angle(freqs)-torch.angle(freq_change))/(torch.pi))
    ed_time = time.time()

    
    
    del model_test, X_, X_grad
    torch.cuda.empty_cache()
    
    #print("Time:",ed_time-st_time, e)
    return scores, ed_time-st_time


if __name__ == "__main__":
    with torch.no_grad():
        run()


##################### Packages ###################
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet152, densenet201, inception_v3, vit_b_32, resnet50
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
corr_cn = 0
step_num = 10
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda:3"

log_en = False


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        std_inv = 1 / (self.std)
        mean_inv = -self.mean * std_inv
        self.std = self.std.unsqueeze(-1).unsqueeze(-1).repeat(1, 224, 224).to(device)
        self.mean = self.mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 224, 224).to(device)
        #super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return tensor * self.std + self.mean

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

unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(
    root='/data01/img_net_dataset/val',
    transform=data_transform)

methods_bs_dict = {'fullgrad': 8, 'Freq': 128,
                   'random': 16, 'inputgrad': 100,
                   'gradcam': 100, 'IG': 4,
                   'smoothgrad': 100}  # ,'energy':100,'sort_freq':128,'input_grad':128,'smooth_grad':128}
special_methods = {'input_grad': 128, 'smooth_grad': 128}
focus_method = {'test_method': 300}
done_method = ['Freq', 'inputgrad', 'gradcam', 'fullgrad', 'random']
root_path = "/data01/InterpretRes/"
###################################################################


def sort_score(rows=224, cols=224):
    """
    Calculate the distance of each pixel to the center of the image
    """
    y = torch.arange(rows).reshape(-1, 1)
    x = torch.arange(cols).reshape(1, -1)

    center_y = (rows) / 2
    center_x = (cols) / 2

    dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
    return dist


def save_as_pic(tensor: torch.Tensor, fig_name):
    """
    Save a tensor as an image
    """
    tensor4show = unnormalize(tensor)
    tensor4show = tensor4show.cpu()
    #tensor4show = torch.transpose(tensor4show, (1, 2, 0))
    denormalized_img = transforms.ToPILImage()(tensor4show)

    plt.imshow(denormalized_img)

    plt.savefig('error_sample_analysis/' + fig_name + '.png', format='png')
    plt.close('all')


def heat_map(data: torch.Tensor, fig_name, form):
    """
    Plot three heatmaps for the R, G, and B channels of the input tensor
    """
    if data.dim() < 3:
        data = data.unsqueeze(-1)
    data1 = data[0]  # 
    data2 = data[1]  # 
    data3 = data[2]  # 

    # 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 

    # 
    sns.heatmap(data1, ax=axes[0], annot=False, cmap='viridis', cbar=True)
    axes[0].set_title("Heatmap R")

    # 
    sns.heatmap(data2, ax=axes[1], annot=False, cmap='viridis', cbar=True)
    axes[1].set_title("Heatmap G")

    # 
    sns.heatmap(data3, ax=axes[2], annot=False, cmap='viridis', cbar=True)
    axes[2].set_title("Heatmap B")

    # 
    plt.tight_layout()
    plt.savefig('error_sample_analysis/' + fig_name + '.' + form, format=form)
    plt.close('all')


def generate_spiral_matrix(n):
    """
    Generate a spiral matrix of size n x n
    """
    matrix = [[0] * n for _ in range(n)]

    number = 1
    left, right, up, down = 0, n - 1, 0, n - 1
    while left < right and up < down:
        # 
        for i in range(left, right):
            matrix[up][i] = number
            number += 1

        # 
        for i in range(up, down):
            matrix[i][right] = number
            number += 1

        # 
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

    if n % 2 != 0: 
        matrix[n // 2][n // 2] = number
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def generate_squre_matrix(N):
    """
    Generate a square matrix of size N x N with increasing values from the center
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
    Expand the dimensions of a tensor by adding new dimensions at the end
    """
    res = tensor.clone()
    for i in range(order):
        res = res.unsqueeze(-1)
    return res


def construct_masks(scores: torch.Tensor, maintain_rate, imp: bool):
    """
    Construct masks based on the scores and maintain rate
    scores: [B, C, H, W]
    imp == True means it is deleting important values
    """
    flatten_score = scores.view(scores.shape[0], -1)
    if maintain_rate == 1:
        mask = torch.zeros_like(scores).to(device).float()
        return mask
    thred_ind = math.floor(flatten_score.shape[-1] * maintain_rate)
    sorted_score, inds = torch.sort(flatten_score, dim=-1, descending=imp)
    if imp:
        thred_value = sorted_score[:, thred_ind]
        mask = expand_dim(thred_value, 3)
        mask = mask.repeat(1, 3, 224, 224)
        mask = torch.where(scores >= mask, 0, 1)
        return mask
    else:
        thred_value = sorted_score[:, thred_ind]
        mask = expand_dim(thred_value, 3)
        mask = mask.repeat(1, 3, 224, 224)
        mask = torch.where(scores <= mask, 0, 1)
        return mask


def test_method(model_name, X: torch.Tensor, e):
    """
    Test a method on the given model and input tensor
    """
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
    lr = 1000
    st = time.time()
    X_.requires_grad = True
    with torch.no_grad():
        pred_label = model_test(X_).argmax(-1)
    for i in range(e):
        X_.requires_grad = True
        
        pred = model_test(X_)
        loss = loss_fn(pred, pred_label)
        loss.backward()
        #print(loss.item())
        X_grad = X_.grad.clone()
        #grad_list[i] = X_grad
        X_.grad.zero_()
        with torch.no_grad():
            X_ = X_ - lr * X_grad
    #save_as_pic(X_[0], 'mymethod/processed.png')
    with torch.no_grad():
        freqs = torch.fft.fft2(X)
        freq_after = torch.fft.fft2(X_)
        mag_ori = torch.abs(freqs)
        ori_after_mutual_energy = 2 * (torch.conj(freq_after) * freqs).real
        scores = (ori_after_mutual_energy / (mag_ori) - mag_ori)
        scores = torch.log(1 - scores.min() + scores)
    print(time.time() - st)
    del model_test, X_, X_grad
    torch.cuda.empty_cache()
    return scores

selected_batch = 143
def del_elements(scores: torch.Tensor, 
                 pics: torch.Tensor, 
                 imp: bool,
                 Freq: bool,
                 rand: bool):
    """
    Delete elements from the input tensor based on the scores and other parameters
    """
    steps = list(range(step_num))
    res_pic = pics.clone()
    # Select a sample here
    select_sample = 26
    del_step = 0
    res_pic = pics.clone()[select_sample].unsqueeze(0)
    scores = scores.to(pics.device)[select_sample].unsqueeze(0)
    origin_pic = res_pic.clone()
    save_as_pic(origin_pic[0], 'origin')
    #heat_map((scores).cpu()[0], 'origin_score', form='png')
    wrong_pic = None
    wrong_score = None
    correct_pic = None
    correct_score = None
    if Freq:
        for i in steps:
            if imp:
                ratio = i / 10000 + 0.0001
            else:
                ratio = i / 200 + 0.95
            #scores = torch.abs(torch.fft.ifft2(scores))
            mask = construct_masks(scores, ratio, imp)
            freqs = torch.fft.fft2(pics)
            masked_pic = torch.fft.ifft2(freqs * mask).real
                
            if i == del_step:
                correct_pic = masked_pic[select_sample].clone()
                save_as_pic(correct_pic, 'correct')
                #correct_score = (scores * mask)
                #heat_map(correct_score.cpu()[0], 'correct_score', form='png')
                break
            #pic4show = masked_pic[0]
            #print(255 * torch.max(torch.abs(unnormalize(pic4show) - unnormalize(pics[0]))))
            #save_as_pic(pic4show, 'inputgrad/test-' + str(i) + '-' + str(imp))
            #heat_map((scores * mask).cpu()[0], 'inputgrad/test-' + str(i) + '-' + str(imp), form='png')
            res_pic = torch.concat([res_pic, masked_pic], dim=0)
    else:
        #scores = torch.abs(torch.fft.fft2(scores)).real
        #freqs = torch.fft.fft2(pics)
        for i in steps:
            if imp:
                ratio = i / 10 + 0.1
            else:
                ratio = i / 10 + 0.1
            #mask = construct_masks(scores, ratio, imp)
            if rand is False:
                mask = construct_masks(scores, ratio, imp)
            else:
                mask = construct_masks(torch.rand_like(scores.real).to(device), ratio, imp)
            masked_pic = pics * mask
            pic4show = masked_pic[0]
            #save_as_pic(pic4show, 'inputgrad/test-' + str(i) + '-' + str(imp))
            #heat_map((torch.abs(scores) * mask).cpu()[0], 'fullgrad/test-' + str(i) + '-' + str(imp), form='png')
            res_pic = torch.concat([res_pic, masked_pic], dim=0)
    minus_pic = origin_pic - correct_pic
    save_as_pic(minus_pic[0], 'difference')
    #minus_score = scores - correct_score
    #heat_map(minus_score.cpu()[0], 'minus_score', form='png')
    return res_pic


def get_change_rate(conf, bs):
    """
    Calculate the change rate of the confidence
    conf: [B, 1000 * step]
    """
    rate = bs * torch.ones(step_num).float().to(device)
    ind = torch.arange(bs).to(device).long()
    base = torch.softmax(conf[:bs, :], dim=-1)
    pred_label = base.argmax(-1)
    base = base[ind, pred_label]
    for i in range(1, step_num):
        conf_step = torch.softmax(conf[bs * i:bs * (i + 1), :], dim=-1)
        conf_step = conf_step[ind, pred_label]
        rate[i] = torch.sum(conf_step / base)
    
    return rate

def get_conf(pics: torch.Tensor, 
             model: nn.Module, 
             scores: torch.Tensor, 
             imp: bool,
             bs: int,
             Freq: bool,
             rand: bool,
             y,
             batch_no):
    """
    Get the confidence of the model on the given pictures
    """
    corr = 0
    #rate = torch.zeros(step_num).float().to(device)
    new_pics = del_elements(scores, pics, imp, Freq, rand)
    #new_pics = del_elements_sort_freq(scores, pics, imp)
    model.eval()
    conf = model(new_pics)
    conf_temp = conf.argmax(-1)
    conf_temp = conf_temp.view(-1, y.shape[0])
    #conf_p = torch.softmax(conf, dim=-1)[torch.arange(11), conf_temp]
    #print(conf_p)
    temp = conf_temp.T
    
    for i in range(y.shape[0]):
        if y[i] in temp[i]:
            corr += 1
            with open('error_sample_analysis/select_sample.txt', 'a') as f:
                print(y[i].item(), ',', temp[i], i, file=f)
    with open('error_sample_analysis/select_sample.txt', 'a') as f:
                print('------------', batch_no, file=f)
    return corr


def print_info(msg, file_name='eval_ch_rate_local.log'):
    """
    Print the message to the console or a log file
    """
    if log_en:
        f = open(file_name, 'a')
        print(msg, file=f)
        f.close()
    else:
        print(msg)


def evaluate(model_name, method_name, imp, e):
    """
    Evaluate the given model and method
    """
    res_path = root_path + method_name + "/" + model_name + "/"
    wrong_sum = 0
    shift_sum = 0
    bs = 200
    #############
    #############
    # shuffle becomes True
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    rate = torch.zeros(step_num).float().to(device)
    rand = False
    scores = sort_score().unsqueeze(0).unsqueeze(0).repeat(bs, 3, 1, 1)
    for batch, (X, y) in enumerate(val_loader):
        if batch != selected_batch:
            continue
        y = y.to(device)
        if model_name == 'ViT':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) 
        model.to(device)
        model.eval()    
        X = X.to(device)
        wrong_samples = None
        pred_label = model(X).argmax(-1)
        del model
        label4wrong = None
        for i in range(X.shape[0]):
            if pred_label[i] == y[i]:
                continue
            else:
                if wrong_samples is None:
                    wrong_samples = X[i].unsqueeze(0)
                    label4wrong = y[i].unsqueeze(0)
                else:
                    wrong_samples = torch.concat([wrong_samples, X[i].unsqueeze(0)], dim=0)
                    label4wrong = torch.concat([label4wrong, y[i].unsqueeze(0)], dim=0)
        if wrong_samples is None:
            continue
        else:
            X = wrong_samples
            y = label4wrong
        if batch % 10 == 0:
            msg = "Method " + method_name + " " + str(batch) + " evaluating"
            print_info(msg)
        
        if method_name in ['fullgrad', 'inputgrad', 'smoothgrad', 'gradcam', 'smooth_grad', 'input_grad']:
            scores = torch.load(res_path + str(batch + 1), map_location=device)
        elif method_name == 'random':
            scores = torch.rand_like(X).to(device)
            rand = True
        elif method_name == 'energy':
            scores = torch.abs(torch.fft.fft2(X))
            scores = torch.log(1 + scores - scores.min())
        elif method_name == 'sort_freq':
            scores = sort_score().unsqueeze(0).unsqueeze(0).repeat(X.shape[0], 3, 1, 1)
        elif method_name == 'test_method':
            with torch.enable_grad():
                scores = test_method(model_name, X, e)
        else:
            scores = torch.load(res_path + str(batch), map_location=device)
        
        if X.shape[0] != scores.shape[0]:
            msg = "Error in " + str(batch)
            print_info(msg)
            continue
        
        Freq = True
        if method_name == 'Freq' or method_name == 'energy' or method_name == 'test_method':
            Freq = True
        if model_name == 'ViT':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) 
        model.to(device)
        model.eval()   
        flag = get_conf(X, model, scores, imp, X.shape[0], Freq, rand, y, batch)
        wrong_sum += X.shape[0]
        shift_sum += flag
        print(shift_sum / wrong_sum, batch)
        
    with open('error_sample_analysis/shift_rate.log', 'a') as f:
        print(method_name, wrong_sum, shift_sum, file=f)
    #print_info(msg, 'temp_ev_res.log')
    return rate


def test_net(model_name):
    """
    Test the given neural network model
    """
    bs = 500
    #############
    #############
    # shuffle becomes True
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    rate = torch.zeros(step_num).float().to(device)
    acc = 0
    for batch, (X, y) in enumerate(val_loader):
        y = y.to(device)
        if model_name == 'ViT':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) 
        model.to(device)
        model.eval()    
        X = X.to(device)
        pred_label = model(X).argmax(-1)
        acc += (pred_label == y).sum().item()
        del model
        torch.cuda.empty_cache()
        
        
    with open('error_sample_analysis/test_net.log', 'a') as f:
        print(model_name, acc / 50000, file=f)
    #print_info(msg, 'temp_ev_res.log')
    return rate


def run():
    """
    Run the evaluation process
    """
    methods = focus_method.keys()
    models = ['ViT', 'ResNet50']
    res_dict = {}
    for method in methods:
        res_dict[method] = {}
        for m in models:
            """if m == 'ViT':
                continue"""
            if m == 'ViT' and method == 'fullgrad':
                continue
            if m == 'test_method':
                continue
            #if method != 'sort_freq':
            #    continue
            res_dict[method][m] = {}
            for imp in [True]:
                for e in [1]:
                    res = evaluate(m, method, imp, e)
                    res_dict[method][m][str(imp)] = res
    #with open('results/Evaluation-ifftFFT-ResultRes.json', 'w', encoding='UTF-8') as f:
    #    json.dump(res_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    with torch.no_grad():
        run()
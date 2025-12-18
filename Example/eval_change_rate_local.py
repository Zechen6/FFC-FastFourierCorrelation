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
#import cv2
import matplotlib.pyplot as plt
import seaborn as sns
########################################################
#################### Global Params #####################

step_num = 10
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"

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
    root='example_data/',
    transform=data_transform)


focus_method = {'ffc':1}


###################################################################



def save_as_pic(tensor:torch.Tensor, fig_name):
    tensor4show = unnormalize(tensor)
    tensor4show = tensor4show.cpu().numpy()
    tensor4show = (tensor4show * 255).astype(np.uint8)

    # turn shapes to (224, 224, 3) 
    tensor4show = np.transpose(tensor4show, (1, 2, 0))

    # save fig
    cv2.imwrite('pics/'+fig_name+'.png', tensor4show)



def expand_dim(tensor:torch.Tensor, order:int):
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
                 rand:bool,
                 model_name = None):
    steps = list(range(step_num))
    res_pic = pics.clone()
    preds = []
    pred_label = None
    #base = None

    for i in steps:
        if imp:
            ratio = i/10+0.1
        else:
            ratio = i/10+0.1
        #scores = torch.abs(torch.fft.ifft2(scores))
        mask = construct_masks(scores, ratio,imp)
        freqs = torch.fft.fft2(pics)
        masked_pic = torch.fft.ifft2(freqs*mask).real
        if model_name is not None:
            if model_name == 'ViT':
                model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            else:
                model = resnet50(weights=ResNet50_Weights.DEFAULT) 
            model.to(device)
            model.eval()
            pred = model(masked_pic)
            if i == 0:
                pred_label = pred.argmax(-1)
                conf = torch.softmax(pred, dim=-1)
                #base = conf[torch.arange(masked_pic.shape[0]), pred_label].clone()
                #print(base)
                #preds.append(pred)
            else:
                conf = torch.softmax(pred, dim=-1)
                modified_conf = conf[torch.arange(masked_pic.shape[0]),pred_label]
            del model
        """pic4show = masked_pic[0]
        print(255*torch.max(torch.abs(unnormalize(pic4show)-unnormalize(pics[0]))))
        save_as_pic(pic4show, 'inputgrad/test-'+str(i)+'-'+str(imp))
        heat_map((scores*mask).cpu()[0],'inputgrad/test-'+str(i)+'-'+str(imp),form='png')"""
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
        #print(rate[i]/bs, i)
    return rate


def get_conf(pics:torch.Tensor, 
             model:nn.Module, 
             scores:torch.Tensor, 
             imp:bool,
             bs:int,
             Freq:bool,
             rand:bool,
             model_name:str):

    new_pics = del_elements(scores, pics, imp, Freq,rand,model_name)

    model.eval()
    conf = model(new_pics)

    return get_change_rate(conf, bs)


def print_info(msg, file_name = 'eval_ch_rate_local.log'):
    if log_en:
        f = open(file_name,'a')
        print(msg, file=f)
        f.close()
    else:
        print(msg)


def evaluate(model_name, method_name, imp, e):
    
    bs = focus_method[method_name]
    #############
    #############
    val_loader = DataLoader(val_dataset,batch_size=bs,shuffle=False)
    rate = torch.zeros(step_num).float().to(device)
    rand = False
    time_sum = 0
    total_X = 0
    for batch,(X,y) in enumerate(val_loader):
        msg = "Method "+method_name+" "+str(batch) + " evaluating"
        print_info(msg)
        X = X.to(device)
        total_X += X.shape[0]
        with torch.enable_grad():
            scores, time_cost = ffc(model_name, X, e)
            time_sum += time_cost
            print("Time:",time_sum/(batch+1), e)


        if X.shape[0] != scores.shape[0]:
            msg = "Error in "+str(batch)+"X:"+str(X.shape[0])+" scores:"+str(scores.shape[0])
            print_info(msg)
            continue
        if model_name == 'ViT':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) 


        model.to(device)
        model.eval()
        rate += get_conf(X,model,scores,imp,X.shape[0],None,rand,model_name)

        msg = str(rate/((batch+1)*bs))
        print_info(msg)

    rate /= total_X
    rate = rate.cpu()
    rate = rate.numpy().tolist()
    msg = method_name+":"+model_name+":"+str(rate)
    print_info(msg,'temp_ev_res.log')
    return rate


def run():
    methods = focus_method.keys()
    models = ['ResNet50','ViT']
    res_dict = {}
    for method in methods:
        res_dict[method] = {}
        for m in models:

            res_dict[method][m] = {}
            for imp in [False, True]:
                for e in [1]:
                    res = evaluate(m, method, imp, e)
                    res_dict[method][m][str(imp)] = res


def ffc(model_name,X:torch.Tensor, e):
    X_ = X.clone()
    if model_name == 'ViT':
        model_test = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    else:
        model_test = resnet50(weights=ResNet50_Weights.DEFAULT) 
    model_test.to(device)
    model_test.eval()
    
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
        
        X_.grad.zero_()
        with torch.no_grad():
            X_ = X_ - lr*X_grad
    with torch.no_grad():
        freqs = torch.fft.fft2(X)  
        freq_after = torch.fft.fft2(X_)
        mag_ori = torch.abs(freqs)
        ori_after_mutual_energy = 2*(torch.conj(freq_after)*freqs).real
        scores = (ori_after_mutual_energy/(mag_ori)-mag_ori)

    ed_time = time.time()

    
    
    del model_test, X_, X_grad
    torch.cuda.empty_cache()
    
    #print("Time:",ed_time-st_time, e)
    return scores, ed_time-st_time


if __name__ == "__main__":
    with torch.no_grad():
        run()


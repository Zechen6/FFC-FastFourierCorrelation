"""
This experiment is used to compare the distinguish ability between
Fourier Feature and Space Domain Feature
"""
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

device = "cuda:3"
""""""
methods_bs_dict = {'fullgrad':8,'IG':16,'smoothgrad':128,
                   'energy':100,
                   'random':16,'inputgrad':100,
                   'grad_cam':128,'test_method':500}
done = ['fullgrad','IG','smoothgrad','inputgrad','random','test_method']
root_path = "/data01/InterpretRes/"

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


def get_concentration(model_name, method_name):
    res_path = root_path+method_name+"/"+model_name+"/"
    if model_name == 'ViT':
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    else:
        model = resnet50(weights=ResNet50_Weights.DEFAULT) 

    Freq = True
    if method_name == 'Freq':
        Freq = True

    model.to(device)
    bs = methods_bs_dict[method_name]
    val_loader = DataLoader(val_dataset,batch_size=bs,shuffle=False)
    same_sample = None
    with torch.no_grad():
        for batch,(X,y) in enumerate(val_loader):

            if batch % 10 == 0:
                msg = "Method "+method_name+" "+str(batch) + " evaluating"
                print(msg)
            X = X.to(device)
            if method_name in ['fullgrad','inputgrad','smoothgrad','gradcam','grad_cam']:
                scores = torch.load(res_path+str(batch+1), map_location=device)
            elif method_name == 'random':
                scores = torch.rand_like(X).to(device)
                rand = True
            elif method_name == 'energy':
                scores = torch.abs(torch.fft.fft2(X))
            elif method_name == 'test_method':
                scores = torch.load(res_path+str(batch)+'.pt', map_location=device)
                scores = torch.where(torch.isnan(scores),0,scores)
            else:
                scores = torch.load(res_path+str(batch), map_location=device)
            if X.shape[0] != scores.shape[0]:
                msg = "Error in "+str(batch)
                print(msg)
                continue
            for i in range(scores.shape[0]):
                if same_sample == None:
                    same_sample = scores[i]
                    l = y[i]
                    continue
                if y[i] == l:
                    same_sample += scores[i]
                if y[i] != l:
                    torch.save(same_sample/50, 'distinguish/scores_only/'+method_name+'-'+model_name+str(l.item())+'.pt')
                    same_sample = None
                    torch.cuda.empty_cache()



def get_inner_class_kurtosis(model_name, method_name):
    # inner label kurtosis
    kurtorsis_list = []
    with torch.no_grad():
        for l in range(999):
            sc = torch.load('distinguish/scores_only/'+method_name+'-'+model_name+str(l)+'.pt',
                            map_location=device)
            sc_t = sc.clone()
            mu_sc = torch.mean(sc)
            sc = torch.where(sc>=mu_sc, 1, 0).float()
            mu_sc = torch.mean(sc)
            sigma_sc = torch.sqrt(torch.var(sc))
            if sigma_sc == 0:
                continue
            sc = (sc-mu_sc)/sigma_sc
            kurtorsis_list.append(torch.mean(torch.pow(sc,4)).item())
            if np.isnan(kurtorsis_list[-1]):
                print('nan appearred')
        with open('distinguish/kurtorsis.log','a') as f:
            print(method_name, np.mean(kurtorsis_list), file=f)



def get_label_among_kurtosis(model_name, method_name):
    score_all = None
    with torch.no_grad():
        for l in range(999):
            sc = torch.load('distinguish/scores_only/'+method_name+'-'+model_name+str(l)+'.pt',
                            map_location=device)
            sc = torch.relu(sc-torch.mean(sc))
            sc = torch.where(sc > 0, 1, 0).float()
            if score_all is None:
                score_all = sc.unsqueeze(0)
            else:
                score_all = torch.concat([score_all, sc.unsqueeze(0)],dim=0)
    
    score_all = score_all.view(score_all.shape[0], -1)
    score_all = score_all.T
    
    mu = torch.mean(score_all, dim=-1,keepdim=True).repeat(1, 999)

    cn = torch.where(mu < (2/1000), 1, 0).sum()
    temp = torch.where(mu == 0, 1, 0).sum()
    print(method_name, (cn-temp)/999)
    #print(mu.shape)
    sc = torch.where(score_all>mu, 1, 0).float()
    #print(sc.sum().item())
    mu = torch.mean(sc, dim=-1,keepdim=True).repeat(1, 999)
    sigma = torch.sqrt(torch.var(sc, dim=-1,keepdim=True)).repeat(1, 999)
    #sigma = torch.where(sigma == 0, 1, sigma)
    sc = (sc-mu)/sigma
    sc = torch.where(sigma == 0, mu, sc)
    kurt = torch.mean(torch.pow(sc,4), dim=-1)
    kurt_mean = torch.mean(kurt).item()
    with open('distinguish/AmongLabelViT.log','a') as f:
        #print("---label---")
        print(method_name, (cn-temp)/999,file=f)





if __name__ == "__main__":
    model_name = "ResNet50"
    for model_name in ['ViT', 'ResNet50']:
        with open('distinguish/AmongLabelViT.log','a') as f:
        #print("---label---")
            print(model_name, file=f)
        with open('distinguish/kurtorsis.log','a') as f:
            print(model_name, file=f)
        for m in methods_bs_dict.keys():
            if m in done:
                continue
            if m in ['fullgrad'] and model_name == 'ViT':
                continue
            if m == 'energy':
                continue
            if m == 'IG' and model_name == 'ResNet50':
                model_name == 'Resnet50'
            else:
                model_name == 'ResNet50'
            get_concentration(model_name, m)
            get_inner_class_kurtosis(model_name, m)
            get_label_among_kurtosis(model_name, m)
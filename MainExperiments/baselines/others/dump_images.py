#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils, models
import os

# Import saliency methods
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.smooth_fullgrad import SmoothFullGrad
from torchvision.models import ResNet152_Weights, DenseNet201_Weights, Inception_V3_Weights, ViT_B_32_Weights, ResNet50_Weights
from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.smoothgrad import SmoothGrad
from torchvision.models import resnet152, densenet201, inception_v3, vit_b_32,resnet50
from misc_functions import *
import time 
import warnings
warnings.filterwarnings("ignore")

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")

# Dataset loader for sample images
data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
val_dataset = datasets.ImageFolder(
    root='/data01/img_net_dataset/val',
    transform=data_transform)

sample_loader = DataLoader(val_dataset,batch_size=32)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])




# Initialize saliency methods


def compute_saliency_and_save(model, model_name):
    saliency_methods = {
    # FullGrad-based methods
    # Other saliency methods from literature
    'input_grad': InputGradient(model),
    'smooth_grad': SmoothGrad(model),
    'fullgrad': FullGrad(model),
    'grad_cam': GradCAM(model),
    }

    # Compute saliency maps for the input data
    time_list ={
        # Other saliency methods from literature
        'smoothgrad':0
    }    

    for s in saliency_methods:
        try:
            os.mkdir(save_path+"/"+s+"/")
            try:
                os.mkdir(save_path+"/"+s+"/"+model_name)
            except:
                pass
        except:
            try:
                os.mkdir(save_path+"/"+s+"/"+model_name)
            except:
                pass
            pass

        # Save saliency maps
        time_sum = 0
        for batch_idx, (data, _) in enumerate(sample_loader):
            data = data.to(device).requires_grad_()
           
            start_time = time.time()
            saliency_map = saliency_methods[s].saliency(data)
            time_sum += time.time() - start_time
            with open('record.log', 'a') as f:
                print(s,time_sum/(batch_idx+1), batch_idx, file=f)
            torch.save(saliency_map, save_path+"/"+s+"/"+model_name+"/"+str(batch_idx+1))
            torch.cuda.empty_cache()
            #for i in range(data.size(0)):
                #filename = save_path + str( (batch_idx+1) * (i+1))
                #image = unnormalize(data[i].cpu())
                #save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg')
            
        end_time = time.time()
        print(s+":"+str(time_sum))
        time_list[s] = end_time-start_time
    print(time_list)

if __name__ == "__main__":
    # Create folder to saliency maps
    save_path = '/data01/InterpretRes/'
    create_folder(save_path)
    for m in ['ResNet50']:
        print(m,"is explaining")
        if m == "ViT":
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.to(device)
        compute_saliency_and_save(model,m)
        print('Saliency maps saved.')





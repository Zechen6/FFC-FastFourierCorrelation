import torch
import torch.nn as nn
from utils import *
from torchvision import datasets
from torchvision import transforms
from torchvision.models import  vit_b_32,resnet50
from torchvision.models import ResNet50_Weights, ViT_B_32_Weights
import time


def interpret_net(model_name='ViT'):
    print('Begin')
    if model_name == 'ViT':
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    else:
        model = resnet50(weights=ResNet50_Weights.DEFAULT) 
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    img_loader = DataLoader(val_dataset,batch_size=128, shuffle=False)
    time_sum = 0
    for batch, (X,y) in enumerate(img_loader):
        X,y = X.to(device),y.to(device)
        start_time = time.time()
        X.requires_grad = True
        pred = model(X)
        pred_label = pred.argmax(-1)
        optimizer.zero_grad()
        loss = loss_fn(pred, pred_label)
        loss.backward()
        X_grad = X.grad.clone()
        with torch.no_grad():
            freq_origin = torch.fft.fft2(X)
            X_new = X * (X_grad)
            freq_new = torch.fft.fft2(X_new,dim=(-2,-1))
            freq_en_origin = torch.abs(freq_origin)
            freq_en_new = torch.abs(freq_new)
            score = freq_en_new - freq_en_origin
        ed_time = time.time()
        if batch == 0:
            continue
        time_sum += ed_time-start_time
        print(time_sum/(batch))
    return score

interpret_net()
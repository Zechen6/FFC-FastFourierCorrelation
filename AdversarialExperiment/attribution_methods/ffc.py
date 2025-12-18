"""
本文件用于实现FFC
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from confs.device_conf import device
import torch
import torch.nn as nn
import math


def select_bottom_element(scores:torch.Tensor,
                          ratio:float):
    """
    本函数用于选取低分特征
    """
    assert ratio <= 1
    scores4sort = scores.view(scores.shape[0],-1)
    thred_idx = torch.tensor(math.ceil(scores4sort.shape[-1]*ratio)).to(scores.device).unsqueeze(-1)
    #thred_idx = thred_idx.repeat(scores.shape[0]).unsqueeze(-1)
    values,_ = torch.sort(scores4sort, dim=-1,descending=False)
    #axis = torch.arange(scores.shape[0]).to(scores.device).unsqueeze(-1)
    #indices = torch.cat([axis,thred_idx],dim=-1)
    thred_values = values[:,thred_idx]
    mask = torch.where(scores4sort>thred_values,1,0)
    mask = mask.view(scores.shape)
    return mask


def select_top_element(scores:torch.Tensor, ratio:float):
    """
    本函数用于选取最高分特征
    """
    assert ratio <= 1
    scores4sort = scores.view(scores.shape[0],-1)
    thred_idx = torch.tensor(math.ceil(scores4sort.shape[-1]*ratio)).to(scores.device).unsqueeze(-1)
    #thred_idx = thred_idx.repeat(scores.shape[0]).unsqueeze(-1)
    values,_ = torch.sort(scores4sort, dim=-1,descending=True)
    #axis = torch.arange(scores.shape[0]).to(scores.device).unsqueeze(-1)
    #indices = torch.cat([axis,thred_idx],dim=-1)
    thred_values = values[:,thred_idx]
    mask = torch.where(scores4sort>thred_values,1,0)
    mask = mask.view(scores.shape)
    return mask
    

def malicious_ffc(net:nn.Module, sample:torch.Tensor, 
                  tgt_label:torch.Tensor, lr=1000, echo=100):
    """
    malicious_ffc用于找到原始样本里和目标类最相似的特征
    """
    y = tgt_label*torch.ones(sample.shape[0]).to(device).long()
    with torch.enable_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        data_new = sample.clone()
        data_new.requires_grad = True
        for e in range(echo):
            pred = net(data_new)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            grad = data_new.grad.data.clone()        
            with torch.no_grad():
                data_new -= lr*grad
                data_new.grad.zero_()
        ori_freq = torch.fft.fft2(sample)
        new_freq = torch.fft.fft2(data_new)
        mag_ori = torch.abs(ori_freq)
        ori_after_mutual_energy = 2*(torch.conj(new_freq)*ori_freq).real       

        scores = (ori_after_mutual_energy/(mag_ori)-mag_ori)
    return scores


def ffc(net:nn.Module, sample:torch.Tensor, lr=1000, echo=100):
    """
    对于单个样本，寻找其最高归因的
    """
    net.eval()
    torch.use_deterministic_algorithms(True)

    with torch.no_grad():
        pred_label = net(sample).argmax(-1)
    with torch.enable_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        data_new = sample.clone()
        data_new.requires_grad = True
        for _ in range(echo):
            pred = net(data_new)
            """print(net.layer4[-1].left[-2])
            print(net.layer4[-1].left[-2].weight[pred_label[0]])"""
            loss = loss_fn(pred, pred_label)
            optimizer.zero_grad()
            loss.backward()
            grad = data_new.grad.data.clone()
            with torch.no_grad():
                data_new -= lr*grad
                data_new.grad.zero_()
        ori_freq = torch.fft.fft2(sample)
        new_freq = torch.fft.fft2(data_new)
        mag_ori = torch.abs(ori_freq)
        ori_after_mutual_energy = 2*(torch.conj(new_freq)*ori_freq).real       

        scores = (ori_after_mutual_energy/(mag_ori)-mag_ori)
    return scores


def find_most_important_feature(samples:torch.Tensor, 
                                net:nn.Module):
    """
    本函数根据FFC来选择最少的高分特征使得网络维持原始判断
    """
    with torch.no_grad():
        net.eval()
        original_pred = net(samples).argmax(-1)
        scores = ffc(net, samples, lr=1000, echo=20)
        founded_flag = torch.zeros(samples.shape[0]).to(device)
        ratio4samples = torch.zeros(samples.shape[0]).to(device)
        step = 0.01
        filtered_sample_list = samples.clone()
        for e in range(30):
            masks = select_top_element(scores, (e+1)*step)
            freq_sample = torch.fft.fft2(samples)
            masked_sample = freq_sample*masks
            filtered_sample = torch.fft.ifft2(masked_sample).real
            new_pred = net(filtered_sample).argmax(-1)
            maintained_flag = (new_pred==original_pred).int()
            new_maintained = maintained_flag-founded_flag
            if (new_maintained==1).sum() > 0:
                filtered_sample_list[new_maintained==1] = filtered_sample[new_maintained==1]
            ratio4samples[new_maintained==1] = (e+1)*step
            founded_flag = torch.where(maintained_flag>founded_flag,maintained_flag, founded_flag)

    return scores, filtered_sample_list


def find_top_malicious_feature(samples:torch.Tensor, 
                                net:nn.Module,
                                tgt_label:torch.Tensor|int):
    """
    本函数根据FFC来选择最少的高分特征使得网络维持原始判断
    """
    filtered_sample_tensor = torch.zeros_like(samples)
    with torch.no_grad():
        net.eval()
        scores = malicious_ffc(net, samples, lr=100, echo=100, tgt_label=tgt_label)

        step = 0.02

        for e in range(10):
            masks = select_top_element(scores, (e+1)*step)
            freq_sample = torch.fft.fft2(samples)
            masked_sample = freq_sample*masks
            filtered_sample = torch.fft.ifft2(masked_sample).real
            new_pred = net(filtered_sample).argmax(-1)
            maintained_flag = (new_pred==tgt_label)
            if maintained_flag.sum() > 0:
                filtered_sample_tensor[maintained_flag] = filtered_sample[maintained_flag]

    return scores, filtered_sample_tensor


def recurrent_filt(net:nn.Module, 
                   sample:torch.Tensor,
                   iters:int):
    """
    通过循环归因来强力的筛选关键特征
    """
    scores, features = find_most_important_feature(sample,net)
    for _ in range(iters):
        scores, features = find_most_important_feature(features, net)
    return scores, features


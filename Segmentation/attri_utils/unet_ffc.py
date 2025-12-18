
"""
本文件用于实现FFC
"""

import torch
import torch.nn as nn
import math
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    thred_idx = torch.where(thred_idx==scores4sort.shape[-1],-1,thred_idx)
    #thred_idx = thred_idx.repeat(scores.shape[0]).unsqueeze(-1)
    values,_ = torch.sort(scores4sort, dim=-1,descending=True)
    #axis = torch.arange(scores.shape[0]).to(scores.device).unsqueeze(-1)
    #indices = torch.cat([axis,thred_idx],dim=-1)
    thred_values = values[:,thred_idx]
    mask = torch.where(scores4sort>thred_values,1,0)
    mask = mask.view(scores.shape)
    return mask


def unet_ffc(net:nn.Module, sample:torch.Tensor, 
              lr=1000, echo=100, loss_fn=torch.nn.CrossEntropyLoss()):
    """
    对于单个样本，寻找其最高归因的
    """
    net.eval()
    with torch.no_grad():
        pred_label = net(sample).round().float()
    with torch.enable_grad():
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        data_new = sample.clone()
        data_new.requires_grad = True
        for _ in range(echo):
            pred = net(data_new)
            loss = loss_fn(pred, pred_label)
            optimizer.zero_grad()
            loss.backward()
            print(loss.item(), _)
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


def confidence_analysis(net:nn.Module, 
                        samples:torch.Tensor, 
                        scores:torch.Tensor):
    """
    用于分析相对置信度变化
    """
    net.eval()
    confidence_list = []
    pred_consistency_list = []
    with torch.no_grad():
        original_logits = net(samples)
        original_probs = torch.softmax(original_logits, dim=-1)
        original_confidences, original_pred = torch.max(original_probs, dim=-1)
        freq_samples = torch.fft.fft2(samples)
        for e in range(9):
            masks = select_top_element(scores, 1-(e+1)*0.1)
            masked_freq_samples = freq_samples*masks
            filtered_samples = torch.fft.ifft2(masked_freq_samples).real

            filtered_logits = net(filtered_samples)
            filtered_probs = torch.softmax(filtered_logits, dim=-1)
            filtered_confidences, filtered_pred = torch.max(filtered_probs, dim=-1)

            confidence_drop = filtered_confidences/original_confidences
            consistent_flag = (original_pred==filtered_pred).int()
            confidence_list.append(confidence_drop.cpu().numpy())   
            pred_consistency_list.append(consistent_flag.cpu().numpy())
    return confidence_list, pred_consistency_list

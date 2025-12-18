"""
本文件用于实现FFC
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
    values,_ = torch.sort(scores4sort, dim=-1,descending=True)
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
    y = tgt_label*torch.ones(sample.shape[0]).to(sample.device).long()
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


def ffc4binary_model(net:nn.Module, sample:torch.Tensor, loss_fn=nn.BCELoss(), lr=1000, echo=100):
    """
    对于单个样本，寻找其最高归因的
    """
    net.eval()
    with torch.no_grad():
        pred_label = net(sample)
        pred_label = torch.where(pred_label > 0.5, 1, 0).float()
    with torch.enable_grad():
        if loss_fn is None:
            loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        data_new = sample.clone()
        data_new.requires_grad = True
        for _ in range(echo):
            pred = net(data_new)
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
        scores = ffc4binary_model(net, samples, lr=1000, echo=20)
        founded_flag = torch.zeros(samples.shape[0]).to(samples.device)
        ratio4samples = torch.zeros(samples.shape[0]).to(samples.device)
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


def confidence_analysis_batch4binary_model(net:nn.Module, 
                        samples:torch.Tensor, 
                        scores:torch.Tensor,
                        step = 0.05):
    """
    用于分析相对置信度变化
    """
    net.eval()
    confidence_list = []
    pred_consistency_list = []
    modified_mel_features = []
    with torch.no_grad():
        original_confidences = net(samples)
        original_pred = (original_confidences>0.5).int()
        original_confidences = torch.where(original_confidences > 0.5,  
                                           original_confidences,
                                           1 - original_confidences)
        freq_samples = torch.fft.fft2(samples)
        all_e = int(1/step)
        for e in range(all_e-1):
            masks = select_top_element(scores, 1-(e+1)*step)
            masked_freq_samples = freq_samples*masks
            filtered_samples = torch.fft.ifft2(masked_freq_samples).real
            filtered_confidences = net(filtered_samples)

            filtered_pred = (filtered_confidences>0.5).int()
            filtered_confidences = torch.where(original_pred == 1,
                                               filtered_confidences,
                                               1 - filtered_confidences)

            relative_confidence = filtered_confidences/original_confidences
            consistent_num = (original_pred==filtered_pred).sum().item()
            confidence_list.append(relative_confidence.cpu().numpy())   
            pred_consistency_list.append(consistent_num)
            modified_mel_features.append(filtered_samples)
        for i in range(len(confidence_list)):
            confidence_list[i] = confidence_list[i].mean().item()

    return confidence_list, pred_consistency_list, modified_mel_features


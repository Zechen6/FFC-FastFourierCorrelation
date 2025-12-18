"""
本文件专门用于寻找特征融合的办法
"""

import sys
cwd = '/data01/lzc/Experiments/NonPoisonBackdoor/'
sys.path.insert(0, cwd)
from confs.device_conf import device
from net_structures.resnet18 import ResNet18
from dataset_arranger.load_cifar10 import load_clean_cifar10
import torch
import torch.nn as nn
from attribution_methods.ffc import find_most_important_feature, select_top_element
from draw_tools.tensor2img import tensor2img
import random as rd
LOG_NAME = 'logs/cifar10-trigger.log'
POTENTIAL_FEATURE_PATH='cache/cifar10/potential_trigger_feature/'
CIFAR_CLASSNUM = 10


def replace_based_merge_features(tgt_features:torch.Tensor, 
                   victim_sample:torch.Tensor,
                   scores:torch.Tensor,
                   pretrained_net:nn.Module,
                   tgt_label:torch.Tensor,
                   original_pred:torch.Tensor,
                   tgt_net:nn.Module):
    """
    2025/11/12
    本函数根据特征的评分高低，对特征进行替换
    """
    top_mask = select_top_element(scores, 0.04)
    top_mask2 = select_top_element(scores, 0.04)
    signals = torch.fft.fft2(victim_sample)
    top_signals = signals*top_mask
    rest_signals = signals*(1-top_mask)
    tgt_signals = torch.fft.fft2(tgt_features)
    potential = tgt_signals*top_mask2
    merged_signal = torch.fft.ifft2(1*rest_signals+0.8*potential).real
    pred = pretrained_net(merged_signal).argmax(-1)
    tgt_pred = tgt_net(merged_signal).argmax(-1)
    success_pretrained = (pred==tgt_label)
    success_tgt = (tgt_pred==tgt_label)
    overlap = success_pretrained&success_tgt
    #print(success_pretrained.sum().item())
    indices = torch.nonzero(overlap).squeeze(-1)
    if len(indices) > 0:
        tensor2img(merged_signal[indices[rd.randint(0,len(indices)-1)]], 'cache/cifar10/merged_picture/temp.png')
    return merged_signal, overlap.sum().item(), success_pretrained.sum().item()


def replace_based_merge_features_v2(tgt_features:torch.Tensor, 
                   victim_sample:torch.Tensor,
                   scores:torch.Tensor,
                   pretrained_net:nn.Module,
                   tgt_label:torch.Tensor,
                   original_pred:torch.Tensor,
                   tgt_net:nn.Module):
    """
    本函数在单纯根据top_mask替换的基础上
    即2025/11/12版本的replace_based_merge_features
    改成:将所有的top_mask都给填满
    """
    top_mask = select_top_element(scores, 0.08)
    signals = torch.fft.fft2(victim_sample)
    top_signals = signals*top_mask
    rest_signals = signals*(1-top_mask)
    tgt_signals = torch.fft.fft2(tgt_features)
    mask = torch.where(tgt_signals.abs()>1e-5,1,0)
    selected = (mask-top_mask).abs().view(mask.shape[0],-1).sum(-1)
    potential = tgt_signals[selected==0]
    merged_signal = torch.fft.ifft2(rest_signals+potential).real
    pred = pretrained_net(merged_signal).argmax(-1)
    tgt_pred = tgt_net(merged_signal).argmax(-1)
    success_pretrained = (pred==tgt_label)
    success_tgt = (tgt_pred==tgt_label)
    overlap = success_pretrained&success_tgt
    print(overlap.sum().item())
    all_success = success_pretrained|success_tgt
    print(all_success.sum().item())
    asr = (overlap).sum().item()/((all_success).sum().item()+1e-7)
    #tensor2img(merged_signal[5], 'cache/cifar10/merged_picture/temp.png')
    return merged_signal, asr


def avg_agg_feature(tgt_features:torch.Tensor):
    """
    尝试形成一个整体的tgt_feature
    最直觉的是直接取平均值
    """
    agg_features = tgt_features.mean(dim=0,keepdim=True)
    return agg_features


def select_agg_feature(tgt_features:torch.Tensor):
    """
    尝试形成一个整体的tgt_feature
    现在是把每个频率振幅最大的给保留
    """
    B,C,W,H = tgt_features.shape
    tgt_freqs:torch.Tensor = torch.fft.fft2(tgt_features)
    tgt_freq_mags = tgt_freqs.abs().view(tgt_features.shape[0],-1)
    tgt_freqs = tgt_freqs.view(tgt_features.shape[0],-1)
    v,idx=tgt_freq_mags.max(dim=0)
    temp = tgt_freqs[idx,torch.arange(idx.shape[0])]
    agg_features = temp.reshape(1,C,W,H)
    return torch.fft.ifft2(agg_features).real


def find_best_mislead_features(tgt_features:torch.Tensor, 
                   victim_sample:torch.Tensor,
                   scores:torch.Tensor,
                   pretrained_net:nn.Module,
                   tgt_label:torch.Tensor,
                   original_pred:torch.Tensor,
                   tgt_net:nn.Module):
    top_mask = select_top_element(scores, 0.04)
    top_mask2 = select_top_element(scores, 0.04)
    signals = torch.fft.fft2(victim_sample)
    top_signals = signals*top_mask
    rest_signals = signals*(1-top_mask)
    tgt_signals = torch.fft.fft2(tgt_features)
    potential = tgt_signals*top_mask2
    merged_signal = torch.fft.ifft2(1*rest_signals+0.8*potential).real
    pred = pretrained_net(merged_signal).argmax(-1)
    tgt_pred = tgt_net(merged_signal).argmax(-1)
    success_pretrained = (pred==tgt_label)
    success_tgt = (tgt_pred==tgt_label)
    overlap = success_pretrained&success_tgt
    return success_pretrained, success_tgt, overlap


"""
基于之前的test_trigger方法
本文件用于优化注入后的视觉效果
"""

import sys
cwd = '/data01/lzc/Experiments/NonPoisonBackdoor/'
sys.path.insert(0, cwd)
from confs.device_conf import device
from net_structures.resnet18 import ResNet18
from net_structures.resnet34 import ResNet34_CIFAR10
from dataset_arranger.load_cifar10 import load_clean_cifar10
import torch
import torch.nn as nn
from cifar10trigger.merge_feature import replace_based_merge_features, avg_agg_feature, select_agg_feature
from cifar10trigger.merge_feature import find_best_mislead_features
from attribution_methods.ffc import malicious_ffc, select_top_element, find_top_malicious_feature
from attribution_methods.ffc import find_most_important_feature
from draw_tools.tensor2img import tensor2img
import random as rd
import math
import warnings
warnings.filterwarnings('ignore')
LOG_NAME = 'logs/cifar10-trigger.log'
POTENTIAL_FEATURE_PATH='cache/cifar10/potential_trigger_feature/'
CIFAR_CLASSNUM = 10


def load_tgt_feature(label:int):
    """
    本函数用于加载tgt_label的样本
    """
    return torch.load(POTENTIAL_FEATURE_PATH+str(label)+'.pt',map_location=device)


def load_tgt_test_feature(label:int):
    """
    本函数用于加载tgt_label的样本
    """
    return torch.load(POTENTIAL_FEATURE_PATH+'test-'+str(label)+'.pt',map_location=device)


def test_trigger_asr(tgt_sample:torch.Tensor,
                    net:nn.Module,
                    tgt_label:torch.Tensor|int,
                    tgt_features:torch.Tensor,
                    tgt_net:nn.Module,
                    pic_name:int):
    ratio = 0.005 # 选取top0.005的触发器字典的特征加入输入样本
    const_ratio = 1 # 触发器特征的振幅的缩放因子
    B,C,W,H = tgt_features.shape
    """tensor2img(tgt_sample[0],
               save_path=f'cache/cifar10/4figure/original-feature{pic_name}.png')"""
    scores, features = find_most_important_feature(tgt_sample, net)
    iter_times = 2 # 保留多少原始输入样本的特征，数值越大，保留越多
    for e in range(iter_times):
        scores_recurrent, features = find_most_important_feature(features, net)
        """tensor2img(features,
               save_path=f'cache/cifar10/4figure/replaced-feature{pic_name}.png')"""

    tgt_freqs = torch.fft.fft2(tgt_features)
    ba_freqs = torch.fft.fft2(features)
    ba_masks = torch.where(ba_freqs.abs()>1e-5, 1, 0)
    #ba_masks[:,0,0] = 0
    
    ba_ratio = ba_masks.sum()/(3*32*32)
    if ba_ratio > ratio:
        ba_masks = select_top_element(scores, ratio)
    mask4tgt_feature = select_top_element(scores, ratio)

    tgt_masks = torch.where(tgt_freqs.abs()>1E-5,1,0)
    sample_freqs = torch.fft.fft2(tgt_sample)
    tgt_mask_sum = tgt_masks.view(tgt_features.shape[0],-1).sum(dim=-1)
    tgt_mask_mask = torch.where(tgt_mask_sum>ratio,0,1)
    tgt_masks = tgt_mask_mask[:,None,None,None]*tgt_masks
    #ba_top_feature = torch.fft.ifft2(ba_freqs*ba_masks).real
    
    features4select = torch.fft.ifft2(const_ratio*tgt_freqs*(mask4tgt_feature+tgt_masks+ba_masks)+\
                                      (sample_freqs*(1-ba_masks))).real
    tgt_pred = torch.softmax(tgt_net(features4select),dim=-1)
    sor_pred = torch.softmax(net(features4select),dim=-1)
    tgt_suc = (tgt_pred.argmax(-1)==tgt_label)
    sor_suc = (sor_pred.argmax(-1)==tgt_label)
    if sor_suc.sum().item() == 0:
        return 0, 0, 0
    overlapping_rate = (tgt_suc*sor_suc).sum().item()/sor_suc.sum().item()
    print(f'Overlapping Rate:{overlapping_rate:.3f}')
    """tensor2img(features4select[tgt_pred[:,tgt_label].argmax(-1)],
               save_path=f'cache/cifar10/4figure/triggered_feature{pic_name}.png')
    inserted_feature = torch.fft.ifft2(const_ratio*tgt_freqs*(mask4tgt_feature+tgt_masks+ba_masks))[tgt_pred[:,tgt_label].argmax(-1)]
    tensor2img(inserted_feature,
               save_path=f'cache/cifar10/4figure/inserted-feature{pic_name}.png')"""
    return (tgt_suc*sor_suc).sum().item(), sor_suc.sum().item(), overlapping_rate
    """label_conf = pred[:,tgt_label]
    #print(label_conf.max())
    similar_tgt_freq = tgt_freqs[label_conf.argmax(-1)]*mask4tgt_feature
    indices = torch.nonzero(pred.argmax(-1)==tgt_label).squeeze()
    selected_tgt_masks = tgt_masks[label_conf.argmax(-1)]
    triggered_features = features4select[indices]
    if len(indices.shape) == 0:
        indices = indices.unsqueeze(0)
    print(len(indices))    
    if len(indices) < 2:
        triggered_features = triggered_features.unsqueeze(0)
    if len(indices) == 0:
        return 0
    triggered_features =  triggered_features.mean(dim=0,keepdim=True)
    tensor2img(tgt_sample[0], 'cache/cifar10/merged_picture/tgt_sample.png')
    tensor2img(ba_top_feature[0], 'cache/cifar10/merged_picture/ba_top_feature.png')
    tensor2img(ma_top_feature[0], 'cache/cifar10/merged_picture/ma_top_feature.png')
    tensor2img(triggered_features, 'cache/cifar10/merged_picture/triggered_feature.png')"""
    """maintained = sample_freqs*(1-ba_masks-selected_tgt_masks)
    swapped_sample = const_ratio*similar_tgt_freq+maintained
    merged_sample = torch.fft.ifft2(swapped_sample).real
    tensor2img(merged_sample[0],'cache/cifar10/merged_picture/merged_sample.png')
    pred = tgt_net(merged_sample).argmax(-1)
    print(pred==tgt_label, pred.item(), tgt_label)
    return (pred==tgt_label).int().item()"""



if __name__ == "__main__":
    """
    1、首先先采用4和5,攻击其他所有的类,并以此做参数分析
    2、再验证强类与弱类的关系，就是2攻击4、5，0、8、9攻击4、5....
    """
    tgt_label = 0
    vul_label = [0,1,2,3,4,5,6,7,8,9]
    vul_label.remove(tgt_label)
    net = ResNet18()
    net.load_state_dict(torch.load('pretrained_model/cifar10_resnet18_model.pt'))
    net.to(device)
    net.eval()
    tgt_net = ResNet34_CIFAR10()
    tgt_net.load_state_dict(torch.load('pretrained_model/cifar10_resnet34_model.pt'))
    tgt_net.to(device)
    tgt_net.eval()
    tgt_features = load_tgt_feature(tgt_label) # 加载触发器字典
    tgt_features = tgt_features[torch.randperm(1000)]
    tgt_test_features = load_tgt_test_feature(tgt_label)
    train_loader, test_loader = load_clean_cifar10(1,True)
    rate = 0
    total = 0
    tgt_suc_total = 0
    sor_suc_total = 0
    ovr_total = 0
    query_time_99 = -1
    tgt_suc_total_test = 0
    sor_suc_total_test = 0
    ovr_total_test = 0
    query_time_99_test = -1
    asr = 0
    asr_test = 0
    for batch_idx, (X,y) in enumerate(test_loader):
        if y == tgt_label or y not in vul_label: # vul_label是受害者类
            continue
        X = X.to(device)
        total += X.shape[0]
        with torch.no_grad():
            # tgt_suc是触发器字典成功目标网络的样本数量
            # sor_suc是触发器字典成功触发提取器网络的样本数量
            # ovr是同时触发了目标网络和触发器提取器网络的样本的占比
            tgt_suc, sor_suc, ovr = test_trigger_asr(X,net,
                                                    tgt_label,tgt_test_features,tgt_net,
                                                    batch_idx)
            tgt_suc_test, sor_suc_test, ovr_test = test_trigger_asr(X,net,
                                                    tgt_label,tgt_features,tgt_net,
                                                    batch_idx)
            
        tgt_suc_total += tgt_suc
        sor_suc_total += sor_suc
        ovr_total += ovr
        if ovr_total != 0:
            query_time_99 = math.log(0.01, (1-ovr_total/total))
        
        if ovr > 0:
            asr += 1
        
        tgt_suc_total_test += tgt_suc_test
        sor_suc_total_test += sor_suc_test
        ovr_total_test += ovr_test
        if ovr_total_test != 0:
            query_time_99_test = math.log(0.01, (1-ovr_total_test/total))
        
        if ovr_test > 0:
            asr_test += 1
        print('----------------')
        print(f'Total Rate:{ovr_total/total:.3f}, \
              ASR:{asr/total:.3f}, \
                99% Query Times:{math.ceil(query_time_99)}')
        print('Test')
        print(f'Total Rate:{ovr_total_test/total:.3f}, \
              ASR:{asr_test/total:.3f}, \
                99% Query Times:{math.ceil(query_time_99_test)}')

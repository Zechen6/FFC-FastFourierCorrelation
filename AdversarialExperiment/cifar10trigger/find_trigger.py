"""
本文件用于挖掘cifar10中的潜在触发器
即把cifar10中最高分的少量特征给找出来
"""
import sys
cwd = '/data01/lzc/Experiments/NonPoisonBackdoor/'
sys.path.insert(0, cwd)
from confs.device_conf import device
from net_structures.resnet18 import ResNet18
from dataset_arranger.load_cifar10 import load_clean_cifar10
import torch
import torch.nn as nn
from attribution_methods.ffc import find_most_important_feature
LOG_NAME = 'logs/cifar10-trigger.log'
POTENTIAL_FEATURE_PATH='cache/cifar10/potential_trigger_feature/test-'
CIFAR_CLASSNUM = 10

def potential_trigger_cifar():
    """
    利用归因算法来找到每个类的触发器字典
    """
    net = ResNet18()
    net.load_state_dict(torch.load('pretrained_model/cifar10_resnet18_model.pt'))
    net.to(device)
    train_loader, test_loader = load_clean_cifar10(batch_size=128,shuffle=True)
    
    #for e in range(10):
        # 利用数据增强来增加样本量
    filtered_sample_list = [[] for _ in range(CIFAR_CLASSNUM)]
    finnished_X = 0
    for batch_idx, (X,y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        att_scores, filtered_X = find_most_important_feature(X, net)
        #filtered_X = torch.cat(filtered_X, dim=0)
        for l in range(CIFAR_CLASSNUM):
            temp = filtered_X[y==l]
            if len(temp) > 0:
                filtered_sample_list[l].append(temp)
        finnished_X += X.shape[0]
        print(f'{finnished_X}/10000')
    for l in range(CIFAR_CLASSNUM):
        temp = torch.cat(filtered_sample_list[l],dim=0)
        torch.save(temp, POTENTIAL_FEATURE_PATH+str(l)+'.pt')


if __name__ == "__main__":
    potential_trigger_cifar()
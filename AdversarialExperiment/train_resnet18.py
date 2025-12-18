"""
本文件用于预训练一个CIFAR10上的Resnet18网络
"""
from confs.device_conf import device
from net_structures.resnet18 import ResNet18
from dataset_arranger.load_cifar10 import load_clean_cifar10
import torch
import torch.nn as nn
LOG_NAME = 'logs/cifar10-resnet18-noaug.log'

def pretrain_cifar():
    train_loader, test_loader = load_clean_cifar10(data_aug=True)
    pre_trained_net = ResNet18()
    pre_trained_net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_def = torch.optim.Adam(pre_trained_net.parameters(), lr=0.001)
    max_test_acc = 0
    max_train_acc = 0
    pre_trained_net.train()
    for e in range(100):
        loss_sum = 0
        train_acc = 0 #
        test_acc = 0
        asr = 0
        train_sample_num = 0
        test_sample_num = 0
        dirty_sample_num = 0
        for batch_idx_train, (X,y) in enumerate(train_loader):
            X = X.to(device)
            #with torch.no_grad():
            #    X = filt_freq_by_mag(X)
            y = y.to(device)
            pred = pre_trained_net(X)
            loss = loss_fn(pred, y) # 采用毒化的标签训练
            optimizer_def.zero_grad()
            loss.backward()
            optimizer_def.step()
            loss_sum += loss.item()
            train_acc += (pred.argmax(-1)==y).sum().item()
            train_sample_num += X.shape[0]

        pre_trained_net.eval()
        for batch_idx, (X,y) in enumerate(test_loader):
            X = X.to(device)
            #with torch.no_grad():
            #    X = filt_freq_by_mag(X)

            y = y.to(device)
            pred = pre_trained_net(X)
            test_acc += (pred.argmax(-1)==y).sum().item() # 以干净标签为目标
            test_sample_num += X.shape[0]
        
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(pre_trained_net.state_dict(),'pretrained_model/cifar10_resnet18_tgt_model.pt')
        with open(LOG_NAME, 'a') as f:
            print(f'Epoch:{e}, Loss{loss_sum/batch_idx_train}, \n\
                Training Acc:{train_acc/train_sample_num}, \n\
                Test Acc:{test_acc/test_sample_num},\n', file=f)
            

def align_train_cifar():
    """
    本函数用于和中毒模型进行对齐
    """
    train_loader, test_loader = load_clean_cifar10(data_aug=False)
    pre_trained_net = ResNet18()
    pre_trained_net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_def = torch.optim.Adam(pre_trained_net.parameters(), lr=0.001)
    max_test_acc = 0
    max_train_acc = 0
    pre_trained_net.train()
    for e in range(100):
        loss_sum = 0
        train_acc = 0 #
        test_acc = 0
        asr = 0
        train_sample_num = 0
        test_sample_num = 0
        dirty_sample_num = 0
        for batch_idx_train, (X,y) in enumerate(train_loader):
            X = X.to(device)
            #with torch.no_grad():
            #    X = filt_freq_by_mag(X)
            y = y.to(device)
            pred = pre_trained_net(X)
            loss = loss_fn(pred, y) # 采用毒化的标签训练
            optimizer_def.zero_grad()
            loss.backward()
            optimizer_def.step()
            loss_sum += loss.item()
            train_acc += (pred.argmax(-1)==y).sum().item()
            train_sample_num += X.shape[0]

        pre_trained_net.eval()
        for batch_idx, (X,y) in enumerate(test_loader):
            X = X.to(device)
            #with torch.no_grad():
            #    X = filt_freq_by_mag(X)

            y = y.to(device)
            pred = pre_trained_net(X)
            test_acc += (pred.argmax(-1)==y).sum().item() # 以干净标签为目标
            test_sample_num += X.shape[0]
        
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        if train_acc > max_train_acc:
            max_train_acc = train_acc
            torch.save(pre_trained_net.state_dict(),'pretrained_model/cifar10_resnet18_align_model.pt')
        with open(LOG_NAME, 'a') as f:
            print(f'Epoch:{e}, Loss{loss_sum/batch_idx_train}, \n\
                Training Acc:{train_acc/train_sample_num}, \n\
                Test Acc:{test_acc/test_sample_num},\n', file=f)


if __name__ == "__main__":
    align_train_cifar()
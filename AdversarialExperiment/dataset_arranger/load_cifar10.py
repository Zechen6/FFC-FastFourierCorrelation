import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        tensor: C×H×W
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # x = x * std + mean
        return tensor
    

def load_clean_cifar10(batch_size=128, shuffle=True, data_aug=True):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # 四周补 4 像素后随机裁剪
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),   # 三通道均值
                                (0.2023, 0.1994, 0.2010)),  # 三通道方差
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),   # 三通道均值
                                (0.2023, 0.1994, 0.2010)),  # 三通道方差
        ])

    # --- 测试集无增强 ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # --- 下载/加载数据 ---
    train_dataset = datasets.CIFAR10(
        root='./clean_dataset',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root='./clean_dataset',
        train=False,
        download=True,
        transform=test_transform
    )

    # --- DataLoader ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    return train_loader, test_loader
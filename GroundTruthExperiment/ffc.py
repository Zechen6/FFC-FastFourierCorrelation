import torch
import torch.nn as nn
import os
import numpy as np
from gen_gt_data import MLP,train_mlp
import json

def ffc1d(net:nn.Module, X:torch.Tensor,lr:float, echo:int):
    """
    Docstring for ffc1d
    """
    net.eval()
    with torch.no_grad():
        logits = net(X)
        pred = logits.argmax(-1)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
    
    with torch.enable_grad():
        X_new = X.data.clone()
        X_new.requires_grad = True
        for e in range(echo):
            logits = net(X_new)
            loss = loss_fn(logits, pred)
            optimizer.zero_grad()
            loss.backward()
            X_grad = X_new.grad.data.clone()
            with torch.no_grad():
                X_new = X_new - lr*X_grad
            X_new.requires_grad = True    
    with torch.no_grad():
        original_signal = torch.fft.fft(X)
        modified_signal = torch.fft.fft(X)
        original_signal_mag = original_signal.abs()
        proj = 2*original_signal*modified_signal.conj()/(original_signal_mag+1e-7)
        scores = proj - original_signal_mag

    return scores.real


def get_top1_feature(scores:torch.Tensor, samples:torch.Tensor):
    """
    因为第一个是直流信号，剩下两个有共轭，所以是前三个
    """
    samples_freq = torch.fft.fft(samples)
    sort_scores, ind = torch.sort(scores, dim=-1, descending=True)
    thred = sort_scores[:,1].unsqueeze(-1)
    mask = torch.where(scores>=thred,1,0)
    maintained_signal = torch.fft.ifft(samples_freq*mask).real
    return maintained_signal, mask


def build_signal_matrix(y, class_signals):
    """
    输入：
        y: shape = (N,)   每个样本的标签（可被打乱）
        class_signals: dict[class_id] = clean signal (shape = sample_dim)

    输出：
        signal_matrix: shape = (N, sample_dim)
                       每行是 y[i] 对应类别的 clean signal
    """
    N = len(y)
    sample_dim = len(next(iter(class_signals.values())))
    signal_matrix = torch.zeros((N, sample_dim))

    for i in range(N):
        cls = y[i].item()
        signal_matrix[i] = torch.tensor(class_signals[cls])

    return signal_matrix


def interpret_mlp(sample_dim, num_classes):

    datas = torch.load('SampleInfos.dat')
    X = datas['X']
    y = datas['y']
    class_feature = datas['class_feature']
    model = MLP(sample_dim,num_classes)
    model.load_state_dict(torch.load('model_params.pt'))
    scores = ffc1d(model, X, 1000, 20)
    filtered_X, mask = get_top1_feature(scores, X)
    class_dict = {}
    for item in class_feature:
        complex_freq_index1 = class_feature[item]['complex_freq_index']
        complex_freq_index2 = sample_dim-complex_freq_index1
        class_dict[item] = [complex_freq_index1, complex_freq_index2]
    
    acc = 0
    include_rate = 0
    redundent_rate = 0
    for i in range(mask.shape[0]):
        indices = torch.nonzero(mask[i])
        label = class_dict[y[i].item()]
        indices = np.sort(indices.detach().numpy())
        label = np.sort(label)
        if len(label) == len(indices):
            temp = abs(label[0]-indices[0])+abs(label[1]-indices[1])
            if temp == 0:
                acc += 1
        if (label[0] in indices):
            include_rate += 0.5
        if (label[1] in indices):
            include_rate += 0.5
        
        redundent_rate += len(indices)/2

    return acc/mask.shape[0], include_rate/mask.shape[0], redundent_rate/mask.shape[0]

def test_main():
    sample_dim=200
    num_samples = 2000
    res_dict = {}
    n_class = 4
    res_dict = {}
    for noise_var in range(1,51):
        net_acc = train_mlp(n_class, noise_var/10, sample_dim, num_samples)
        int_acc, incl_rate, redun_rate = interpret_mlp(sample_dim, n_class)
        res_dict[noise_var/10] = {}
        res_dict[noise_var/10]['net_acc'] = net_acc
        res_dict[noise_var/10]['interpret'] = int_acc
        res_dict[noise_var/10]['include_rate'] = incl_rate
        res_dict[noise_var/10]['redundent_rate'] = redun_rate

    with open('res.json','w') as f:
        json.dump(res_dict, f, indent=4)


if __name__ == "__main__":
    test_main()

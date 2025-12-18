import torch
import numpy as np
from datasets import load_from_disk
from transformers import BertForSequenceClassification
import pickle
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
import json

# 配置
output_dir = "/data/FeiyangZhang_Genedigger/cache/test20251104/"
datestamp_min = "251104"  # 根据你的实际运行日期调整
output_prefix = "cm_classifier_test"
res_save_path = 'lzc_folder/cache/train_attribution/'

# 加载模型
model_path = f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/"
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载测试数据
test_data = load_from_disk("/data/FeiyangZhang_Genedigger/cache/test20251104/cm_classifier_test_labeled_test.dataset")

class GeneDataset(Dataset):
    def __init__(self, dataset, valid_types=None, start_index=0):
        self.dataset = dataset
        self.valid_types = valid_types
        self.start_index = start_index
        
        # 筛选索引（代替你原来的 if 条件）
        self.indices = []
        for i in range(len(dataset)):
            if i < start_index:
                continue
            if valid_types is not None:
                if dataset[i]['cell_type'] not in valid_types:
                    continue
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_index = self.indices[idx]
        sample = self.dataset[real_index]
        return real_index, sample  # 返回实际的索引，便于保存结果


# ---------------------------
# 2. collate_function，适配 DataLoader
# ---------------------------
def collate_fn(batch):
    # batch = [(index, sample)]
    indices, samples = zip(*batch)
    return indices, samples

# 加载类别字典
with open(f"{output_dir}/{output_prefix}_id_class_dict.pkl", 'rb') as f:
    id_class_dict = pickle.load(f)

class_id_dict = {v: k for k, v in id_class_dict.items()}

print(f"Model loaded from: {model_path}")
print(f"Number of test samples: {len(test_data)}")
print(f"Classes: {id_class_dict}")


def mask_with_scores(scores, ratio):
    sorted_scores, idx = torch.sort(scores.view(scores.shape[0], -1), 
    dim=-1, descending=False)
    thredshold = sorted_scores[:, int(ratio*sorted_scores.shape[-1])].unsqueeze(1)
    mask = torch.where(scores > thredshold, 1, 0)
    return mask


def ffc(model, input_ids, lr, epochs):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    embeddings = model.bert.embeddings.word_embeddings(input_ids)
    embeddings_new = embeddings.detach().clone()
    embeddings_new.requires_grad = True
    for e in range(epochs): 
        # 清零之前的梯度
        if embeddings_new.grad is not None:
            embeddings_new.grad.zero_()
        
        outputs = model(inputs_embeds=embeddings_new)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).long()
        loss = loss_fn(logits, pred_label)
        loss.backward()
        
        # 保存梯度并更新embeddings
        embeddings_grad = embeddings_new.grad.data.clone()
        with torch.no_grad():
            embeddings_new.data = embeddings_new.data - lr * embeddings_grad
        
        model.zero_grad()

    freq_origin = torch.fft.fft2(embeddings)
    freq_new = torch.fft.fft2(embeddings_new)
    scores = 2*freq_origin*freq_new.conj()/freq_origin.abs()-freq_origin.abs()
    
    return scores.real

def compute_gradients(model, input_ids, attention_mask, target_class):
    """
    计算输入相对于目标类别的梯度
    
    参数:
        model: 训练好的模型
        input_ids: 输入token IDs (batch_size, seq_len)
        attention_mask: 注意力掩码
        target_class: 目标类别索引
    
    返回:
        gradients: 相对于embedding的梯度
    """
    model.eval()
    
    # 获取embedding层
    embeddings = model.bert.embeddings.word_embeddings(input_ids)
    embeddings.requires_grad = True
    
    # 前向传播
    outputs = model(inputs_embeds=embeddings)
    logits = outputs.logits
    
    # 对目标类别的logit计算梯度
    target_logit = logits[0, target_class]
    target_logit.backward()
    
    # 返回梯度
    gradients = embeddings.grad
    
    return gradients, logits


def dig_analysis(model, input_ids, scores, original_logits, num_ratios=10):
    """
    完整 batch 版本
    返回:
        maintain_rates:  list[num_ratios] of (B,) tensors
        relative_confidence: list[num_ratios] of (B,) tensors
    """
    B = input_ids.shape[0]

    original_pred_label = torch.argmax(original_logits, dim=-1)  # (B,)
    original_probs = softmax(original_logits, dim=-1)
    original_conf = original_probs[torch.arange(B), original_pred_label]  # (B,)

    embeddings = model.bert.embeddings.word_embeddings(input_ids)

    if scores.dim() == 3:
        scores = scores.mean(dim=-1)

    maintain_rates = []
    relative_confidence = []

    with torch.no_grad():
        for r_i in range(num_ratios):
            r = r_i / num_ratios

            mask = mask_with_scores(scores, ratio=r)     # (B, L)
            mask = mask.unsqueeze(-1)                    # (B, L, 1)

            emb_freq = torch.fft.fft2(embeddings)
            emb_new = torch.fft.ifft2(emb_freq * mask).real

            outputs = model(inputs_embeds=emb_new)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)

            pred_label = torch.argmax(logits, dim=-1)
            pred_conf = probs[torch.arange(B), pred_label]

            maintain_rates.append((pred_label == original_pred_label).sum().item())   # (B,)
            relative_confidence.append((pred_conf / original_conf).mean().item())                # (B,)

    return maintain_rates, relative_confidence, embeddings


def analyze_batch(model, samples, id_class_dict):
    """
    批量分析 batch
    samples: list of dict，长度 = batch_size
    返回:
        maintain_rates:  (batch_size, num_ratios)
        relative_confs:  (batch_size, num_ratios)
    """
    batch_size = len(samples)

    # ---- 构造 batch input_ids ----
    max_len = 2048

    # 提取每个样本的长度


    # 统一 pad 到 max_len
    padded_ids = []
    padded_attn = []

    for s in samples:
        ids = s["input_ids"]
        pad_len = max_len - len(ids)

        padded_ids.append(ids + [0] * pad_len)               # input_ids pad = 0
        padded_attn.append([1] * len(ids) + [0] * pad_len)   # attention_mask pad = 0

    input_ids = torch.tensor(padded_ids).to(device)
    attention_mask = torch.tensor(padded_attn).to(device)

    # ---- 原始 logits ----
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits   # (B, C)

    # ---- 批量 FFC attribution ----
    scores = ffc(model, input_ids, lr=1000, epochs=10)   # (B, L, H) 或 (B, L)

    # ---- 批量 deletion-insertion ----
    maintain_rates, relative_confidence, _ = dig_analysis(
        model, input_ids, scores, logits
    )
    # maintain_rates: list of length num_ratios，每个是 int
    # relative_confidence: list of length num_ratios，每个是 tensor(B)

    

    return maintain_rates, relative_confidence

    

if __name__ == "__main__":


    # ---------------------------
    # 3.1 构建 Dataset & DataLoader
    # ---------------------------
    valid_types = ['Cardiomyocyte1', 'Cardiomyocyte2', 'Cardiomyocyte3']


    dataset = GeneDataset(test_data, valid_types=valid_types, start_index=0)

    dataloader = DataLoader(
        dataset,
        batch_size=32,     # 你的代码是单样本处理
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # ---------------------------
    # 3.2 准备统计变量
    # ---------------------------
    avg_maintains = []
    avg_confs = []

    all_maintains = []   # list of (B, R)
    all_confs = []       # list of (B, R)

    for batch_id, (indices, samples) in enumerate(dataloader):

        print(f"Processing batch {batch_id}, size = {len(samples)}")

        maintain_rates, relative_confidence = analyze_batch(model, samples, id_class_dict)
        # shapes:
        # maintain_rates: (B, R)
        # relative_confidence: (B, R)

        all_maintains.append(maintain_rates)
        all_confs.append(relative_confidence)

    # ----------- 汇总所有 batch -----------
    avg_maintains = [0 for _ in range(10)]
    for maintain in all_maintains:
        for i in range(len(maintain)):
            avg_maintains[i] += maintain[i]

    avg_confs = [0 for _ in range(10)]
    for conf in all_confs:
        for i in range(len(conf)):
            avg_confs[i] += conf[i]
    for i in range(len(avg_maintains)):
        avg_maintains[i] /= len(test_data)
        avg_confs /= len(all_confs)

    print("Avg Maintains =", avg_maintains)
    print("Avg Relative Confs =", avg_confs)




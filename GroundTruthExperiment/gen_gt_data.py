import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# ----------- 生成数据集 -----------

def generate_frequency_dataset_with_complex_signals(
    num_classes: int,
    noise_var: float,
    sample_dim: int,
    num_samples: int,
):
    """
    使用复数频域的共轭对生成时域实信号。
    
    返回：
        X: (num_samples, sample_dim) 带噪声信号
        y: (num_samples,) 标签
        class_signals: {
            cls: {
                "clean_signal": 时域实信号，
                "complex_spectrum": 完整复数频谱，
                "complex_freq_index": 选用的频率索引 k
            }
        }
    """

    X = np.zeros((num_samples, sample_dim))
    y = np.zeros(num_samples, dtype=int)

    class_signals = {}

    samples_per_class = num_samples // num_classes

    for cls in range(num_classes):

        # -------- 1. 为每个类随机选择一个复数频率 k --------
        # 避免 0 和 Nyquist 点（它们必须为实数）
        k = np.random.randint(1, sample_dim // 2)

        # 构建频域（复数），长度 sample_dim
        spectrum = np.zeros(sample_dim, dtype=np.complex128)

        # 随机生成共轭对的实部和虚部
        real_part = np.random.uniform(0.5, 1.5)
        imag_part = np.random.uniform(-1.0, 1.0)

        # 设置共轭对
        spectrum[k] = real_part + 1j * imag_part
        spectrum[-k] = real_part - 1j * imag_part

        # -------- 2. IFFT 得到时域实信号 --------
        clean_signal = np.fft.ifft(spectrum).real
        clean_signal = clean_signal / np.max(np.abs(clean_signal))  # 归一化

        # -------- 3. 保存用于后续分析的复数频域信息 --------
        class_signals[cls] = {
            "clean_signal": clean_signal.copy(),
            "complex_spectrum": spectrum.copy(),
            "complex_freq_index": k
        }

        # -------- 4. 按类别生成带噪声样本 --------
        idx_start = cls * samples_per_class
        idx_end = idx_start + samples_per_class

        for idx in range(idx_start, idx_end):
            noise = np.random.normal(0, np.sqrt(noise_var), sample_dim)
            X[idx] = clean_signal + noise
            y[idx] = cls
 
    # -------- 5. 多出样本（若无法整除） --------
    idx = samples_per_class * num_classes
    while idx < num_samples:
        cls = np.random.randint(0, num_classes)
        clean_signal = class_signals[cls]["clean_signal"]
        noise = np.random.normal(0, np.sqrt(noise_var), sample_dim)
        X[idx] = clean_signal + noise
        y[idx] = cls
        idx += 1

    return X, y, class_signals


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ================================================
# 3. 主训练流程
# ================================================
def train_mlp(num_classes,
        noise_var,
        sample_dim,
        num_samples,):


    X, y, class_feature = generate_frequency_dataset_with_complex_signals(
        num_classes=num_classes,
        noise_var=noise_var,
        sample_dim=sample_dim,
        num_samples=num_samples,
    )

    #print(class_feature)

    # 转成 PyTorch Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 划分数据集 (80% train, 20% test)
    split = int(0.8 * num_samples)
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    y_train, y_test = y_tensor[:split], y_tensor[split:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=128,
        shuffle=False
    )

    # ----------- 初始化模型 -----------
    model = MLP(input_dim=sample_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ----------- 训练 -----------
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #print(f"Epoch {epoch+1}/{epochs} | Loss = {total_loss:.4f}")

    # ----------- 测试准确率 -----------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    print(f"\nTest Accuracy: {correct / total:.4f}")
    
    torch.save(model.state_dict(),'model_params.pt')
    torch.save({'X':X_train,'y':y_train,'class_feature':class_feature},'SampleInfos.dat')
    return correct / total

# ================================================
# 运行训练
# ================================================
if __name__ == "__main__":
    train_mlp(num_classes = 4,
            noise_var = 1,
            sample_dim = 200,
            num_samples = 2000)

# Config file containing all of the model hyperparameters.

# ResNet parameters
dropout = 0.3   # <<< 修改点 1: 提高 Dropout 率 (从 0.3 提高到 0.5)
# Learning rate
lr = 3e-5
# 权重衰减 (L2 正则化) - 解决严重过拟合的关键
weight_decay = 1e-4 # <<< 添加点 2: 增加 L2 正则化 (Weight Decay)
max_overrun = 5
epochs = 200
batch_size = 128
pretrained = True
# Number of classes for multi class classification
n_classes = 3

# Parameters for generating the log mel spectrograms used during training.
# Architectural constants.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.
# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 4000 # 8000 for Yaseen, else 4000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 10
MEL_MAX_HZ = 2000
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 4.0 # 0.5 for Yaseen, else 4.0
EXAMPLE_HOP_SECONDS = 1.0 # 0.01 for Yaseen, else 1.0
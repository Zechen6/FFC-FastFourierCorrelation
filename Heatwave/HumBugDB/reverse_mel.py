import librosa
import numpy as np
import soundfile as sf
from librosa.filters import mel as librosa_mel
from Config import hyperparameters
from HumBugDB.LogMelSpecs.compute_LogMelSpecs import waveform_to_examples
from scipy.interpolate import interp1d

def mel_to_audio(
    mel_spec,
    sr=4000,
    n_fft=100,
    hop_length=40,
    n_mels=64,
    fmin=10,
    fmax=2000,
    normalize=True,
    clip_val=0.99
):
    """
    将 log-mel spectrogram 转换回 waveform (单通道)
    mel_spec: shape [frames, n_mels] 或 [frames, n_mels, channels]
    """
    # squeeze 多余维度
    mel_spec = mel_spec.squeeze()
    
    # 如果多通道，取第 0 通道
    if mel_spec.ndim == 3:  # [frames, n_mels, channels]
        mel_spec = mel_spec[0, :, :]

    # log-mel → mel power
    mel_power = librosa.db_to_power(mel_spec)  # shape [frames, n_mels]

    # 构造 mel filter bank
    mel_basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)

    # mel → linear power spectrum
    linear_spec = np.dot(inv_mel_basis, mel_power.T)  # [freq_bins, frames]

    # Griffin-Lim 重建 waveform
    audio = librosa.griffinlim(
        linear_spec,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=n_fft,
        n_iter=60
    )
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val  # [-1,1]
        audio = audio * clip_val      # [-clip_val, clip_val]

    return audio


def get_index_mapping(idx, records):
    """
    将扁平化 idx 映射回 (patient_id, segment_id)
    records: list of list of segments per patient
    idx: 整体扁平化的索引
    """
    index_map = []
    for p_id, seg_list in enumerate(records):
        for s_id in range(len(seg_list)):
            index_map.append((p_id, s_id))

    if idx < 0 or idx >= len(index_map):
        raise ValueError(f"idx {idx} out of range. Total segments={len(index_map)}")

    return index_map[idx]


def reconstruct_modified_audio(
    idx,
    spectrogram_modified,
    records,
    output_path_modified,
    output_path_original=None,
    sr=4000
):
    # 1. 找到对应 patient/segment
    patient_id, segment_id = get_index_mapping(idx, records)
    original_audio = records[patient_id][segment_id]
    print("Patient ID:",patient_id)
    # 2. 保存原音频
    if output_path_original:
        sf.write(output_path_original, original_audio, sr)

    # 3. 计算原 Mel spectrogram（与训练一致）
    mel_orig = waveform_to_examples(original_audio, sr).cpu().numpy()
    
    # 4. 计算每帧的振幅缩放因子
    # 保证 mel_spec 不为零，避免除零
    scale = np.maximum(spectrogram_modified / (mel_orig + 1e-8), 0)

    # 5. 对原音频做幅度调制
    # 将原 waveform 切分为与 mel 帧对应的窗口
    hop_length = int(hyperparameters.STFT_HOP_LENGTH_SECONDS * sr)
    frame_length = int(hyperparameters.STFT_WINDOW_LENGTH_SECONDS * sr)
    audio_modified = np.copy(original_audio)

    for i in range(scale.shape[0]):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio_modified):
            end = len(audio_modified)

        # 取 scale[i] 的平均值，转为单个幅度系数
        factor = np.mean(scale[i])  
        audio_modified[start:end] *= factor  # 只用第 0 个 mel band 平均幅度

    # 6. 防爆音
    """max_val = np.max(np.abs(audio_modified))
    if max_val > 0:
        audio_modified = 0.99 * audio_modified / max_val"""

    # 7. 保存修改音频
    sf.write(output_path_modified, audio_modified, sr)
    print("Modified audio saved:", output_path_modified)

    return original_audio, audio_modified


def reconstruct_modified_audio_fine(
    spectrogram_modified,
    current_records,
    output_path_modified,
    output_path_original=None,
    sr=4000,
    n_fft=256,
    hop_length=64,
    n_mels=64,
    fmin=10,
    fmax=2000,
    highfreq_threshold=0.05  # 高频阈值，超过该比例会压缩，
):
    import librosa
    import numpy as np
    import soundfile as sf
    from librosa.filters import mel as librosa_mel
    from scipy.interpolate import interp1d

    original_audio = current_records

    if output_path_original:
        sf.write(output_path_original, original_audio, sr)
        print("Original audio saved:", output_path_original)

    # 处理 spectrogram
    spectrogram_proc = np.squeeze(spectrogram_modified)
    if spectrogram_proc.shape[-1] != n_mels:
        spectrogram_proc = spectrogram_proc[..., -n_mels:]
    if spectrogram_proc.shape[0] != n_mels:
        spectrogram_proc = spectrogram_proc.T
    if spectrogram_proc.ndim == 3:
        spectrogram_proc = spectrogram_proc[:, :, 0]

    # STFT
    stft_orig = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length)
    mag_orig, phase_orig = np.abs(stft_orig), np.angle(stft_orig)

    # Mel filter
    mel_basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_orig = mel_basis @ (mag_orig**2)

    # 对齐时间帧
    if spectrogram_proc.shape[1] != mel_orig.shape[1]:
        f = interp1d(
            np.linspace(0, 1, spectrogram_proc.shape[1]),
            spectrogram_proc,
            kind='linear',
            axis=1
        )
        spectrogram_proc = f(np.linspace(0, 1, mel_orig.shape[1]))

    mel_orig = np.maximum(mel_orig, 1e-8)
    scale_mel = spectrogram_proc / mel_orig
    scale_freq = mel_basis.T @ scale_mel  # [freq_bins, time_frames]

    # -------------------------------
    # 高频抑制
    # scale_freq.shape = [freq_bins, time_frames]
    if highfreq_threshold is not None: # 启动听感优化
        freq_bins = scale_freq.shape[0]
        # 高频只是为了听感上更好
        highfreq_start = int(freq_bins * 0.2)  # 高频起始位置，可调
        scale_freq[highfreq_start:, :] = highfreq_threshold
    # -------------------------------

    # 振幅缩放
    mag_modified = mag_orig * scale_freq

    # ISTFT
    audio_modified = librosa.istft(
        mag_modified * np.exp(1j * phase_orig),
        hop_length=hop_length,
        length=len(original_audio)
    )

    # 防爆音
    max_val_orig = np.max(np.abs(original_audio))
    max_val_mod = np.max(np.abs(audio_modified))

    if max_val_mod > 0:
        audio_modified = audio_modified * (max_val_orig / max_val_mod)

    sf.write(output_path_modified, audio_modified, sr)
    print("Modified audio saved:", output_path_modified)

    return original_audio, audio_modified
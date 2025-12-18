import torch
import sys
cwd = '/data01/lzc/Experiments/FFC/HeartWave/'
if cwd not in sys.path:
    sys.path.insert(0,cwd)
from HumBugDB.runTorch import ResnetDropoutFull
from interpret_utils.ffc import ffc4binary_model, confidence_analysis, confidence_analysis_batch4binary_model
import os
import numpy as np
import argparse
from DataProcessing.net_feature_extractor import net_feature_loader
from train_resnet import create_model
from plot_curve import plot_conf
import scipy
import scipy.io.wavfile
import scipy.io
from HumBugDB.LogMelSpecs.compute_LogMelSpecs import waveform_to_examples
from HumBugDB.reverse_mel import reconstruct_modified_audio_fine
from Config import hyperparameters
import matplotlib.pyplot as plt

DATA_PATH = "data/stratified_data/cv_False/seed_14/"

def plot_scores_heatmap(mat1, titles=None, save_path=None):
    """
    输入三个矩阵，绘制三张并排热力图

    参数：
        mat1, mat2, mat3: 2D numpy 数组
        titles: 可选，长度为3的标题列表，例如 ['A', 'B', 'C']
    """
    matrices = [mat1]
    if titles is None:
        titles = ['Scores']

    plt.figure(figsize=(15, 4))
    
    for i, mat in enumerate(matrices):
        plt.imshow(mat, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(titles[i])
        plt.xlabel("X")
        plt.ylabel("Y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def load_interpret_model(model_path:str):
    model = ResnetDropoutFull(dropout=0.5, bayesian=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_wav_file(filename):
    frequency, recording = scipy.io.wavfile.read(filename)
    return recording, frequency

def load_spectrograms_yaseen(file_path):

    mel_specs = list()
    recording, frequency = load_wav_file(file_path)
    recording = recording / 32768
    mel_spec = waveform_to_examples(recording, frequency)
    mel_specs.append(mel_spec)

    return mel_specs

def list_wav_files(data_directory):
    wav_files = []
    subfolder_names = []

    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                subfolder_names.append(os.path.basename(root))
    
    return wav_files, subfolder_names


def param_loader():
    parser = argparse.ArgumentParser(prog="DBResAndXGBoostIntegration")
    
    # 修复: 匹配用户原始命令行参数 --full_data_directory
    parser.add_argument(
        "--full_data_directory",
        type=str,
        dest="data_directory",
        help="The directory containing all of the data.",
        default="physionet.org/files/circor-heart-sound/1.0.3/training_data",
    )
    
    parser.add_argument(
        "--stratified_directory",
        type=str,
        help="The directory to store the split data.",
        default="data/stratified_data",
    )
    parser.add_argument(
        "--vali_size", type=float, default=0.16, help="The size of the test split."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="The size of the test split."
    )
    
    # 修复: 将 cv 更改为 action="store_true"
    parser.add_argument(
        "--cv", action="store_true", help="Whether to run cv."
    )
    
    # 修复: 添加缺失的 random_state 参数
    parser.add_argument(
        "--random_state", type=int, default=14, help="The random seed for data splitting."
    )
    
    parser.add_argument(
        "--recalc_features",
        action="store_true",
        help="Whether or not to recalculate the log mel spectrograms used as "
        "input to the ResNet.",
    )
    parser.add_argument(
        "--no-recalc_features", dest="recalc_features", action="store_false"
    )
    parser.set_defaults(recalc_features=True)
    parser.add_argument(
        "--spectrogram_directory",
        type=str,
        help="The directory in which to save the spectrogram training data.",
        default="data/spectrograms",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The ResNet to train. Current options are resnet50 or resnet50dropout.",
        choices=["resnet50", "resnet50dropout"],
        default="resnet50dropout",
    )
    parser.add_argument(
        "--recalc_output",
        action="store_true",
        help="Whether or not to recalculate the output from DBRes.",
    )
    parser.add_argument(
        "--no-recalc_output", dest="recalc_output", action="store_false"
    )
    parser.set_defaults(recalc_output=True)
    parser.add_argument(
        "--dbres_output_directory",
        type=str,
        help="The directory in which DBRes's output is saved.",
        default="data/dbres_outputs",
    )
    parser.add_argument(
        '--disable-bayesian', 
        dest='bayesian', 
        action='store_false', 
        default=True,
        help='Disable Bayesian features (default: Bayesian is enabled)'
    )

    args = parser.parse_args()

    # 保持模型名称与 Bayesian 标志的关联逻辑不变 (修复历史上的 TypeError)
    if "dropout" in args.model_name:
        args.bayesian = True
    else:
        args.bayesian = False
    return args


def find_sharp_change(x, threshold=None):
    x = np.array(x)
    diff = np.diff(x)        # x[i+1] - x[i]
    
    if threshold is None:
        threshold = diff.mean() - 2 * diff.std()
    if threshold < 0:
        drop_points = np.where(diff < threshold)[0]
    if threshold > 0:
        drop_points = np.where(diff > threshold)[0]
    return drop_points  # 返回下降点的 index


def cut_records_to_patches(current_records, sr=4000):
    """
    将原始 waveform 切成与 waveform_to_examples 生成的 spectrogram 对齐的 patch。
    current_records: list of 1D np.array，每条音频
    sr: 采样率
    """
    records_patches = []  # 每条音频对应多个 patch

    hop_length = int(hyperparameters.STFT_HOP_LENGTH_SECONDS * sr)
    frame_length = int(hyperparameters.STFT_WINDOW_LENGTH_SECONDS * sr)
    example_window = int(round(hyperparameters.EXAMPLE_WINDOW_SECONDS / hyperparameters.STFT_HOP_LENGTH_SECONDS))
    example_hop = int(round(hyperparameters.EXAMPLE_HOP_SECONDS / hyperparameters.STFT_HOP_LENGTH_SECONDS))

    for audio in current_records:
        # 计算 log mel patch 的数量
        num_frames = 1 + max(0, (len(audio) - frame_length) // hop_length)
        mel_specs = waveform_to_examples(audio, sr).cpu().numpy()
        num_patches = mel_specs.shape[0]

        audio_patches = []
        for i in range(num_patches):
            start = i * example_hop * hop_length
            end = start + example_window * hop_length
            if end > len(audio):
                end = len(audio)
            audio_patch = audio[start:end]
            audio_patches.append(audio_patch)
        records_patches.append(audio_patches)

    return records_patches


def gen_report(
    data_directory,
    stratified_directory,
    test_size,
    vali_size,
    cv,
    random_state,  # <<< 修复: 添加 random_state 参数
    recalc_features,
    spectrogram_directory,
    model_name,
    recalc_output,
    dbres_output_directory,
    bayesian
):
    device = 'cuda:1'
    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    classes_name = "binary_present"
    # 51.94%的正常样本

    
    # 3. 构造实际的、正确的子目录路径
    split_dir = DATA_PATH

    train_data_directory = os.path.join(split_dir, "train_data")
    vali_data_directory = os.path.join(split_dir, "vali_data")
    #test_data_directory = os.path.join(split_dir, "test_data")
    
    (
        spectrograms_train,
        murmurs_train,
        outcomes_train,
        spectrograms_valid,
        murmurs_valid,
        outcomes_valid,
        original_records
    ) = net_feature_loader(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
    )



    X_train = spectrograms_train.to(device)
    #X_test = spectrograms_valid.to(device)
    def load_patient_data(filename):
        with open(filename, "r") as f:
            data = f.read()
        return data
    # 29045_TV
    current_patient_data = load_patient_data(os.path.join(train_data_directory,'2530.txt'))
    
    def load_spectrograms(data_directory, data):
        def get_num_locations(data):
            num_locations = None
            for i, l in enumerate(data.split("\n")):
                if i == 0:
                    num_locations = int(l.split(" ")[1])
                else:
                    break
            return num_locations
        num_locations = get_num_locations(data)
        recording_information = data.split("\n")[1 : num_locations + 1]

        mel_specs = list()
        record_list = []
        for i in range(num_locations):
            entries = recording_information[i].split(" ")
            recording_file = entries[2]
            filename = os.path.join(data_directory, recording_file)
            recording, frequency = load_wav_file(filename)
            
            recording = recording / 32768
            record_list.append(recording)
            mel_spec = waveform_to_examples(recording, frequency)
            mel_specs.append(mel_spec)
        return mel_specs, record_list # 第三个是TV
    current_spectrograms, current_records = load_spectrograms(train_data_directory, current_patient_data)
    spectrograms_tensor = torch.cat([torch.tensor(x, dtype=torch.float32) for xs in current_spectrograms for x in xs], dim=0)
    # 29045的前面两个分别为19和18,TV有19
    # 如果需要加 channel 维度 (num_examples, 1, num_frames, num_bands)
    current_records = current_records[2]
    record_patches = cut_records_to_patches([current_records])[0]
    X = spectrograms_tensor[37:37+19, None, :, :].to(device)
    idx = 1
    if classes_name == "binary_present":
        knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
        for i in range(len(murmurs_train)):
            if (
                torch.argmax(murmurs_train[i]) == 1
                or torch.argmax(murmurs_train[i]) == 2
            ):
                knowledge_train[i, 1] = 1
            else:
                knowledge_train[i, 0] = 1
        knowledge_test = torch.zeros((murmurs_valid.shape[0], 2))
        for i in range(len(murmurs_valid)):
            if torch.argmax(murmurs_valid[i]) == 1 or torch.argmax(murmurs_valid[i]) == 2:
                knowledge_test[i, 1] = 1
            else:
                knowledge_test[i, 0] = 1
        y_train = knowledge_train.to(device)
        y_test = knowledge_test.to(device)
        model, training = create_model(model_name, 2, bayesian)
        
    else:
        raise ValueError("classes_name must be one of outcome, murmur or knowledge.")
    if classes_name == "binary_present":
        model_path = "data/models/model4interpret/model_BinaryPresent_final.pth"
    if classes_name == "binary_unknown":
        model_path = "data/models/model4interpret/model_BinaryUnknown_final.pth"

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    with torch.no_grad():

        X = X.repeat(1,3,1,1)[idx].unsqueeze(0)
        y = y_train[idx]
        conf = model(X)
        pred = torch.where(conf > 0.5, 1, 0)
        
        step = 0.1
        scores = ffc4binary_model(model, X, lr=10,echo=50)
        
        plot_scores_heatmap(torch.log(torch.fft.fftshift(scores[0,0,:,:], dim=(-2,-1)).real.cpu()),
                            'Scores In Mel Feature',
                            save_path='interpret_report/scores.png')
        confidence_list, pred_consistency_list, modified_features =\
              confidence_analysis_batch4binary_model(model, X, scores,step=step)
        thred = 0.05
        buttom = True
        if buttom:
            temp = find_sharp_change(confidence_list,threshold=-thred)
        else:
            temp = find_sharp_change(confidence_list,threshold=thred)
        if len(temp) == 0:
            print('No sharp drop found.')
            conf_sharp_change = -1
        else:
            conf_sharp_change = temp[0]+1  # 因为 diff 导致 index 偏移了 1
        highlight_points = {conf_sharp_change:('r','Confidence Sharp Change')}
        label_reversed = -1
        try:
            label_reversed = pred_consistency_list.index(0)
            highlight_points[label_reversed] = ('g','Pred Changed')
        except:
            print('Pred didn\'t change.')
            pass
        
        plot_conf(confidence_list,'interpret_report/conf_curve.pdf', highlight_points, step=step)
        reconstruct_modified_audio_fine(
            spectrogram_modified=modified_features[conf_sharp_change].cpu().numpy(),current_records=record_patches[idx],
            output_path_modified='interpret_report/ConfidenceRapidChange.wav',
            output_path_original='interpret_report/Original.wav'
            )
        if label_reversed is not None:
            reconstruct_modified_audio_fine(
            spectrogram_modified=modified_features[label_reversed].cpu().numpy(),current_records=record_patches[idx],
            output_path_modified='interpret_report/PredLastPersist.wav',
            output_path_original=None
            )

if __name__ == "__main__":
    args = param_loader()
    with torch.no_grad():
        gen_report(
            args.data_directory,
            args.stratified_directory,
            args.test_size,
            args.vali_size,
            args.cv,
            args.random_state,  # <<< 修复: 传递 random_state 参数
            args.recalc_features,
            args.spectrogram_directory,
            args.model_name,
            args.recalc_output,
            args.dbres_output_directory,
            args.bayesian
        )


    
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


DATA_PATH = "data/stratified_data/cv_False/seed_14/"


def load_interpret_model(model_path:str):
    model = ResnetDropoutFull(dropout=0.5, bayesian=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


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


def interpret_main(
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
    device = 'cuda:0'
    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    classes_name = "binary_present"

    
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
    ) = net_feature_loader(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
    )

    X_train = spectrograms_train.to(device)
    X_test = spectrograms_valid.to(device)

    
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
        
    elif classes_name == "binary_unknown":
        knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
        for i in range(len(murmurs_train)):
            if (
                torch.argmax(murmurs_train[i]) == 0
                or torch.argmax(murmurs_train[i]) == 2
            ):
                knowledge_train[i, 1] = 1
            else:
                knowledge_train[i, 0] = 1
        knowledge_test = torch.zeros((murmurs_valid.shape[0], 2))
        for i in range(len(murmurs_valid)):
            if torch.argmax(murmurs_valid[i]) == 0 or torch.argmax(murmurs_valid[i]) == 2:
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
    print(X_test.shape[0])
    batch_size = 128
    with torch.no_grad():
        conf_list = []
        consistence_list = []
        total_shape = 0
        for e in range((X_test.shape[0]-1)//batch_size+1):
            X = X_test[e*batch_size:(e+1)*batch_size].repeat(1,3,1,1)
            scores = ffc4binary_model(model, X, lr=1000,echo=50)
            confidence_list, pred_consistency_list,_ = confidence_analysis_batch4binary_model(model, X, scores)
            total_shape += X.shape[0]
            print('Processed batch {}/{}'.format(e+1, (X_test.shape[0]-1)//batch_size+1))
            
            conf_list.append(confidence_list)
            consistence_list.append(pred_consistency_list)
            if total_shape >= X_test.shape[0]:
                break
            temp = [0 for _ in range(len(conf_list[0]))]
            for conf in conf_list:
                temp = [temp[i]+conf[i] for i in range(len(conf))]
            temp = [temp[i]/len(conf_list) for i in range(len(temp))]
            print('Confidence List:', np.array(temp))
            print('Pred Consistency List:', np.array(pred_consistency_list)/X.shape[0])
        conf_agg = [0 for _ in range(len(conf_list[0]))]
        for conf in conf_list:
            conf_agg = [conf_agg[i]+conf[i] for i in range(len(conf))]
        conf_agg = [conf_agg[i]/len(conf_list) for i in range(len(conf_agg))]
        print("Train Confidence Analysis:")
        print(conf_agg)
        consist_agg = [0 for _ in range(len(consistence_list[0]))]
        for consist in consistence_list:
            consist_agg = [consist_agg[i]+consist[i] for i in range(len(consist))]
        consist_agg = [consist_agg[i]/X_test.shape[0] for i in range(len(consist_agg))]
        print("Train Pred Consistency Analysis:")
        print(consist_agg)



def baseline_main(
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

    device = 'cuda:0'
    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    classes_name = "binary_present"

    
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
    ) = net_feature_loader(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
    )

    X_train = spectrograms_train.to(device)
    X_test = spectrograms_valid.to(device)

    
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
        
    elif classes_name == "binary_unknown":
        knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
        for i in range(len(murmurs_train)):
            if (
                torch.argmax(murmurs_train[i]) == 0
                or torch.argmax(murmurs_train[i]) == 2
            ):
                knowledge_train[i, 1] = 1
            else:
                knowledge_train[i, 0] = 1
        knowledge_test = torch.zeros((murmurs_valid.shape[0], 2))
        for i in range(len(murmurs_valid)):
            if torch.argmax(murmurs_valid[i]) == 0 or torch.argmax(murmurs_valid[i]) == 2:
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
    print(X_test.shape[0])
    batch_size = 128
    with torch.no_grad():
        conf_list = []
        consistence_list = []
        total_shape = 0
        for e in range((X_test.shape[0]-1)//batch_size+1):
            X = X_test[e*batch_size:(e+1)*batch_size].repeat(1,3,1,1)
            scores = torch.fft.fft2(X).abs()
            confidence_list, pred_consistency_list = confidence_analysis_batch4binary_model(model, X, scores)
            total_shape += X.shape[0]
            print('Processed batch {}/{}'.format(e+1, (X_test.shape[0]-1)//batch_size+1))
            print('Confidence List:', np.array(confidence_list))
            print('Pred Consistency List:', np.array(pred_consistency_list)/X.shape[0])
            conf_list.append(confidence_list)
            consistence_list.append(pred_consistency_list)
            if total_shape >= X_test.shape[0]:
                break
        conf_agg = [0 for _ in range(len(conf_list[0]))]
        for conf in conf_list:
            conf_agg = [conf_agg[i]+conf[i] for i in range(len(conf))]
        conf_agg = [conf_agg[i]/len(conf_list) for i in range(len(conf_agg))]
        print("Train Confidence Analysis:")
        print(conf_agg)
        consist_agg = [0 for _ in range(len(consistence_list[0]))]
        for consist in consistence_list:
            consist_agg = [consist_agg[i]+consist[i] for i in range(len(consist))]
        consist_agg = [consist_agg[i]/X_test.shape[0] for i in range(len(consist_agg))]
        print("Train Pred Consistency Analysis:")
        print(consist_agg)



if __name__ == "__main__":
    args = param_loader()
    interpret_main(
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
    """baseline_main(
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
    )"""

    
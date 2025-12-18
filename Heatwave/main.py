import argparse
import os
import shutil # 导入 shutil 用于删除目录

from dbres import calculate_dbres_scores
from data_splits import stratified_test_vali_split
from train_resnet import run_model_training
from xgboost_integration import calculate_xgboost_integration_scores


def main(
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
    
    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    # 构造将要创建的目录的完整路径
    target_split_dir = os.path.join(
        stratified_directory, f"cv_{cv}", f"seed_{random_state}"
    )

    # 修复 1: 检查目录是否存在，并删除，以避免 FileExistsError
    if os.path.exists(target_split_dir):
        print(f"Warning: Deleting existing split directory: {target_split_dir}")
        shutil.rmtree(target_split_dir)

    # 2. 执行分层划分，创建目录 (使用关键字参数避免参数顺序错误)
    stratified_test_vali_split(
        stratified_features=stratified_features,
        data_directory=data_directory,
        stratified_directory=stratified_directory,
        test_size=test_size,
        vali_size=vali_size,
        cv=cv,
        random_states=[random_state], 
    )
    
    # 3. 构造实际的、正确的子目录路径
    split_dir = target_split_dir

    train_data_directory = os.path.join(split_dir, "train_data")
    vali_data_directory = os.path.join(split_dir, "vali_data")
    test_data_directory = os.path.join(split_dir, "test_data")
    
    # 4. 训练模型
    run_model_training(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
        model_name,
        "OutComeBinary",
        "data/models",
        "outcome_binary",
        bayesian,
        None,
    )

    """run_model_training(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
        model_name,
        "BinaryUnknown",
        "data/models",
        "binary_unknown",
        bayesian,
        None,
    )"""

    # 5. 修复 2: calculate_dbres_scores 调用 (解决最新的 TypeErrors)
    dbres_scores = calculate_dbres_scores(
        recalc_output=recalc_output,
        # 修正参数名 (data_directory 和 output_directory)
        data_directory=test_data_directory, 
        output_directory=dbres_output_directory, 
        
        # <<< 关键修正：添加缺失的两个参数 >>>
        model_name=model_name,
        model_binary_pth="data/models/model_OutComeBinary.pth", 
        # model_binary_pth 在这里作为通用路径，虽然名字重复了，但函数需要它
        
        # 保持不变的特定模型路径
        model_binary_present_pth="data/models/model_BinaryPresent.pth",
        model_binary_unknown_pth="data/models/model_BinaryUnknown.pth",
    )

    # 6. 修复 3: calculate_xgboost_integration_scores 调用
    xgb_scores = calculate_xgboost_integration_scores(
        # 修正参数名和路径
        train_data_directory=train_data_directory,
        test_data_directory=test_data_directory,
        output_directory=dbres_output_directory,

        # 修复参数类型：将 recordings_file 设置为 空字符串 ""
        model_name=model_name,
        model_xgb_pth="data/models/xgb_model_murmur.bin",
        dbres_output_directory=dbres_output_directory,
        model_binary_pth="data/models/data/models/model_OutComeBinary.pth",
        recordings_file="", # <<< 核心修复：必须是字符串
        use_weights=False,
        
        # 保持特定的模型路径不变
        model_binary_present_pth="data/models/model_BinaryPresent.pth",
        model_binary_unknown_pth="data/models/model_BinaryUnknown.pth",
        bayesian=bayesian
    )

    return dbres_scores, xgb_scores


if __name__ == "__main__":

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

    dbres_scores, xgb_scores = main(**vars(args))
"""
本文件用于分析能让网络进行分类的最小特征数量
"""

import sys
cwd = '/data01/lzc/Experiments/FFC/UNet/'
sys.path.insert(0, cwd)

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from predict import load_model, preprocess_image, predict_mask
import numpy as np
from attri_utils.unet_ffc import unet_ffc, select_bottom_element, select_top_element
from draw_tools.tensor2img import *
import random as rd
from PIL import Image, ImageDraw, ImageFont
import os
import math
import warnings
warnings.filterwarnings('ignore')
LOG_NAME = 'logs/cifar10-trigger.log'
POTENTIAL_FEATURE_PATH='cache/cifar10/potential_trigger_feature/'
MINIMAL_PICS_PATH = 'cache/cifar10/minimal_picture/'
CIFAR_CLASSNUM = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class UnNormalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

    def __call__(self, tensor):
        """
        tensor: C×H×W
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # x = x * std + mean
        return tensor


def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce


def mask_dice(mask1, mask2):
    inter = ((mask1 == 1) & (mask2 == 1)).sum()
    total = (mask1 == 1).sum() + (mask2 == 1).sum()
    return (2 * inter.float()) / (total.float() + 1e-6)


def merge_pngs_to_pdf_with_overall_title(png_list, pdf_path, overall_title="Overall Report",
                                         titles=None, title_font_path=FONT_PATH,
                                         overall_title_font_size=80, title_font_size=48,
                                         line_thickness=3, line_margin=10):
    """
    将多张 PNG 图片合并为 PDF：
        1. 第一页为封面，显示整体大标题
        2. 后续每页显示图片，上方带每页标题和横线

    参数：
        png_list: list[str] 图片路径列表
        pdf_path: str 输出 PDF 路径
        overall_title: str 整体大标题文本
        titles: list[str] 每张图片的标题，可为 None（使用 "Page X"）
        title_font_path: str 字体路径，None 使用默认字体
        overall_title_font_size: int 封面标题字体大小
        title_font_size: int 每页标题字体大小
        line_thickness: int 横线厚度
        line_margin: int 横线距离标题的距离
    """
    images_for_pdf = []

    if titles is None:
        titles = [f"Page {i+1}" for i in range(len(png_list))]

    # 加载字体
    try:
        font_title = ImageFont.truetype(title_font_path, title_font_size) if title_font_path else ImageFont.load_default()
        font_overall = ImageFont.truetype(title_font_path, overall_title_font_size) if title_font_path else ImageFont.load_default()
    except:
        font_title = ImageFont.load_default()
        font_overall = ImageFont.load_default()

    # --- 1. 封面页 ---
    first_img = Image.open(png_list[0])
    width = first_img.width
    # 封面高度可以根据字体大小设置，保证标题可见
    height = overall_title_font_size * 3
    cover_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(cover_img)

    # 计算文字 bounding box 用于居中
    bbox = draw.textbbox((0,0), overall_title, font=font_overall)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), overall_title, fill=(0,0,0), font=font_overall)
    images_for_pdf.append(cover_img)

    # --- 2. 每页图片 ---
    for i, png_path in enumerate(png_list):
        img = Image.open(png_path).convert("RGB")
        w, h = img.size
        title_height = title_font_size + line_thickness + 2*line_margin
        new_img = Image.new("RGB", (w, h + title_height), (255, 255, 255))
        draw = ImageDraw.Draw(new_img)

        # 绘制每页标题
        draw.text((10, 0), titles[i], fill=(0,0,0), font=font_title)

        # 绘制横线
        y_line = title_font_size + line_margin
        draw.line([(0, y_line), (w, y_line)], fill=(0,0,0), width=line_thickness)

        # 粘贴原图片
        new_img.paste(img, (0, title_height))
        images_for_pdf.append(new_img)

    # 保存为 PDF
    images_for_pdf[0].save(pdf_path, save_all=True, append_images=images_for_pdf[1:])


def generate_squre_matrix(N):
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    center = (N-1) // 2
    value = 1
    radius = 0
    while radius <= center:
        for i in range(center - radius, center + radius + 1):
            matrix[center - radius][i] = value
            matrix[center + radius][i] = value
            matrix[i][center - radius] = value
            matrix[i][center + radius] = value
        radius += 1
        value += 1
    return torch.tensor(matrix).to(device).float()


def delete_analysis(scores:torch.Tensor, 
                    samples:torch.Tensor, 
                    net:nn.Module, 
                    step:float,
                    buttom=True):
    """
    生成置信度峰值对应的图片
    生成网络保持原始判度对应的图片
    生成置信度变化曲线
    生成不同频段的得分热力图
    """
    all_steps = int(1/step)
    maintain_rate = []
    original_confidence = net(samples)[0]
    original_pred_mask = original_confidence.round()
    print(original_pred_mask.sum().item())

    changed_flag = []
    mask_list = []
    flag = True
    predicted_mask_list = []
    for e in range(-1,all_steps-1,1):
        if buttom:
            masks = select_bottom_element(scores, (e+1)*step)
        else:
            masks = select_top_element(scores, (e+1)*step)
        #print(masks.sum().item())
        freq_sample = torch.fft.fft2(samples)
        masked_sample = freq_sample*masks
        filtered_sample = torch.fft.ifft2(masked_sample).real
        new_pred = net(filtered_sample)[0]
        pred_mask = new_pred.round()
        
        maintain_rate.append(mask_dice(pred_mask, original_pred_mask).item())
        mask_list.append(masks.clone())
        predicted_mask_list.append(pred_mask.cpu().detach().clone())

    return maintain_rate, mask_list, predicted_mask_list


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


def gen_report_p1(confidence_list:list, 
                  sample:torch.Tensor, 
                  mask_list:list[torch.Tensor],
                  scores:torch.Tensor, 
                  predicted_mask_list:list[torch.Tensor],
                  buttom=True):
    thred = 0.05
    if buttom:
        temp = find_sharp_change(confidence_list,threshold=-thred)
    else:
        temp = find_sharp_change(confidence_list,threshold=thred)
    if len(temp) == 0:
        print('No sharp drop found.')
        conf_sharp_change = -1
    else:
        conf_sharp_change = temp[0]+1  # 因为 diff 导致 index 偏移了 1

    max_conf_mask = mask_list[conf_sharp_change]
    
    original_img = sample[0]
    max_conf_img = torch.fft.ifft2(torch.fft.fft2(sample)*max_conf_mask).real[0]
    deleted_max_conf_img = torch.fft.ifft2(torch.fft.fft2(sample)*(1-max_conf_mask)).real[0]
    
    unormalize = UnNormalize()
    original_img = unormalize(original_img)
    deleted_max_conf_img = unormalize(deleted_max_conf_img)
    max_conf_img = unormalize(max_conf_img)
    between_change_mask = mask_list[conf_sharp_change+1]-mask_list[conf_sharp_change]
    between_change_img = torch.fft.ifft2(torch.fft.fft2(sample)*between_change_mask).real[0]
    between_change_img = unormalize(between_change_img)
    after_rapid_change_mask = mask_list[conf_sharp_change+1]
    after_rapid_change_img = torch.fft.ifft2(torch.fft.fft2(sample)*after_rapid_change_mask).real[0]
    after_rapid_change_img = unormalize(after_rapid_change_img)

    img_list = [original_img,
                original_img,
                deleted_max_conf_img,
                max_conf_img,
                max_conf_img,
                between_change_img,
                after_rapid_change_img]
    
    title_list = ['Original Image',
                  'Original Predicted Mask',
                  'Deleted Low-Confidence Features',
                  'Features Before Rapid Change',
                  'Modified Predicted Mask',
                  'Features Between Rapid Change',
                  'Features After Rapid Change']
    original_pred_idx = 0 if buttom else -1
    save_tensor_list(img_list, 
                     titles=title_list,
                     save_path='cache/Interpret_report/p1img/EffectShow.png',
                     mask_list=[torch.zeros_like(predicted_mask_list[0]),
                                predicted_mask_list[original_pred_idx],
                                torch.zeros_like(predicted_mask_list[0]),
                                torch.zeros_like(predicted_mask_list[0]),
                                predicted_mask_list[conf_sharp_change],
                                torch.zeros_like(predicted_mask_list[0]),
                                torch.zeros_like(predicted_mask_list[0]),])
    

    plt.figure(figsize=(8, 6))
    x = [i*(1/len(confidence_list)) for i in range(len(confidence_list))]
    y = confidence_list
    plt.plot(x, 
             confidence_list, marker='o', label='Confidence Ratio')
    highlight_idx = [conf_sharp_change]

    # 给每个点指定颜色
    highlight_colors = ['red', 'green', 'blue']  # 与 index 一一对应

    for idx, color in zip(highlight_idx, highlight_colors):
        plt.scatter(x[idx], y[idx], color=color, s=80, zorder=5)
    plt.xlabel('Proportion of Deleted Low-Confidence Features')
    plt.ylabel('Relative Confidence')
    plt.savefig('cache/Interpret_report/p1img/ConfidenceCurve.png', dpi=300, bbox_inches="tight")
    plt.close()
    scores4plot = scores[0].cpu().detach()
    plot_three_heatmaps(torch.log(torch.fft.fftshift(scores4plot[0],dim=(-2,-1))), 
                        torch.log(torch.fft.fftshift(scores4plot[1],dim=(-2,-1))), 
                        torch.log(torch.fft.fftshift(scores4plot[2],dim=(-2,-1))),
                        save_path='cache/Interpret_report/p1img/ScoreHeatmap.png')
    def merge_images_horizontally(image_paths, save_path):
        """
        将多张图片横向合并
        """
        images = [Image.open(p) for p in image_paths]

        # 统一高度为最高的那张
        max_height = max(img.height for img in images)
        total_width = sum(img.width for img in images)

        # 创建新画布
        new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # 粘贴图片
        x_offset = 0
        for img in images:
            # 如果高度不够，垂直居中
            y_offset = (max_height - img.height)//2
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width

            new_img.save(save_path)

    # 使用示例
    merge_images_horizontally(
        ['cache/Interpret_report/p1img/ConfidenceCurve.png',
        'cache/Interpret_report/p1img/ScoreHeatmap.png'],
        'cache/Interpret_report/p1img/Combined.png'
    )
    return conf_sharp_change


def gen_report_pn(scores:torch.Tensor, 
                  sample:torch.Tensor,
                  freq_range:list,
                  p=2):
    """
    freq_range: [low, high]
    """
    k_space = generate_squre_matrix(scores.shape[-1]).unsqueeze(0).repeat(scores.shape[1],1,1)
    scores_shift = torch.fft.fftshift(scores, dim=(-2,-1))
    step = 0.2
    range_mask = (k_space>freq_range[0]) & (k_space<=freq_range[1])
    range_original_mask = torch.fft.ifftshift(range_mask.int(), dim=(-2,-1))
    sample_freq = torch.fft.fft2(sample)
    range_original_img = torch.fft.ifft2(sample_freq*range_original_mask).real[0]
    unormalize = UnNormalize()
    range_original_img = unormalize(range_original_img)
    range_scores = torch.where(range_mask, scores_shift[0], -torch.inf)
    range_scores = range_scores[range_scores!=-torch.inf]
    range_scores = range_scores.view(-1)
    values, _ = torch.sort(range_scores, dim=-1, descending=True)
    step_all = int(1/step)
    range_mask_list = []
    last_thred = torch.inf
    for e in range(step_all):
        thred_value = values[math.ceil(values.shape[0]*(e+1)*step)-1]
        range_step_mask = torch.where((scores_shift>=thred_value)*(scores_shift<last_thred),1,0).view(range_mask.shape)
        range_step_mask = range_step_mask & range_mask
        last_thred = thred_value
        range_mask_list.append(range_step_mask)
    accumulate_img_list = []
    split_img_list = []
    accumulate_mask = range_mask_list[0]*0
    for e in range(len(range_mask_list)):
        freq_mask = torch.fft.ifftshift(range_mask_list[e].int(), dim=(-2,-1))
        split_img = torch.fft.ifft2(sample_freq*freq_mask).real[0]
        split_img_list.append(unormalize(split_img.clone()))
        if e == 0:
            accumulate_img = split_img.clone()
            accumulate_mask = freq_mask.clone()
        else:
            accumulate_mask = accumulate_mask | freq_mask
            accumulate_img = torch.fft.ifft2(sample_freq*accumulate_mask).real[0]
        accumulate_img_list.append(unormalize(accumulate_img.clone()))
    split_img_list.insert(0, range_original_img)
    accumulate_img_list.insert(0, range_original_img)
    save_tensor_list(split_img_list, 
                     save_path=f'cache/Interpret_report/p{p}img/SplitFreqImgs.png', 
                     )
    save_tensor_list(accumulate_img_list, 
                     save_path=f'cache/Interpret_report/p{p}img/AccumulateFreqImgs.png', 
                     )


def gen_report(net:nn.Module, sample:torch.Tensor):
    direction = True
    stride = 0.001
    scores = unet_ffc(net, sample, lr=1000, echo=20, loss_fn=combined_loss)
    confidence_list, mask_list, predicted_mask_list = \
        delete_analysis(scores, sample, net, step=stride, buttom = direction)
    sharp_change = gen_report_p1(confidence_list, sample.clone(), 
                  mask_list, scores.clone(), predicted_mask_list, buttom=direction)

    freq_upper = sample.shape[-1]//2
    freq_range_dict = {2:[0,5],
                       3:[6,25],
                       4:[25,freq_upper]}
    gen_report_pn(scores, sample, freq_range=freq_range_dict[2], p=2)
    gen_report_pn(scores, sample, freq_range=freq_range_dict[3], p=3)
    gen_report_pn(scores, sample, freq_range=freq_range_dict[4], p=4)
    
    png_list = ['cache/Interpret_report/p1img/EffectShow.png',
                'cache/Interpret_report/p1img/Combined.png']
    title_list = ['Effect Show', 'Statsistics Charts']
    range_dict = {2:'Usually Texture Features',
                  3:'Usually Shape Features',
                  4:'Usually Noise'}
    for i in range(2,5):
        for img_name in os.listdir(f'cache/Interpret_report/p{i}img/'):
            if img_name.endswith('.png'):
                png_list.append(f'cache/Interpret_report/p{i}img/{img_name}')
                title_list.append(f'Important Signals in Freq{freq_range_dict[i]}- {range_dict[i]} - {img_name[:-4]}')
    merge_pngs_to_pdf_with_overall_title(png_list, 
                                         overall_title='Interpretation Report',
                                 pdf_path='cache/Interpret_report/Interpretation_Report.pdf', 
                                 titles=title_list,
                                 title_font_size=108)


def gen_report_run():
    net = load_model()
    #torch.manual_seed(rd.randint(1,10000))
    imgs_path = 'dataset/train/'
    imgs = os.listdir(imgs_path)
    selected_img = rd.choice(imgs)
    image_path = imgs_path+selected_img
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
        
    print(f"Processing image: {image_path}")
    image_tensor, original_image = preprocess_image(image_path)
    gen_report(net, image_tensor.to(device))




if __name__ == "__main__":
    with torch.no_grad():
        gen_report_run()


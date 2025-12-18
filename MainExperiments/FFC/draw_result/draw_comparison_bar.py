import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import cm
from draw_result import result_value

############################ Form Conf ############################

highlight_color = {
    'gradcam':"#2E2E2E",  # 
    'fullgrad':"#1188aa",  # 
    'random':"#11aa88",  # 
    'inputgrad':"#8A8A8A",  # 
    'IG':"#A3A3A3",  # 
    'smoothgrad':"#a7cca7",  # 
    ############# Spatial baseline
    'energy':"#1155cc",  # 
    'sorted_freq':"#bcbd22",  # 
    'FASTShap':"#17becf",   # 
    'signal_projection':"#bb0030",  # 
}

short_name_dict = {
    'gradcam':"grCam",  # 
    'fullgrad':"fulGr",  # 
    'random':"rand",  # 
    'inputgrad':"in*Gr",  # 
    'IG':"IG",  # 
    'smoothgrad':"smoGr",  # 
    ############# Spatial baseline
    'energy':"en",  # 
    'sorted_freq':"sortF",  # 
    'FASTShap':"FaSHA",   # 
    'signal_projection':"SigPr",  # 
}

def draw_bar(data,
             y_label_name,
             x_label_name,
             x_labels,
             fig_name,
             title,
             form='pdf'):
    plt.title(title)
    my_colors = []
    for name in x_labels:
        my_colors.append(highlight_color[name])
    tick_labels = []
    for name in x_labels:
        tick_labels.append(short_name_dict[name])
    plt.bar(range(len(data)), data, tick_label=tick_labels,
            color=my_colors)

    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.savefig('ResultPictures/'+fig_name+'.'+form, format=form)
    plt.close('all')

####################### ResNet50 ##############################
def resnet_spatial_auc():
    false_score = result_value.get_resnet_spatial_false_score()
    true_score = result_value.get_resnet_spatial_true_score()
    method_name = list(false_score.keys())
    bar_value = []
    for m in method_name:
        bar_value.append(sum(false_score[m])-sum(true_score[m]))
    draw_bar(bar_value,'Least Sum - Most Sum',
             x_label_name='Methods',
             x_labels=method_name,
             title='ResNet50 Spatial',
             fig_name='Spatial-ResNet50-Auc')
        

def resnet_fft_auc():
    false_score = result_value.get_resnet_fft_false_score()
    true_score = result_value.get_resnet_fft_true_score()
    method_name = list(false_score.keys())
    bar_value = []
    for m in method_name:
        bar_value.append(sum(false_score[m])-sum(true_score[m]))
    draw_bar(bar_value,'Least Sum - Most Sum',
             x_label_name='Methods',
             x_labels=method_name,
             title='ResNet50 FFT',
             fig_name='FFT-ResNet50-Auc')


def resnet_ifft_auc():
    false_score = result_value.get_resnet_ifft_false_score()
    true_score = result_value.get_resnet_ifft_true_score()
    method_name = list(false_score.keys())
    bar_value = []
    for m in method_name:
        bar_value.append(sum(false_score[m])-sum(true_score[m]))
    draw_bar(bar_value,'Least Sum - Most Sum',
             x_label_name='Methods',
             x_labels=method_name,
             title='ResNet50 IFFT',
             fig_name='IFFT-ResNet50-Auc')

########################### ViT #############################
def vit_spatial_auc():
    false_score = result_value.get_vit_spatial_false_score()
    true_score = result_value.get_vit_spatial_true_score()
    method_name = list(false_score.keys())
    bar_value = []
    for m in method_name:
        bar_value.append(sum(false_score[m])-sum(true_score[m]))
    draw_bar(bar_value,'Least Sum - Most Sum',
             x_label_name='Methods',
             x_labels=method_name,
             title='ViT Spatial',
             fig_name='Spatial-ViT-Auc')
        

def vit_fft_auc():
    false_score = result_value.get_vit_fft_false_score()
    true_score = result_value.get_vit_fft_true_score()
    method_name = list(false_score.keys())
    bar_value = []
    for m in method_name:
        bar_value.append(sum(false_score[m])-sum(true_score[m]))
    draw_bar(bar_value,'Least Sum - Most Sum',
             x_label_name='Methods',
             x_labels=method_name,
             title='ViT FFT',
             fig_name='FFT-ViT-Auc')


def vit_ifft_auc():
    false_score = result_value.get_vit_ifft_false_score()
    true_score = result_value.get_vit_ifft_true_score()
    method_name = list(false_score.keys())
    bar_value = []
    for m in method_name:
        bar_value.append(sum(false_score[m])-sum(true_score[m]))
    draw_bar(bar_value,'Least Sum - Most Sum',
             x_label_name='Methods',
             x_labels=method_name,
             title='ViT IFFT',
             fig_name='IFFT-ViT-Auc')
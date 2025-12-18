import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import cm

######## form config ########
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


def draw_group_bar(categories,
                   values1,
                   values2,
                   fig_name,
                   title,
                   bar_width = 0.35,
                   form = 'pdf'):
    tick_labels = []
    x = np.arange(len(categories))
    
    plt.bar(x - bar_width/2, values1, width=bar_width, label='Group 1')
    plt.bar(x + bar_width/2, values2, width=bar_width, label='Group 2')
    plt.xlabel('Methods')
    plt.ylabel('Values')
    plt.title(title)
    for name in categories:
        tick_labels.append(short_name_dict[name])
    plt.xticks(x, tick_labels)
    plt.legend(['Kurtosis','Specific'])
    plt.savefig('ResultPictures/'+fig_name+'.'+form, format=form)
    plt.close('all')


def resnet_bar():
    kurtorsis_dict = {'fullgrad':1.038348312671478,
                      'IG': 0.9999985690708753,
                      'smoothgrad': 1.0949449156139706,
                      'random': 0.9999965931679513,
                      'inputgrad': 1.0513160438747615,
                      'gradcam': 1.1196671205836612,
                      'signal_projection': 1.7949303505137053,}
    specific_dict = {'fullgrad':3772,
                      'IG': 0,
                      'smoothgrad': 3754,
                      'random': 0,
                      'inputgrad': 1549,
                      'gradcam': 2806,
                      'signal_projection': 3138,}
    

    categories = kurtorsis_dict.keys()
    k_values = []
    s_values = []
    k_max = 1.7949303505137053
    s_max = 3772
    for name in categories:
        k_values.append(kurtorsis_dict[name]/k_max)
        s_values.append(specific_dict[name]/s_max)
    draw_group_bar(categories, k_values, s_values, title='ResNet50', fig_name='ResnetDistribution')


def vit_bar():
    kurtorsis_dict = {
                      'IG': 1.0004014147891178,
                      'smoothgrad': 1.0969907772075664,
                      'random': 0.9999963653099548,
                      'inputgrad': 1.1725943031253758,
                      'gradcam': 1.0322404397023213,
                      'signal_projection': 1.7657933904930896,}
    specific_dict = {
                      'IG': 0,
                      'smoothgrad': 3668,
                      'random': 0,
                      'inputgrad': 3352,
                      'gradcam': 0,
                      'signal_projection': 3274,}
    

    categories = kurtorsis_dict.keys()
    k_values = []
    s_values = []
    k_max = 1.7657933904930896
    s_max = 3668
    for name in categories:
        k_values.append(kurtorsis_dict[name]/k_max)
        s_values.append(specific_dict[name]/s_max)
    draw_group_bar(categories, k_values, s_values, title='ViT_B/32', fig_name='ViTDistribution')

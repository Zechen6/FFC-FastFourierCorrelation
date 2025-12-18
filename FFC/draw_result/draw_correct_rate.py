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
    # 绘制柱状图
    plt.bar(x - bar_width/2, values1, width=bar_width, label='Group 1')
    plt.bar(x + bar_width/2, values2, width=bar_width, label='Group 2')
    plt.xlabel('Methods')
    plt.ylabel('Correct Rate')
    plt.title(title)
    for name in categories:
        tick_labels.append(short_name_dict[name])
    plt.xticks(x, tick_labels)
    plt.legend(['ResNet50','ViT_B/32'])
    plt.savefig('ResultPictures/'+fig_name+'.'+form, format=form)
    plt.close('all')

def correct_rate_bar():
    vit_rate = {
                'random': 2.75,
                'sorted_freq': 21.11,
                'energy': 25.88,
                'signal_projection': 26.24
                }
    resnet_rate = {
                    'random': 3.44,
                    'sorted_freq': 22.01,
                    'energy': 28.56,
                    'signal_projection': 28.46
                    }
    

    categories = vit_rate.keys()
    k_values = []
    s_values = []
    for name in categories:
        k_values.append(vit_rate[name])
        s_values.append(resnet_rate[name])
    draw_group_bar(categories, k_values, s_values, title='Correct Rate', fig_name='CorrectRate')


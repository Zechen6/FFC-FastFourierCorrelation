import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import cm
from draw_result import result_value

###################### Form Conf ########################

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


marker_shape_dict = {
    'gradcam':"^",  # 
    'fullgrad':"*",  # 
    'random':"o",  # 
    'inputgrad':"o",  #
    'IG':"x",  #
    'smoothgrad':"o",  # 
    ############# Spatial baseline
    'energy':"s",  # 
    'sorted_freq':"s",  # 
    'FASTShap':"s",   # 
    'signal_projection':"v",  # 
}

marker_size_dict = {
    'gradcam':"4",  # 
    'fullgrad':"4",  # 
    'random':"4",  # 
    'inputgrad':"4",  # 
    'IG':"4",  #
    'smoothgrad':"4",  # 
    ############# Spatial baseline
    'energy':"4",  # 
    'sorted_freq':"8",  # 
    'FASTShap':"8",   #
    'signal_projection':"8",  # 
}

############## Spatial ######################

def plot_spatial_res_result_False():
    
    res_dict = result_value.get_resnet_spatial_false_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Spatial-ResNet50-Del-False.pdf',format='pdf')
    plt.close('all')


def plot_spatial_vit_result_False():
    
    res_dict = result_value.get_vit_spatial_false_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('ViT_B/32')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Spatial-ViT-Del-False.pdf',format='pdf')
    plt.close('all')


def plot_spatial_vit_result_True():
    
    res_dict = result_value.get_vit_spatial_true_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('ViT_B/32')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Spatial-ViT-Del-True.pdf',format='pdf')
    plt.close('all')


def plot_spatial_res_result_True():
    
    res_dict = result_value.get_resnet_spatial_true_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Spatial-ResNet50-Del-True.pdf',format='pdf')
    plt.close('all')

###################### Ablation ResNet ###################################

def plot_fft_abla_res_false():
    
    res_dict = result_value.get_resnet_fft_false_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('FFT ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-FFT-ResNet50-Del-False.pdf',format='pdf')
    plt.close('all')


def plot_fft_abla_res_true():
    
    res_dict = result_value.get_resnet_fft_true_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('FFT ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    plt.yscale('log')
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-FFT-ResNet50-Del-True.pdf',format='pdf')
    plt.close('all')


def plot_ifft_abla_res_false():
    
    res_dict = result_value.get_resnet_ifft_false_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('IFFT ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-IFFT-ResNet50-Del-False.pdf',format='pdf')
    plt.close('all')
    

def plot_ifft_abla_res_true():
    
    res_dict = result_value.get_resnet_ifft_true_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('IFFT ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    plt.yscale('log')
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-IFFT-ResNet50-Del-True.pdf',format='pdf')
    plt.close('all')

############## Ablation ViT #####################

def plot_fft_abla_vit_false():
    res_dict = result_value.get_vit_fft_false_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('FFT ViT_B/32')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-FFT-ViT-Del-False.pdf',format='pdf')
    plt.close('all')


def plot_fft_abla_vit_true():
    res_dict = result_value.get_vit_fft_true_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('FFT ViT_B/32')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    plt.yscale('log')
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-FFT-ViT-Del-True.pdf',format='pdf')
    plt.close('all')


def plot_ifft_abla_vit_false():
    res_dict = result_value.get_vit_ifft_false_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('IFFT ViT_B/32')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-IFFT-ViT-Del-False.pdf',format='pdf')
    plt.close('all')
    

def plot_ifft_abla_vit_true():
    res_dict = result_value.get_vit_ifft_true_score()
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('IFFT ViT_B/32')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    plt.yscale('log')
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name],
                 marker=marker_shape_dict[name],markersize=marker_size_dict[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/Abalation-IFFT-ViT-Del-True.pdf',format='pdf')
    plt.close('all')


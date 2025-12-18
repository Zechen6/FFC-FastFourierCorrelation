import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import cm
from draw_result import draw_comparison_line, draw_comparison_bar, draw_parameter_analysis, draw_distribution, draw_distinguish_bar
from draw_result import draw_correct_rate
# visualize local explanation results
def visualize_del_ins_game(imp,file_name):
    file = file_name
    model_name = 'ViT'
    methods_bs_dict = {'Freq':128,'random':16,'inputgrad':100,'gradcam':100,'IG':4,'smoothgrad':100,'energy':10,'fullgrad':8,'sort_freq':128}
    methods = list(methods_bs_dict.keys())
    
    if model_name == 'ViT':
        methods.remove('fullgrad')
    with open(file, 'r', encoding='utf-8') as f:
        res_dict = json.load(f)
    #if 'Space' in file_name:
    #    with open('Evaluation-OurSpace-ResultViT.json','r',encoding='utf-8') as f:
    #        ours = json.load(f)
    #    res_dict['Freq'][model_name]['True'] = ours['Freq'][model_name]['False']
    #    res_dict['Freq'][model_name]['False'] = ours['Freq'][model_name]['True']
    #print(res_dict)
    res_list = []
    for m in methods:
        if m == 'fullgrad' and model_name == 'ViT':
            continue
        if m == 'energy' and 'Space' in file_name:
            continue
        #print(m)
        temp = res_dict[m][model_name][str(imp)]
        temp_list = []
        for i in range(len(temp)):
            if i == 0:
                temp_list.append(temp[i])
            if i%2==1:
                temp_list.append(temp[i])
        res_list.append(temp_list)
    
    plt.figure(figsize=(8,6))
    plt.title('ViT_B_32')
    plt.grid(True)
    for item in res_list:
        x_val = np.arange(len(item))/10
        plt.xticks(x_val)
        plt.plot(x_val, item, marker='v')
    methods[0] = 'ER'
    plt.legend(methods)
    if imp:
        plt.savefig('results/ViTDel-Imp(IFFT)-sorted_freq.pdf',format='pdf')
    else:
        plt.savefig('results/ViTDel-NoImp(IFFT)-sorted_freq.pdf',format='pdf')
    plt.close('all')


def visualize_max_deletion_rate():
    x = [0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
    vit = [0.95322, 0.94434, 0.9334, 0.9206, 0.90172, 0.8758, 0.83792, 0.7801, 0.68588, 0.5132]
    res = [0.83414, 0.82064, 0.80494, 0.7873, 0.76614, 0.7355, 0.69856, 0.64172, 0.5503, 0.3761]
    fig, ax = plt.subplots()
    plt.grid(True)
    
    ax.plot(x, vit, marker='o')

    
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    
    ax.set_title('ViT_B_32')
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Acc')

    
    plt.savefig('results/max_deletion_rate_vit.pdf',format='pdf')


def get_auc(file_name):
    file = file_name
    model_name = 'ViT'
    methods_bs_dict = {'Freq':128,'random':16,'inputgrad':100,'gradcam':100,'IG':4,'smoothgrad':100,'energy':10,'fullgrad':8,'sort_freq':128}
    methods = list(methods_bs_dict.keys())
    
    if model_name == 'ViT':
        methods.remove('fullgrad')
    with open(file, 'r', encoding='utf-8') as f:
        res_dict = json.load(f)
    #if 'Space' in file_name:
    #    with open('Evaluation-OurSpace-ResultViT.json','r',encoding='utf-8') as f:
    #        ours = json.load(f)
    #    res_dict['Freq'][model_name]['True'] = ours['Freq'][model_name]['False']
    #    res_dict['Freq'][model_name]['False'] = ours['Freq'][model_name]['True']
    #print(res_dict)
    res_list = []
    for m in methods:
        if m == 'fullgrad' and model_name == 'ViT':
            continue
        if m == 'energy' and 'Space' in file_name:
            continue
        #print(m)
        tempA = res_dict[m][model_name][str(True)]
        temp_listA = []
        for i in range(len(tempA)):
            if i == 0:
                temp_listA.append(tempA[i])
            if i%2==1:
                temp_listA.append(tempA[i])

        tempB = res_dict[m][model_name][str(False)]
        temp_listB = []
        for i in range(len(tempB)):
            if i == 0:
                temp_listB.append(tempB[i])
            if i%2==1:
                temp_listB.append(tempB[i])

        res_list.append(sum(tempB)-sum(tempA))
    if model_name == 'ResNet50':
        x_labels = ['ER','rand','in*Gr','grCam','IG','smoGr','Ener','fulGr','SortFre']
    else:
        x_labels = ['ER','rand','in*Gr','grCam','IG','smoGr','Ener','SortFre']
    draw_bar(res_list, y_label_name='Least Sum - Most Sum',
             x_label_name='Methods', x_labels=x_labels,
             fig_name=model_name+'IFFT',title='Least Sum - Most Sum')

def draw_bar(data,
             y_label_name,
             x_label_name,
             x_labels,
             fig_name,
             title,
             form='pdf'):
    plt.title(title)
    my_colors = cm.BuGn(0.2+1-(np.arange(len(data)) / len(data)))
    plt.bar(range(len(data)), data, tick_label=[x_labels[i] for i in range(len(data))],
            color=my_colors)

    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.savefig('results/'+fig_name+'.'+form, format=form)
    plt.close('all')


colors_dict = {
    'gradcam':"#1f77b4",  # 
    'fullgrad':"#ff7f0e",  # 
    'random':"#2ca02c",  # 
    'inputgrad':"#d62728",  # 
    'IG':"#9467bd",  # 
    'smoothgrad':"#8c564b",  # 
    'signal_projection':"#e377c2",  # 
    'energy':"#7f7f7f",  # 
    'sorted_freq':"#bcbd22",  # 
    'FASTShap':"#17becf"   #
}

highlight_color = {
    'gradcam':"#606060",  # 
    'fullgrad':"#dcdcdc",  # 
    'random':"#e6e6e6",  # 
    'inputgrad':"#7f7f7f",  # 
    'IG':"#808080",  # 
    'smoothgrad':"#404040",  # 
    'signal_projection':"#ff4500",  # 
    'energy':"#bcbd22",  # 
    'sorted_freq':"#bcbd22",  # 
    'FASTShap':"#17becf"   # 
}

def plot_spatial_res_result_False():
    gradcam = [1.0,0.8321104645729065,0.7296780347824097,0.634721577167511,0.5462977886199951,
                0.4646143317222595,0.3858781158924103,0.3062486946582794,0.2121751457452774,0.10898779332637787]
    fullgrad = [1.0,0.7852626442909241,0.681408166885376,0.5878832936286926,0.5017032027244568,
                0.40902775526046753,0.32244113087654114,0.23916950821876526,0.1602829396724701,0.09838025271892548]
    random = [1.0,0.49388405680656433,0.3581681251525879,0.26630085706710815,0.19742561876773834,
                0.13424052298069,0.08917318284511566,0.05784692242741585,0.04028722271323204,0.03340838477015495]
    inputgrad = [1.0,0.7313532829284668,0.5882681012153625,0.470459520816803,0.3594665825366974,0.2644740641117096,
                0.17396800220012665,0.10421502590179443,0.06033238768577576,0.03508881852030754]
    IG = [1.0,0.37550896406173706,0.22887492179870605,0.15793977677822113,0.12313136458396912,
            0.10825145989656448,0.09775345772504807,0.08397963643074036,0.06847492605447769,0.05583648383617401]
    smoothgrad = [1.0,0.6958408951759338,0.5746040940284729,0.48711439967155457,0.4193032681941986,
                0.3586384356021881,0.29704251885414124,0.2225629687309265,0.14570404589176178,0.06971432268619537]
    signal_projection = [1.0, 2.0692524909973145, 2.1559083461761475, 2.1503281593322754, 2.0732903480529785, 
                         1.9143292903900146, 1.6733224391937256, 1.3830153942108154, 1.0816482305526733, 0.7673652768135071]
    energy = [1.0, 0.9996797442436218, 0.9967201948165894, 0.9914652705192566, 0.9817413687705994, 
              0.9628854393959045, 0.9295899868011475, 0.8799527287483215, 0.8111939430236816, 0.684890627861023]
    res_dict = {'gradcam':gradcam,'fullgrad':fullgrad,'random':random,'inputgrad':inputgrad,'IG':IG,
                'smoothgrad':smoothgrad,'signal_projection':signal_projection}
    x_axes = list(range(10))
    method_name = list(res_dict.keys())
    legend = method_name
    fig = plt.figure(figsize=(10,10))
    plt.title('ResNet50')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Relative Confidence")
    for name in method_name:
        plt.plot(x_axes,res_dict[name],color=highlight_color[name])
    plt.legend(legend)
    plt.savefig('ResultPictures/ResNet50-Del-False.pdf',format='pdf')
    plt.close('all')


#plot_spatial_res_result_False()
draw_comparison_line.plot_spatial_res_result_False()
draw_comparison_line.plot_spatial_res_result_True()
draw_comparison_line.plot_spatial_vit_result_False()
draw_comparison_line.plot_spatial_vit_result_True()
draw_comparison_bar.resnet_spatial_auc()
draw_comparison_bar.vit_spatial_auc()
draw_comparison_line.plot_fft_abla_res_false()
draw_comparison_line.plot_fft_abla_res_true()
draw_comparison_line.plot_ifft_abla_res_false()
draw_comparison_line.plot_ifft_abla_res_true()
draw_comparison_line.plot_fft_abla_vit_false()
draw_comparison_line.plot_fft_abla_vit_true()
draw_comparison_line.plot_ifft_abla_vit_false()
draw_comparison_line.plot_ifft_abla_vit_true()
draw_comparison_bar.resnet_fft_auc()
draw_comparison_bar.resnet_ifft_auc()
draw_comparison_bar.vit_fft_auc()
draw_comparison_bar.vit_ifft_auc()
"""draw_parameter_analysis.get_res_bubble_file()
draw_parameter_analysis.get_vit_bubble_file()"""
#draw_distribution.draw_acc()
#draw_distinguish_bar.vit_bar()
#draw_correct_rate.correct_rate_bar()

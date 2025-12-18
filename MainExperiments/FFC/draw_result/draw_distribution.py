from draw_result import result_value
import matplotlib.pyplot as plt

def draw_acc():
    accs = result_value.get_acc()
    fig = plt.figure(figsize=(9,6))
    plt.grid(True, which='major', axis='y', linestyle='--', color='#dddddd', alpha=0.6)
    plt.title('Maintain Prediction')
    plt.xlabel("Deletion Rate")
    plt.ylabel("Maintain Rate")
    method_name = list(accs.keys())
    x_axes = list(range(len(accs['vit'])))
    marker_shape_dict = {'res':'s',
                         'vit':'v'}
    for name in method_name:
        temp = accs[name]
        plt.plot(x_axes, temp,
                 marker=marker_shape_dict[name],markersize=8)
    plt.legend(method_name)
    plt.savefig('ResultPictures/Distribution-Acc.pdf',format='pdf')
    plt.close('all')



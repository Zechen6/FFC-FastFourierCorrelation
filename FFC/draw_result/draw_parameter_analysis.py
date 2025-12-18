from draw_result import result_value
import csv

def get_res_bubble_file():
    rows = result_value.get_resnet_lr_echo_value()
    with open('draw_result/resnet_param_anly.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def get_vit_bubble_file():
    rows = result_value.get_vit_lr_echo_value()
    with open('draw_result/vit_param_anly.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
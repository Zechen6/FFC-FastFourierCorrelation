import matplotlib.pyplot as plt
import numpy as np

def plot_conf(
    conf_curve: list[float],
    save_path: str,
    highlight_map: dict[int, tuple[str, str]] = None, # {index: (color, label)},
    step = 0.1
):
    """
    绘制相对置信度曲线，并可高亮指定索引的点。

    Args:
        conf_curve (list[float]): 相对置信度值列表。
        save_path (str): 图片保存路径。
        highlight_map (dict[int, tuple[str, str]], optional):
            高亮点的映射：{索引: (颜色字符串, 释义/标签)}。
            例如：{3: ('red', 'Key Insight 1'), 5: ('green', 'Key Insight 2')}
            默认 None。
    """
    x = np.array(range(len(conf_curve))) * (1/step)
    
    plt.figure(figsize=(8, 5))

    # 绘制置信度曲线 (Line Plot)
    plt.plot(
        x, conf_curve, marker='o', linewidth=2,
        label="Relative Confidence Curve", zorder=1 # zorder确保曲线在下层
    )

    # --- 增加高亮点的功能 ---
    highlight_handles = [] # 用于收集高亮点的图例句柄
    
    if highlight_map:
        # 提取唯一的颜色和标签组合，以构建左下角的图例
        # { (color, label): [indices] }
        color_label_map = {}
        for index, (color, label) in highlight_map.items():
            key = (color, label)
            if key not in color_label_map:
                color_label_map[key] = []
            color_label_map[key].append(index)
        
        # 遍历唯一的颜色/标签组合
        for (color, label), indices in color_label_map.items():
            
            # 1. 绘制高亮点
            highlight_x = x[indices]
            highlight_y = np.array(conf_curve)[indices]
            
            # 使用 scatter 绘制点， zorder=2 确保它在折线上层
            plt.scatter(
                highlight_x, highlight_y,
                color=color, marker='o', s=60, linewidth=1.5,
                edgecolor='black', zorder=2
            )
            
            # 2. 收集用于自定义图例的句柄
            # 创建一个仅用于图例的代理散点图句柄
            handle = plt.scatter([], [], color=color, marker='o', s=60, 
                                 edgecolor='black', label=label)
            highlight_handles.append(handle)

    # 设置轴和标题
    plt.xlabel("Deletion Ratio Step (%)", fontsize=12)
    plt.ylabel("Relative Confidence", fontsize=12)
    plt.title("Deletion-Insertion Game", fontsize=14)

    plt.grid(alpha=0.3)
    plt.xticks(x)
    
    # 原始曲线的图例 (默认位置，通常是右上)
    plt.legend(fontsize=12, loc='upper right')

    # 高亮点释义的图例 (位置在左下角)
    if highlight_handles:
        # 提取高亮标签
        highlight_labels = [h.get_label() for h in highlight_handles]
        
        # 添加高亮点的图例，使用 'lower left' 定位
        plt.legend(
            highlight_handles, highlight_labels, 
            loc='lower left', 
            title="Highlights", # 可以给高亮图例一个标题
            fontsize=10, 
            framealpha=0.8, # 增加背景透明度
            borderaxespad=0.5 # 离坐标轴的距离
        )


    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close() # 建议在函数末尾关闭 figure


def plot():
    conf_curve = [1,1.0056421329846252, 1.0075002820105166, 1.011614441066175, 
                  1.0129144626694757, 1.0151620209217072, 1.0125903382494643, 
                  1.0127655839597858, 1.0088121190264419, 0.9962553897419492]
    maintain_curve = [1,0.9617834394904459, 0.96104033970276, 0.9616772823779193, 
                      0.9628450106157113, 0.9628450106157113, 0.9592356687898089, 
                      0.9570063694267515, 0.9504246284501061, 0.9329087048832272]
    x = np.array(range(len(conf_curve)))*10

    plt.figure(figsize=(8, 5))

    # confidence plot
    plt.plot(
        x, conf_curve, marker='o', linewidth=2,
        label="Relative Confidence Curve"
    )

    # maintain rate plot
    plt.plot(
        x, maintain_curve, marker='s', linewidth=2,
        label="Maintain Rate Curve"
    )

    plt.xlabel("Deletion Ratio Step (%)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Deletion-Insertion Confidence & Maintain Rate Curves", fontsize=14)

    plt.grid(alpha=0.3)
    plt.xticks(x)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('ResnetAudio.pdf',format='pdf')


def plot_geneformer():
    conf_curve = [
        1,
    542.4375556707382/538,
    554.6898599863052/538,
    554.3702813386917/538,
    553.1244654655457/538,
    549.8403385281563/538,
    545.5664010047913/538,
    541.707091152668/538,
    539.1654016375542/538,
    538.9865345954895/538,
    ]
    maintain_curve = [
    17228/17228,
    17228/17228,
    17226/17228,
    17204/17228,
    17080/17228,
    16735/17228,
    16196/17228,
    15476/17228,
    14719/17228,
    14039/17228
    ]
    x = np.array(range(len(conf_curve)))*10

    plt.figure(figsize=(8, 5))

    # confidence plot
    plt.plot(
        x, conf_curve, marker='o', linewidth=2,
        label="Relative Confidence Curve"
    )

    # maintain rate plot
    plt.plot(
        x, maintain_curve, marker='s', linewidth=2,
        label="Maintain Rate Curve"
    )

    plt.xlabel("Deletion Ratio Step (%)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Deletion-Insertion Confidence & Maintain Rate Curves", fontsize=14)

    plt.grid(alpha=0.3)
    plt.xticks(x)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('Geneformer.pdf',format='pdf')


if __name__ == "__main__":
    plot_geneformer()
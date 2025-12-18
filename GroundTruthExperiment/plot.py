import json
import matplotlib.pyplot as plt
import json

def plot_metrics_from_json(json_dict):
    # --- 1. 将 key 转为排序后的 x 轴 ---
    x = sorted([float(k) for k in json_dict.keys()])
    
    # --- 2. 按 x 顺序提取数据 ---
    net_acc = [json_dict[str(k)]["net_acc"] for k in x]
    interpret = [json_dict[str(k)]["interpret"] for k in x]
    include_rate = [json_dict[str(k)]["include_rate"] for k in x]
    redundent_rate = [json_dict[str(k)]["redundent_rate"] for k in x]

    # --- 3. 绘制曲线 + 每个点 scatter ---
    plt.figure(figsize=(10, 6))

    # net_acc
    plt.plot(x, net_acc, label="net_acc")
    plt.scatter(x, net_acc)

    # interpret
    plt.plot(x, interpret, label="interpret_acc")
    plt.scatter(x, interpret)

    # include_rate
    plt.plot(x, include_rate, label="include_rate")
    plt.scatter(x, include_rate)

    # redundent_rate
    plt.plot(x, redundent_rate, label="redundent_rate")
    plt.scatter(x, redundent_rate)

    plt.xlabel("Variance")
    plt.ylabel("Metric Value")
    plt.title("Variance & Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('gt-res.pdf', format='pdf')
    #plt.show()


with open("res.json", "r") as f:
    data = json.load(f)

plot_metrics_from_json(data)

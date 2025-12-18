import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def tensor2img(tensor: torch.Tensor,
               save_path: str | Path,
               title: str | None = None,
               scale: float = 4.0,
               dpi: int = 300) -> None:
    save_path = Path(save_path)
    if tensor.dim() == 4:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            raise ValueError(f"只能画一张图")
    if tensor.ndim != 3:
        raise ValueError(f"期望 3 维 tensor，got {tensor.ndim} 维")
    c, h, w = tensor.shape
    if c not in {1, 3}:
        raise ValueError(f"通道数 C 必须是 1 或 3，got {c}")

    img = tensor.detach().cpu().float()
    img = img - img.min()
    img = img / img.max().clamp(min=1e-8)
    img = img.permute(1, 2, 0)
    if c == 1:
        img = img.squeeze(-1)

    # 1. 先算好英寸尺寸
    fig_w, fig_h = w / dpi * scale, h / dpi * scale

    plt.ioff()
    # 2. 一次性创建大画布 + axes
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 让 axes 占满整个画布
    ax.imshow(img, cmap='gray' if c == 1 else None)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=5, pad=10)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt

def plot_three_heatmaps(mat1, mat2, mat3, titles=None, save_path=None):
    """
    输入三个矩阵，绘制三张并排热力图

    参数：
        mat1, mat2, mat3: 2D numpy 数组
        titles: 可选，长度为3的标题列表，例如 ['A', 'B', 'C']
    """
    matrices = [mat1, mat2, mat3]
    if titles is None:
        titles = ['R channel Scores', 'G channel Scores', 'B channel Scores']

    plt.figure(figsize=(15, 4))
    
    for i, mat in enumerate(matrices):
        plt.subplot(1, 3, i+1)
        plt.imshow(mat, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(titles[i])
        plt.xlabel("X")
        plt.ylabel("Y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_three_tensors(img1, img2, img3, titles=None, save_path="output.png"):
    """
    将三个 PyTorch Tensor 图像并排画在一起，并保存到文件（不显示）。

    参数：
        img1, img2, img3: PyTorch Tensor，形状可为 (C,H,W) 或 (H,W)
        titles: 图像标题列表，长度为3，例如 ["A", "B", "C"]
        save_path: 保存路径，例如 "results/three_imgs.png"
    """

    tensors = [img1, img2, img3]
    if titles is None:
        titles = ["Original", "Confidence Sharp Drop", "Maintain Prediction"]

    plt.figure(figsize=(15, 5))

    for i, img in enumerate(tensors):
        plt.subplot(1, 3, i+1)

        # 转 numpy
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

        # 处理 batch 维度 B×C×H×W
        if img.ndim == 4:
            img = img[0]

        # 处理 C×H×W
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)  # -> H×W×C
            else:
                raise ValueError("3D tensor should be (C,H,W)")

        img_np = img.numpy()

        # 灰度图
        if img_np.ndim == 2 or img_np.shape[-1] == 1:
            plt.imshow(img_np.squeeze(), cmap="gray")
        else:
            plt.imshow(img_np)

        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_tensor_list(images, titles=None, save_path="output.png"):
    """
    将任意数量的 PyTorch Tensor 图像并排绘制并保存到文件。

    参数：
        images: list[Tensor]，每个 Tensor 为 (C,H,W) 或 (H,W)
        titles: list[str]，每个图像的标题，可为 None
        save_path: 保存路径，如 "results/output.png"
    """

    n = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]


    # 设置画布
    plt.figure(figsize=(5 * n, 5))

    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)

        # 转成 CPU + numpy
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

        # 如果有 batch 维度 B×C×H×W → C×H×W
        if img.ndim == 4:
            img = img[0]

        # C×H×W → H×W×C
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            else:
                raise ValueError("Tensor must be CHW or HW.")

        img_np = img.numpy()

        # 灰度图
        if img_np.ndim == 2 or img_np.shape[-1] == 1:
            plt.imshow(img_np.squeeze(), cmap="gray")
        else:
            plt.imshow(img_np)

        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


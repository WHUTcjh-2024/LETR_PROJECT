import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional


def plot_gray_profile(
        img_pil: Image.Image,
        calib_center: Optional[np.ndarray] = None,
        exp_blur_center: Optional[np.ndarray] = None,  # 新增：模糊中央点入参声明
        exp_first_order: Optional[np.ndarray] = None,
        scale_start: Optional[np.ndarray] = None,
        scale_end: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    绘制实验图的灰度图和条纹亮度分布曲线
    支持在图上标注预测的关键点
    :param img_pil: 原始PIL图像
    :param calib_center: 中央条纹坐标 (x0, y0)，原始图像像素值
    :param exp_blur_center: exp图模糊中央条纹坐标 (x0, y0)，原始图像像素值（新增）
    :param exp_first_order: 一级条纹坐标 (x1, y1)，原始图像像素值
    :param scale_start: 200mm刻度坐标，原始图像像素值
    :param scale_end: 240mm刻度坐标，原始图像像素值
    :return: matplotlib Figure对象
    """
    # 转换图像格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_h, original_w = gray.shape

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制灰度图
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('实验图灰度图', fontsize=14)
    ax1.axis('off')

    # 标注关键点（颜色+标签适配新刻度）
    colors = ['r', 'm', 'g', 'b', 'y']  # 新增紫色给模糊中央点
    labels = ['calib清晰中央点', 'exp模糊中央点', '一级条纹', '200mm刻度', '240mm刻度']
    points = [calib_center, exp_blur_center, exp_first_order, scale_start, scale_end]

    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        if point is not None:
            x, y = point
            x = np.clip(x, 0, original_w - 1)
            y = np.clip(y, 0, original_h - 1)
            ax1.scatter(x, y, c=color, s=100, marker='+', linewidth=2, label=label)
            ax1.text(x + 10, y, label, c=color, fontsize=12, fontweight='bold')

    if any(p is not None for p in points):
        ax1.legend(loc='upper right', fontsize=10)

    # 绘制亮度分布曲线
    brightness_profile = np.mean(gray, axis=0)
    x_axis = np.arange(original_w)

    ax2.plot(x_axis, brightness_profile, color='b', linewidth=1.5, label='亮度分布')
    ax2.set_title('条纹亮度分布曲线', fontsize=14)
    ax2.set_xlabel('水平像素位置', fontsize=12)
    ax2.set_ylabel('平均亮度（0-255）', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 标注关键点的x位置
    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        if point is not None:
            x, _ = point
            x = np.clip(x, 0, original_w - 1)
            ax2.axvline(x=x, color=color, linestyle='--', linewidth=1.5, label=f'{label}位置')

    if any(p is not None for p in points):
        ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig
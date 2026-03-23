import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional


def plot_gray_profile(
        img_pil: Image.Image,
        calib_center: Optional[np.ndarray] = None,
        exp_first_order: Optional[np.ndarray] = None,
        scale_0mm: Optional[np.ndarray] = None,
        scale_10mm: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    绘制实验图的灰度图和条纹亮度分布曲线
    支持在图上标注预测的关键点
    :param img_pil: 原始PIL图像
    :param calib_center: 中央条纹坐标 (x0, y0)，原始图像像素值
    :param exp_first_order: 一级条纹坐标 (x1, y1)，原始图像像素值
    :param scale_0mm: 刻度尺0mm坐标 (s0x, s0y)，原始图像像素值
    :param scale_10mm: 刻度尺10mm坐标 (s1x, s1y)，原始图像像素值
    :return: matplotlib Figure对象
    """
    # ===================== 转换图像格式 =====================
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_h, original_w = gray.shape

    # ===================== 创建画布 =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ===================== 绘制灰度图 =====================
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('实验图灰度图', fontsize=14)
    ax1.axis('off')

    # 标注关键点（如果提供）
    colors = ['r', 'g', 'b', 'y']  # 红、绿、蓝、黄
    labels = ['中央条纹', '一级条纹', '0mm刻度', '10mm刻度']
    points = [calib_center, exp_first_order, scale_0mm, scale_10mm]

    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        if point is not None:
            x, y = point
            # 确保坐标在图像范围内
            x = np.clip(x, 0, original_w - 1)
            y = np.clip(y, 0, original_h - 1)
            ax1.scatter(x, y, c=color, s=100, marker='+', linewidth=2, label=label)
            ax1.text(x + 10, y, label, c=color, fontsize=12, fontweight='bold')

    if any(p is not None for p in points):
        ax1.legend(loc='upper right', fontsize=10)

    # ===================== 绘制亮度分布曲线 =====================
    # 沿垂直于条纹的方向取平均亮度（这里假设条纹是垂直的，沿水平方向取平均）
    # 如果需要更智能的方向检测，可以结合预测的中央条纹和一级条纹方向
    brightness_profile = np.mean(gray, axis=0)  # 沿y轴取平均，得到x方向的亮度分布
    x_axis = np.arange(original_w)

    ax2.plot(x_axis, brightness_profile, color='b', linewidth=1.5, label='亮度分布')
    ax2.set_title('条纹亮度分布曲线', fontsize=14)
    ax2.set_xlabel('水平像素位置', fontsize=12)
    ax2.set_ylabel('平均亮度（0-255）', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 标注关键点的x位置（如果提供）
    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        if point is not None:
            x, _ = point
            x = np.clip(x, 0, original_w - 1)
            ax2.axvline(x=x, color=color, linestyle='--', linewidth=1.5, label=f'{label}位置')

    if any(p is not None for p in points):
        ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig
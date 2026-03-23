import numpy as np
from typing import Tuple


def calculate_delta_x(
        calib_center: np.ndarray,  # calib图中央条纹坐标 (x0, y0)，原始图像像素值
        exp_first_order: np.ndarray,  # exp图一级条纹坐标 (x1, y1)，原始图像像素值
        scale_0mm: np.ndarray,  # exp图刻度尺0mm坐标 (s0x, s0y)，原始图像像素值
        scale_10mm: np.ndarray  # exp图刻度尺10mm坐标 (s1x, s1y)，原始图像像素值
) -> Tuple[float, float]:
    """
    计算中央条纹与一级条纹的间距delta_x（单位：mm）
    支持任意方向的刻度尺，通过两点计算像素-毫米比例
    :return: (delta_x, pixel_per_mm)
        - delta_x: 条纹间距（mm）
        - pixel_per_mm: 像素-毫米比例（像素/mm）
    """
    # ===================== 计算像素-毫米比例 =====================
    # 计算刻度尺两点之间的像素距离
    scale_pixel_dist = np.linalg.norm(scale_10mm - scale_0mm)
    # 刻度尺两点之间的实际距离是10mm
    pixel_per_mm = scale_pixel_dist / 10.0

    # ===================== 计算条纹像素间距 =====================
    # 计算中央条纹与一级条纹之间的像素距离
    stripe_pixel_dist = np.linalg.norm(exp_first_order - calib_center)

    # ===================== 转换为毫米 =====================
    delta_x = stripe_pixel_dist / pixel_per_mm

    return delta_x, pixel_per_mm
import numpy as np
from typing import Tuple
import config
def calculate_delta_x(
        calib_center: np.ndarray,
        exp_first_order: np.ndarray,
        scale_start: np.ndarray,
        scale_end: np.ndarray,
        apply_offset_correction: bool = True  # 新增：是否应用偏移修正
) -> Tuple[float, float]:
    """
    计算中央条纹与一级条纹的间距delta_x，支持物理偏移修正
    :param apply_offset_correction: 是否应用固定的2cm物理偏移修正
    """
    real_distance = config.SCALE_REAL_DISTANCE
    # 计算像素-毫米比例
    scale_pixel_dist = np.linalg.norm(scale_end - scale_start)
    pixel_per_mm = scale_pixel_dist / real_distance
    # 计算原始的条纹像素间距
    stripe_pixel_dist = np.linalg.norm(exp_first_order - calib_center)
    delta_x_raw = stripe_pixel_dist / pixel_per_mm
    # ===================== 新增：物理偏移修正 =====================
    if apply_offset_correction:
        # 加上固定的物理偏移（注意正负号）
        delta_x = delta_x_raw + config.FIXED_PHYSICAL_OFFSET
    else:
        delta_x = delta_x_raw
    # =================================================================
    return delta_x, pixel_per_mm
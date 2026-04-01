import numpy as np
from typing import Tuple
import config
def calculate_delta_x(
        calib_center: np.ndarray,
        exp_first_order: np.ndarray,
        scale_start: np.ndarray,
        scale_end: np.ndarray,
        apply_offset_correction: bool = True
) -> Tuple[float, float]:
    real_distance = config.SCALE_REAL_DISTANCE
    # 计算像素-毫米比例
    scale_pixel_dist = np.linalg.norm(scale_end - scale_start)
    pixel_per_mm = scale_pixel_dist / real_distance
    # 计算原始的条纹像素间距
    stripe_pixel_dist = np.linalg.norm(exp_first_order - calib_center)
    delta_x_raw = stripe_pixel_dist / pixel_per_mm
    if apply_offset_correction:
        # 加上固定的物理偏移（注意正负号）
        delta_x = delta_x_raw + config.FIXED_PHYSICAL_OFFSET
    else:
        delta_x = delta_x_raw
    return delta_x, pixel_per_mm
import os
import json
import numpy as np
from typing import Dict, Tuple, Optional
import config
from config import logger


def parse_labelme(
        json_path: str,
        original_img_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    解析labelme的JSON标注文件，返回归一化的关键点坐标
    :param json_path: labelme标注文件路径
    :param original_img_size: 原始图像尺寸 (w, h)，如果为None则从JSON中读取
    :return: (keypoints, img_size)
        - keypoints: (num_keypoints, 2) 归一化坐标（0-1），未标记的点为(-1,-1)
        - img_size: 原始图像尺寸 (w, h)
    """
    # 检查文件是否存在
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"标注文件 {json_path} 不存在！")

    # 读取JSON文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取原始图像尺寸
    if original_img_size is None:
        img_w = data.get("imageWidth", config.IMG_SIZE[0])
        img_h = data.get("imageHeight", config.IMG_SIZE[1])
    else:
        img_w, img_h = original_img_size
    img_size = (img_w, img_h)

    # 初始化关键点数组（默认-1表示未标记）
    keypoints = np.full((config.NUM_KEYPOINTS, 2), -1.0, dtype=np.float32)

    # 遍历标注，填充关键点
    for shape in data["shapes"]:
        label = shape["label"]
        # 只处理点标注（labelme中shape_type为"point"）
        if shape["shape_type"] != "point":
            logger.warning(f"标注文件 {json_path} 中 {label} 不是点标注，已跳过")
            continue

        # 获取点坐标（labelme中坐标是[x, y]，原点在左上角）
        point = np.array(shape["points"][0], dtype=np.float32)

        # 归一化坐标到0-1（防止超出图像范围）
        point_norm = np.clip(point / np.array([img_w, img_h]), 0.0, 1.0)

        # 匹配label到关键点索引
        if label == "calib_center":
            keypoints[0] = point_norm
        elif label == "exp_first_order":
            keypoints[1] = point_norm
        elif label == "scale_0mm":
            keypoints[2] = point_norm
        elif label == "scale_10mm":
            keypoints[3] = point_norm
        else:
            logger.warning(f"标注文件 {json_path} 中存在未知label: {label}，已跳过")

    return keypoints, img_size
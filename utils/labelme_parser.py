import os
import json
import numpy as np
from typing import Tuple, Optional
import config
from config import logger
#把JSON文件变为模型能理解的数值矩阵
def parse_labelme(
        json_path: str,
        original_img_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"标注文件 {json_path} 不存在！")
    #read the JSON securely
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if original_img_size is None:
        img_w = data.get("imageWidth", config.IMG_SIZE[0])
        img_h = data.get("imageHeight", config.IMG_SIZE[1])
    else:
        img_w, img_h = original_img_size
    img_size = (img_w, img_h)
    keypoints = np.full((config.NUM_KEYPOINTS, 2), -1.0, dtype=np.float32)

    for shape in data["shapes"]:
        label = shape["label"]
        if shape["shape_type"] != "point":
            logger.warning(f"标注文件 {json_path} 中 {label} 不是点标注，已跳过")
            continue
        #归一化
        point = np.array(shape["points"][0], dtype=np.float32)
        point_norm = np.clip(point / np.array([img_w, img_h]), 0.0, 1.0)
        if label == "calib_center":
            keypoints[0] = point_norm
        elif label == "exp_blur_center":
            keypoints[1] = point_norm
        elif label == "exp_first_order":
            keypoints[2] = point_norm
        elif label == "scale_200mm":
            keypoints[3] = point_norm
        elif label == "scale_240mm":
            keypoints[4] = point_norm
        else:
            logger.warning(f"标注文件 {json_path} 中存在未知label: {label}，已跳过")

    return keypoints, img_size
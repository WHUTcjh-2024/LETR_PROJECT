import os
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import config
from config import logger
from models.dual_input_letr import DualInputLETR
from utils.transforms import preprocess_single_image


def load_model(model_path: str, device: torch.device) -> DualInputLETR:
    """
    加载训练好的模型
    :param model_path: 模型权重路径
    :param device: 设备（CPU或GPU）
    :return: 加载好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件 {model_path} 不存在！")

    logger.info(f"正在加载模型: {model_path}")
    model = DualInputLETR(pretrained_backbone=False).to(device)
    # 加载模型权重（map_location确保CPU/GPU兼容）
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("模型加载成功")
    return model


def predict_keypoints(
        model: DualInputLETR,
        calib_img: Optional[np.ndarray],
        exp_img: Optional[np.ndarray],
        device: torch.device
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    预测关键点
    :param model: 加载好的模型
    :param calib_img: calib图像（BGR格式，HWC），如果为None则只预测exp的关键点
    :param exp_img: exp图像（BGR格式，HWC），如果为None则只预测calib的关键点
    :param device: 设备
    :return: (calib_center, exp_other_kp, calib_img_size, exp_img_size)
        - calib_center: calib中央条纹坐标 (x0, y0)，原始图像像素值
        - exp_other_kp: exp的其他关键点 (3, 2)，原始图像像素值：[exp_first_order, scale_0mm, scale_10mm]
        - calib_img_size: calib原始图像尺寸 (w, h)
        - exp_img_size: exp原始图像尺寸 (w, h)
    """
    calib_center = None
    exp_other_kp = None
    calib_img_size = None
    exp_img_size = None

    # ===================== 预处理图像 =====================
    calib_tensor = None
    exp_tensor = None

    if calib_img is not None:
        calib_tensor, calib_img_size = preprocess_single_image(calib_img, config.IMG_SIZE)
        calib_tensor = calib_tensor.to(device)

    if exp_img is not None:
        exp_tensor, exp_img_size = preprocess_single_image(exp_img, config.IMG_SIZE)
        exp_tensor = exp_tensor.to(device)

    # ===================== 模型推理 =====================
    with torch.no_grad():
        # 如果只有calib图：用dummy exp图，只取calib的中央点
        if calib_tensor is not None and exp_tensor is None:
            dummy_exp_tensor = torch.zeros_like(calib_tensor).to(device)
            pred_kp = model(calib_tensor, dummy_exp_tensor)
            # 取第一个关键点（calib_center），反归一化
            calib_center_norm = pred_kp[0, 0].cpu().numpy()
            calib_center = calib_center_norm * np.array(calib_img_size)

        # 如果只有exp图：用dummy calib图，只取exp的其他点
        elif exp_tensor is not None and calib_tensor is None:
            dummy_calib_tensor = torch.zeros_like(exp_tensor).to(device)
            pred_kp = model(dummy_calib_tensor, exp_tensor)
            # 取后三个关键点（exp_first_order, scale_0mm, scale_10mm），反归一化
            exp_other_kp_norm = pred_kp[0, 1:].cpu().numpy()
            exp_other_kp = exp_other_kp_norm * np.array(exp_img_size)

        # 如果都有：正常预测
        elif calib_tensor is not None and exp_tensor is not None:
            pred_kp = model(calib_tensor, exp_tensor)
            calib_center_norm = pred_kp[0, 0].cpu().numpy()
            exp_other_kp_norm = pred_kp[0, 1:].cpu().numpy()
            calib_center = calib_center_norm * np.array(calib_img_size)
            exp_other_kp = exp_other_kp_norm * np.array(exp_img_size)

    return calib_center, exp_other_kp, calib_img_size, exp_img_size


def predict_calib_center(model: DualInputLETR, calib_img_pil: Image.Image, device: torch.device) -> np.ndarray:
    """
    预测calib图的中央条纹坐标（简化接口，供app.py调用）
    :param model: 加载好的模型
    :param calib_img_pil: calib的PIL图像
    :param device: 设备
    :return: 中央条纹坐标 (x0, y0)，原始图像像素值
    """
    # 转换PIL图像为BGR格式
    calib_img = cv2.cvtColor(np.array(calib_img_pil), cv2.COLOR_RGB2BGR)
    calib_center, _, _, _ = predict_keypoints(model, calib_img, None, device)
    if calib_center is None:
        raise ValueError("预测calib中央条纹失败！")
    return calib_center


def predict_exp_keypoints(model: DualInputLETR, exp_img_pil: Image.Image, device: torch.device) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    预测exp图的关键点（简化接口，供app.py调用）
    :param model: 加载好的模型
    :param exp_img_pil: exp的PIL图像
    :param device: 设备
    :return: (exp_first_order, scale_0mm, scale_10mm)，原始图像像素值
    """
    exp_img = cv2.cvtColor(np.array(exp_img_pil), cv2.COLOR_RGB2BGR)
    _, exp_other_kp, _, _ = predict_keypoints(model, None, exp_img, device)
    if exp_other_kp is None:
        raise ValueError("预测exp关键点失败！")
    exp_first_order = exp_other_kp[0]
    scale_0mm = exp_other_kp[1]
    scale_10mm = exp_other_kp[2]
    return exp_first_order, scale_0mm, scale_10mm
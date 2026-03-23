import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict
import config
import numpy as np


def get_train_transforms(img_size: Tuple[int, int]) -> A.Compose:
    """
    训练集数据增强：对calib和exp图应用**相同的几何变换**，保证位置对齐
    使用Albumentations的additional_targets实现双图同步变换
    """
    return A.Compose([
        # 几何变换（calib和exp图必须同步）
        A.Resize(height=img_size[1], width=img_size[0], always_apply=True),
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),

        # 光度变换（calib和exp图可独立应用，这里为了简单也同步）
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),

        # 归一化和转Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet预训练模型的均值
            std=[0.229, 0.224, 0.225],  # ImageNet预训练模型的标准差
            always_apply=True
        ),
        ToTensorV2(always_apply=True),
    ],
        # 定义额外的输入目标（exp_img和exp_kp）
        additional_targets={
            'exp_img': 'image',
            'exp_kp': 'keypoints',
            'calib_kp': 'keypoints'
        },
        # 关键点参数：xy格式，保留不可见点（-1的点）
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            check_each_transform=False  # 不检查每个变换后的关键点是否在图像内
        ))


def get_val_transforms(img_size: Tuple[int, int]) -> A.Compose:
    """
    验证集/推理时的预处理：无增强，仅调整尺寸、归一化和转Tensor
    """
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0], always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2(always_apply=True),
    ],
        additional_targets={
            'exp_img': 'image',
            'exp_kp': 'keypoints',
            'calib_kp': 'keypoints'
        },
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            check_each_transform=False
        ))


def preprocess_single_image(
        img: np.ndarray,
        img_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    推理时预处理单张图像（calib或exp）
    :param img: 原始图像（BGR格式，HWC）
    :param img_size: 目标尺寸 (w, h)
    :return: (img_tensor, original_img_size)
        - img_tensor: 预处理后的Tensor (1, 3, H, W)
        - original_img_size: 原始图像尺寸 (w, h)
    """
    original_img_size = (img.shape[1], img.shape[0])  # (w, h)
    transform = A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0)  # 增加batch维度
    return img_tensor, original_img_size
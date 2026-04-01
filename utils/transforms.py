import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import numpy as np
"""数据增强模块"""

def get_train_transforms(img_size: Tuple[int, int]) -> A.Compose:
    """
    使用Albumentations的additional_targets实现双图同步变换
    """
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0], always_apply=True),
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),

        # 光度变换
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),

        # 归一化和转Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406], #torchvision官方预训练模型的标准配置
            std=[0.229, 0.224, 0.225],
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

    original_img_size = (img.shape[1], img.shape[0])  # (w, h)
    transform = A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0)  # 增加batch维度
    return img_tensor, original_img_size
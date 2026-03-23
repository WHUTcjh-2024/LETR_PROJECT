import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List,Dict
import config
from config import logger
from utils.labelme_parser import parse_labelme
from utils.transforms import get_train_transforms, get_val_transforms


class DualInputDataset(Dataset):
    def __init__(self, mode: str = "train"):
        """
        双输入数据集：成对加载calib图（无激励）和exp图（有激励）
        通过文件名前缀匹配：calib_001.jpg 对应 exp_001_*.jpg
        :param mode: "train" 或 "val"
        """
        self.mode = mode
        self.img_size = config.IMG_SIZE

        # ===================== 路径设置 =====================
        self.calib_img_dir = os.path.join(config.DATA_DIR, mode, "calib")
        self.exp_img_dir = os.path.join(config.DATA_DIR, mode, "exp")
        self.calib_label_dir = os.path.join(config.LABEL_DIR, mode, "calib")
        self.exp_label_dir = os.path.join(config.LABEL_DIR, mode, "exp")

        # ===================== 检查路径是否存在 =====================
        for dir_path in [self.calib_img_dir, self.exp_img_dir, self.calib_label_dir, self.exp_label_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"目录 {dir_path} 不存在！")

        # ===================== 获取所有calib文件 =====================
        self.calib_img_files = sorted([
            f for f in os.listdir(self.calib_img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        if len(self.calib_img_files) == 0:
            raise ValueError(f"目录 {self.calib_img_dir} 中没有找到图片！")

        # ===================== 匹配calib和exp文件 =====================
        self.paired_files: List[Tuple[str, List[str]]] = []  # (calib_file, [exp_files])
        for calib_file in self.calib_img_files:
            # 提取calib的实验ID：calib_001.jpg -> 001
            calib_name = os.path.splitext(calib_file)[0]
            if not calib_name.startswith("calib_"):
                logger.warning(f"calib文件 {calib_file} 命名不规范（应为calib_xxx.jpg），已跳过")
                continue
            exp_id = calib_name.split("calib_")[-1]

            # 查找所有对应的exp文件：exp_001_01.jpg, exp_001_02.jpg...
            exp_img_files = sorted([
                f for f in os.listdir(self.exp_img_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg')) and f.startswith(f"exp_{exp_id}_")
            ])
            if len(exp_img_files) == 0:
                logger.warning(f"calib文件 {calib_file} 没有找到对应的exp文件，已跳过")
                continue

            self.paired_files.append((calib_file, exp_img_files))

        # ===================== 展开成对文件（每个exp图对应一个样本） =====================
        self.samples: List[Tuple[str, str]] = []  # (calib_file, exp_file)
        for calib_file, exp_files in self.paired_files:
            for exp_file in exp_files:
                self.samples.append((calib_file, exp_file))

        if len(self.samples) == 0:
            raise ValueError("没有找到有效的calib-exp成对文件！请检查文件命名和路径。")

        logger.info(f"加载 {mode} 集：{len(self.calib_img_files)} 张calib图，{len(self.samples)} 个calib-exp对")

        # ===================== 加载变换 =====================
        if mode == "train":
            self.transform = get_train_transforms(self.img_size)
        else:
            self.transform = get_val_transforms(self.img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ===================== 获取文件路径 =====================
        calib_file, exp_file = self.samples[idx]
        calib_img_path = os.path.join(self.calib_img_dir, calib_file)
        exp_img_path = os.path.join(self.exp_img_dir, exp_file)
        calib_label_path = os.path.join(self.calib_label_dir, os.path.splitext(calib_file)[0] + ".json")
        exp_label_path = os.path.join(self.exp_label_dir, os.path.splitext(exp_file)[0] + ".json")

        # ===================== 读取图像 =====================
        # 读取calib图（BGR -> RGB）
        calib_img = cv2.imread(calib_img_path)
        if calib_img is None:
            raise ValueError(f"无法读取calib图像：{calib_img_path}")
        calib_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2RGB)

        # 读取exp图（BGR -> RGB）
        exp_img = cv2.imread(exp_img_path)
        if exp_img is None:
            raise ValueError(f"无法读取exp图像：{exp_img_path}")
        exp_img = cv2.cvtColor(exp_img, cv2.COLOR_BGR2RGB)

        # ===================== 读取标注 =====================
        calib_kp, _ = parse_labelme(calib_label_path)
        exp_kp, _ = parse_labelme(exp_label_path)

        # ===================== 应用变换 =====================
        # 将关键点从 (num_keypoints, 2) 展平为列表
        calib_kp_list = calib_kp.reshape(-1, 2).tolist()
        exp_kp_list = exp_kp.reshape(-1, 2).tolist()

        # 应用同步变换
        transformed = self.transform(
            image=calib_img,
            calib_kp=calib_kp_list,
            exp_img=exp_img,
            exp_kp=exp_kp_list
        )

        # ===================== 整理数据 =====================
        # 转换为Tensor
        calib_img_tensor = transformed["image"]
        exp_img_tensor = transformed["exp_img"]
        calib_kp_tensor = torch.tensor(transformed["calib_kp"], dtype=torch.float32).reshape(config.NUM_KEYPOINTS, 2)
        exp_kp_tensor = torch.tensor(transformed["exp_kp"], dtype=torch.float32).reshape(config.NUM_KEYPOINTS, 2)

        # 构建目标关键点：calib的中央点 + exp的其他点
        target_kp_tensor = torch.cat([
            calib_kp_tensor[:1, :],  # calib_center
            exp_kp_tensor[1:, :]  # exp_first_order, scale_0mm, scale_10mm
        ], dim=0)

        return {
            "calib_img": calib_img_tensor,  # (3, H, W)
            "exp_img": exp_img_tensor,  # (3, H, W)
            "target_kp": target_kp_tensor  # (4, 2)
        }


def get_dataloader(mode: str = "train") -> DataLoader:
    """获取DataLoader"""
    dataset = DualInputDataset(mode=mode)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
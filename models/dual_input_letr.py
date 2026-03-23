import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.swin_backbone import SwinBackbone
import config
from config import logger


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块：让exp特征关注calib特征的中央条纹区域"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, calib_feat: torch.Tensor, exp_feat: torch.Tensor) -> torch.Tensor:
        """
        :param calib_feat: calib特征图 (B, C, H, W)
        :param exp_feat: exp特征图 (B, C, H, W)
        :return: 融合后的特征图 (B, C, H, W)
        """
        B, C, H, W = calib_feat.shape

        # 计算Query（来自exp）、Key（来自calib）、Value（来自calib）
        query = self.query_conv(exp_feat).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        key = self.key_conv(calib_feat).flatten(2)  # (B, C, H*W)
        value = self.value_conv(calib_feat).flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # 计算注意力权重
        attention = torch.softmax(torch.bmm(query, key) / (C ** 0.5), dim=-1)  # (B, H*W, H*W)

        # 加权求和
        fused_feat = torch.bmm(attention, value)  # (B, H*W, C)
        fused_feat = fused_feat.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)

        # 残差连接和归一化
        fused_feat = self.out_conv(fused_feat) + exp_feat
        fused_feat = self.norm(fused_feat.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W)

        return fused_feat


class KeypointHead(nn.Module):
    """关键点预测头：从融合特征中预测关键点坐标"""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 卷积层提取特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层预测坐标
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_keypoints * 2),
            nn.Sigmoid()  # 归一化到0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 融合特征图 (B, C, H, W)
        :return: 关键点坐标 (B, num_keypoints, 2)，归一化到0-1
        """
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.fc_layers(x)
        x = x.reshape(-1, self.num_keypoints, 2)
        return x


class DualInputLETR(nn.Module):
    """双输入LETR关键点检测模型：结合calib和exp图预测关键点"""

    def __init__(self, num_keypoints: int = config.NUM_KEYPOINTS, pretrained_backbone: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints

        # ===================== Backbone：两个独立的Swin Transformer =====================
        self.calib_backbone = SwinBackbone(pretrained=pretrained_backbone)
        self.exp_backbone = SwinBackbone(pretrained=pretrained_backbone)

        # ===================== 特征融合模块 =====================
        self.fusion = CrossAttentionFusion(
            in_channels=self.calib_backbone.out_channels,
            out_channels=self.calib_backbone.out_channels
        )

        # ===================== 关键点预测头 =====================
        self.keypoint_head = KeypointHead(
            in_channels=self.calib_backbone.out_channels,
            num_keypoints=num_keypoints
        )

        logger.info(f"初始化双输入LETR模型，关键点数量: {num_keypoints}")

    def forward(self, calib_img: torch.Tensor, exp_img: torch.Tensor) -> torch.Tensor:
        """
        :param calib_img: calib图像Tensor (B, 3, H, W)
        :param exp_img: exp图像Tensor (B, 3, H, W)
        :return: 预测的关键点坐标 (B, num_keypoints, 2)，归一化到0-1
        """
        # 分别提取特征
        calib_feat = self.calib_backbone(calib_img)
        exp_feat = self.exp_backbone(exp_img)

        # 特征融合
        fused_feat = self.fusion(calib_feat, exp_feat)

        # 预测关键点
        pred_kp = self.keypoint_head(fused_feat)

        return pred_kp


class CombinedLoss(nn.Module):
    """组合损失函数：关键点回归损失 + 一致性损失"""

    def __init__(self, lambda_consistency: float = 0.5):
        """
        :param lambda_consistency: 一致性损失的权重
        """
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
            self,
            pred_kp: torch.Tensor,
            target_kp: torch.Tensor,
            calib_backbone: nn.Module,
            exp_backbone: nn.Module,
            calib_img: torch.Tensor,
            exp_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param pred_kp: 预测的关键点 (B, 4, 2)
        :param target_kp: 目标关键点 (B, 4, 2)
        :param calib_backbone: calib的backbone（用于一致性损失）
        :param exp_backbone: exp的backbone（用于一致性损失）
        :param calib_img: calib图像 (B, 3, H, W)
        :param exp_img: exp图像 (B, 3, H, W)
        :return: (total_loss, kp_loss, consistency_loss)
        """
        # ===================== 关键点回归损失 =====================
        # 只计算有标注的点（target_kp != -1）
        mask = (target_kp != -1.0).all(dim=-1, keepdim=True).float()  # (B, 4, 1)
        kp_loss = self.smooth_l1(pred_kp * mask, target_kp * mask)

        # ===================== 一致性损失：强制calib和exp的中央点特征相似 =====================
        # 提取calib和exp的中央点特征（简化版：用backbone输出的全局平均特征）
        with torch.no_grad():
            calib_feat = calib_backbone(calib_img)
            exp_feat = exp_backbone(exp_img)
        calib_feat_global = F.adaptive_avg_pool2d(calib_feat, (1, 1)).flatten(1)
        exp_feat_global = F.adaptive_avg_pool2d(exp_feat, (1, 1)).flatten(1)
        consistency_loss = F.mse_loss(calib_feat_global, exp_feat_global)

        # ===================== 总损失 =====================
        total_loss = kp_loss + self.lambda_consistency * consistency_loss

        return total_loss, kp_loss, consistency_loss
import torch
import torch.nn as nn
import timm #PyTorch Image Models
from config import logger
"""基于Swin Transformer的特征提取类，对timm中的预训练模型轻量化封装"""
"""选择 Swin-Tiny 而不是 ResNet 或 ViT：
轻量（参数少）、层次化特征（多尺度）、窗口注意力（适合条纹的局部周期性）"""
class SwinBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, model_name: str = "swin_tiny_patch4_window7_224"):
        """
        Swin Transformer Backbone，用于提取图像特征
        :param pretrained: 是否使用ImageNet预训练权重
        :param model_name: timm中的Swin模型名称
        """
        super().__init__()
        self.model_name = model_name
        # 加载预训练的Swin模型
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头
            global_pool=''  # 去掉全局池化，保留特征图
        )

        # 获取模型的输出特征通道数
        self.out_channels = self.swin.num_features

        # 获取模型的下采样倍数（Swin-Tiny是32倍）
        self.downsample_ratio = 32

        logger.info(f"加载Swin Backbone: {model_name}, 输出通道数: {self.out_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：兼容任意32倍数的输入尺寸
        :param x: 输入图像Tensor (B, 3, H, W)
        :return: 输出特征图Tensor (B, C, H/32, W/32)
        """
        # 先记录原始输入的H和W
        _, _, orig_h, orig_w = x.shape

        # Swin的前向传播
        x = self.swin.patch_embed(x)
        x = self.swin.pos_drop(x)
        for layer in self.swin.layers:
            x = layer(x)
        x = self.swin.norm(x)

        # 转换维度：用原始输入尺寸计算特征图大小
        B, N, C = x.shape
        H_feat = orig_h // self.downsample_ratio
        W_feat = orig_w // self.downsample_ratio
        x = x.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)

        return x
import torch
import torch.nn as nn
import timm
import config
from config import logger


class SwinBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, model_name: str = "swin_tiny_patch4_window7_224"):
        """
        Swin Transformer Backbone，用于提取图像特征
        :param pretrained: 是否使用ImageNet预训练权重
        :param model_name: timm中的Swin模型名称
        """
        super().__init__()
        self.model_name = model_name

        # 加载预训练的Swin模型（去掉分类头）
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
        前向传播
        :param x: 输入图像Tensor (B, 3, H, W)
        :return: 输出特征图Tensor (B, C, H/32, W/32)
        """
        # Swin的前向传播：(B, 3, H, W) -> (B, N, C)，其中N = (H/32)*(W/32)
        x = self.swin.patch_embed(x)  # (B, N, C)
        x = self.swin.pos_drop(x)

        for layer in self.swin.layers:
            x = layer(x)

        x = self.swin.norm(x)  # (B, N, C)

        # 转换为 (B, C, H/32, W/32) 格式
        B, N, C = x.shape
        H_feat = x.shape[1] // (x.shape[2] // self.out_channels)  # 简化计算，实际可根据输入尺寸推导
        # 更准确的方式：根据输入图像尺寸计算特征图尺寸
        input_h, input_w = x.shape[1] // (x.shape[2] // self.out_channels), x.shape[1] // (
                    x.shape[2] // self.out_channels)
        # 这里为了简单，假设输入图像尺寸是32的倍数
        input_h, input_w = config.IMG_SIZE[1], config.IMG_SIZE[0]
        H_feat = input_h // self.downsample_ratio
        W_feat = input_w // self.downsample_ratio

        x = x.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)

        return x
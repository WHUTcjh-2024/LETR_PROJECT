# 🔬 AI+毛细波衍射条纹间距测量系统

**基于双输入LETR的智能标定与自动测量方案**  
**解决振针激励毛细波衍射实验中“中央条纹模糊 + 人工读数误差大”的核心痛点**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) 
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)

---

## 📋 项目简介

本项目针对振针激励毛细波衍射实验中**中央条纹模糊、人工读数误差大、重复性差**的行业痛点，提出了一种**基于双输入LETR（Line Detection Transformer）的端到端AI自动测量方案**。

系统只需**先上传一张无激励calib图完成智能标定**，之后每次上传exp图即可**一键完成多组条纹间距Δx自动计算**，同时生成专业灰度分布曲线并自动保存实验记录，支持CSV导出。

**已实现从数据 → 训练 → 推理 → Web部署的完整闭环**，可直接用于实验室日常实验与科研产出。

---

## ✨ 核心功能

- **🧭 智能标定**：上传calib图（无激励），AI自动精准定位中央亮纹位置
- **⚡ 多组实验**：支持反复上传exp图，快速计算多组条纹间距Δx
- **📈 可视化分析**：自动生成高精度灰度图 + 水平亮度分布曲线（可直接用于论文）
- **📦 实验记录**：自动保存完整历史，支持一键导出CSV
- **🚀 网页交互**：Streamlit美观界面，零代码即可使用

---

## 🚀 创新亮点（核心竞争力）

1. **双输入LETR架构**（核心创新）  
   同时输入calib图（干净参考）与exp图（测量图），通过**交叉注意力融合模块**让exp特征主动“关注”calib图中央条纹区域，彻底解决中央条纹模糊难题。

2. **Swin-Tiny轻量级Backbone**  
   采用预训练Swin-Tiny提取层次化特征，在精度与实时性之间取得极佳平衡。

3. **物理先验驱动的组合损失**  
   关键点回归损失 + 特征一致性损失，将“calib与exp中央条纹物理一致性”直接融入训练，保证模型输出符合物理规律。

4. **端到端物理量计算**  
   集成刻度尺自动标定 + 固定物理偏移校正，实现像素坐标 → 真实毫米间距Δx的完整转换。

---

## 🛠️ 技术架构

- **模型**：`DualInputLETR`（CrossAttentionFusion + KeypointHead + Swin-Tiny）
- **数据**：自定义`DualInputDataset` + Albumentations同步增强 + LabelMe解析
- **训练**：`train.py`（AdamW + ReduceLROnPlateau + 断点续训 + 早停）
- **推理**：`inference.py`（支持单图/双图 + dummy tensor兼容）
- **物理计算**：`delta_x_calculator.py`（刻度尺标定 + 偏移校正）
- **可视化**：`gray_visualizer.py`（灰度图 + 亮度曲线 + 关键点标注）
- **界面**：`app.py`（Streamlit全流程Web界面）

## 🎯 适用场景

大学物理实验教学
光学/流体力学科研实验
需要高精度、重复性强的条纹间距测量场景

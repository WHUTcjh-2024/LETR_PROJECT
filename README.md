 AI+毛细波衍射条纹间距测量系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)

 项目简介
本项目针对“振针激励毛细波衍射实验中中央条纹模糊、人工读数误差大”的痛点，提出了一种基于双输入LETR（Line Detection Transformer）的AI自动测量方案。

 核心功能
- 📌 智能标定：上传无激励的calib图，自动识别中央亮纹位置
- 🧪 多组实验：支持反复上传有激励的exp图，快速计算多组条纹间距
- 📈 可视化分析：自动生成条纹亮度灰度图和分布曲线
- 📋 实验记录：自动保存实验历史，支持导出CSV

 AI创新性
1. 双输入LETR架构：同时处理calib和exp图，通过交叉注意力融合模块让exp特征“关注”calib的中央条纹区域，解决中央条纹模糊问题
2. Swin Transformer Backbone：使用轻量级预训练Swin-Tiny提取特征，兼顾精度和速度
3. 组合损失函数：关键点回归损失 + 特征一致性损失，保证物理位置对齐

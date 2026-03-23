import os
import torch
import yaml
import logging

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== 加载data.yaml配置 =====================
def load_config(config_path: str = "data.yaml") -> dict:
    """加载并解析data.yaml配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

cfg = load_config()

# ===================== 设备配置（自动切换CPU/GPU） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")

# ===================== 路径配置 =====================
DATA_DIR = cfg["data_dir"]
LABEL_DIR = cfg["label_dir"]
MODEL_SAVE_DIR = cfg["model_save_dir"]
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "last_model.pth")

# 创建模型保存目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===================== 模型与训练参数 =====================
IMG_SIZE = tuple(cfg["img_size"])  # (w, h)
NUM_KEYPOINTS = len(cfg["keypoint_classes"])
BATCH_SIZE = cfg["batch_size"]
EPOCHS = cfg["epochs"]
LR = cfg["learning_rate"]
WEIGHT_DECAY = cfg["weight_decay"]
EARLY_STOP_PATIENCE = cfg["early_stop_patience"]
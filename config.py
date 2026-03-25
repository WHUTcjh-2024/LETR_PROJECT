import os
import torch
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "data.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

cfg = load_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")

DATA_DIR = cfg["data_dir"]
LABEL_DIR = cfg["label_dir"]
MODEL_SAVE_DIR = cfg["model_save_dir"]
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "last_model.pth")

IMG_SIZE = tuple(cfg["img_size"])
NUM_KEYPOINTS = len(cfg["keypoint_classes"])  # 现在是5个关键点
BATCH_SIZE = cfg["batch_size"]
EPOCHS = cfg["epochs"]
LR = cfg["learning_rate"]
WEIGHT_DECAY = cfg["weight_decay"]
EARLY_STOP_PATIENCE = cfg["early_stop_patience"]

# ===================== 新增：物理先验配置 =====================
SCALE_REAL_DISTANCE = cfg["scale_real_distance"]
FIXED_PHYSICAL_OFFSET = cfg["fixed_physical_offset"]  # 固定物理偏移（mm）
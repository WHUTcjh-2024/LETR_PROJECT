import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import config
from config import logger
from models.dual_input_letr import DualInputLETR, CombinedLoss
from utils.data_loader import get_dataloader

"""我认为的亮点：
断点续训：类似于打GTA的游戏存档
学习率调度：factor = 0.5,学不会就慢点学
早停：防止过拟合，再学不会就不学了
"""
"""损失函数：CombinedLoss
   优化器：AdamW
   经典的pytorch写法"""
def train():
    #初始化模型、损失函数、优化器
    model = DualInputLETR(pretrained_backbone=True).to(config.DEVICE)
    criterion = CombinedLoss(lambda_consistency=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    try:
        train_loader = get_dataloader(mode="train")
        val_loader = get_dataloader(mode="val")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    best_val_loss = float('inf')
    early_stop_counter = 0
    start_epoch = 0

    if os.path.exists(config.LAST_MODEL_PATH):
        logger.info(f"发现断点模型 {config.LAST_MODEL_PATH}，正在加载...")
        checkpoint = torch.load(config.LAST_MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        early_stop_counter = checkpoint['early_stop_counter']
        logger.info(f"断点加载成功，从第 {start_epoch + 1} 轮开始训练")

    logger.info("开始训练...")
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_total_loss = 0.0
        train_kp_loss = 0.0
        train_consistency_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]")
        for batch in train_pbar:
            # 移动数据到设备
            calib_img = batch["calib_img"].to(config.DEVICE)
            exp_img = batch["exp_img"].to(config.DEVICE)
            target_kp = batch["target_kp"].to(config.DEVICE)

            # 前向传播
            optimizer.zero_grad()#清空梯度
            pred_kp = model(calib_img, exp_img)
            total_loss, kp_loss, consistency_loss = criterion(
                pred_kp, target_kp,
                model.calib_backbone, model.exp_backbone,
                calib_img, exp_img
            )

            # 反向传播
            total_loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()#更新参数

            # 累加损失
            train_total_loss += total_loss.item() * calib_img.size(0)
            train_kp_loss += kp_loss.item() * calib_img.size(0)
            train_consistency_loss += consistency_loss.item() * calib_img.size(0)

            # 更新进度条
            train_pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "KP Loss": f"{kp_loss.item():.4f}",
                "Cons Loss": f"{consistency_loss.item():.4f}"
            })

        # 计算平均训练损失
        train_total_loss /= len(train_loader.dataset)
        train_kp_loss /= len(train_loader.dataset)
        train_consistency_loss /= len(train_loader.dataset)

        model.eval()
        val_total_loss = 0.0
        val_kp_loss = 0.0
        val_consistency_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Val]")
            for batch in val_pbar:
                calib_img = batch["calib_img"].to(config.DEVICE)
                exp_img = batch["exp_img"].to(config.DEVICE)
                target_kp = batch["target_kp"].to(config.DEVICE)

                pred_kp = model(calib_img, exp_img)
                total_loss, kp_loss, consistency_loss = criterion(
                    pred_kp, target_kp,
                    model.calib_backbone, model.exp_backbone,
                    calib_img, exp_img
                )

                val_total_loss += total_loss.item() * calib_img.size(0)
                val_kp_loss += kp_loss.item() * calib_img.size(0)
                val_consistency_loss += consistency_loss.item() * calib_img.size(0)

                val_pbar.set_postfix({
                    "Total Loss": f"{total_loss.item():.4f}",
                    "KP Loss": f"{kp_loss.item():.4f}",
                    "Cons Loss": f"{consistency_loss.item():.4f}"
                })

        # 计算平均验证损失
        val_total_loss /= len(val_loader.dataset)
        val_kp_loss /= len(val_loader.dataset)
        val_consistency_loss /= len(val_loader.dataset)

        # ===================== 日志记录 =====================
        logger.info(
            f"Epoch {epoch + 1}/{config.EPOCHS} - "
            f"Train: Total={train_total_loss:.4f}, KP={train_kp_loss:.4f}, Cons={train_consistency_loss:.4f} - "
            f"Val: Total={val_total_loss:.4f}, KP={val_kp_loss:.4f}, Cons={val_consistency_loss:.4f}"
        )

        scheduler.step(val_total_loss)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, config.LAST_MODEL_PATH)
        logger.info(f"最新模型已保存到 {config.LAST_MODEL_PATH}")

        # 保存最佳模型
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            logger.info(f"✅ 最佳模型已更新并保存到 {config.BEST_MODEL_PATH}，Val Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            logger.info(f"验证损失未改善，早停计数器: {early_stop_counter}/{config.EARLY_STOP_PATIENCE}")

            # 早停
            if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                logger.info(f"验证损失连续 {config.EARLY_STOP_PATIENCE} 轮未改善，提前停止训练")
                break
    logger.info("训练完成！")

if __name__ == "__main__":
    train()
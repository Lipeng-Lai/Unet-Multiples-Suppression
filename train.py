from model.unet import UNet2D
from data.build_data import MyDataset
import torch
import torch.nn as nn
from torch import optim
import time
import os
import csv
from datetime import datetime
from tqdm import tqdm
from loss.customloss import CustomLoss

def main():
    # ================== 配置参数 ==================
    config = {
        "image_path": "/home/wwd/deeplearning/data/image/",
        "label_path": "/home/wwd/deeplearning/data/label/",
        "batch_size": 2,
        "epochs": 100,
        "learning_rate": 1e-4,
        "validation_split": 0.1,
        "save_interval": 20,
        "model_dir": "./checkpoint",
        "log_dir": "./logs",
        "lr_patience": 10,      # 新增调度参数
        "lr_factor": 0.5        # 学习率衰减比率
    }

    # ================== 初始化设置 ==================
    # 创建保存目录
    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # 初始化日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["log_dir"], f"training_log_{timestamp}.csv")
    
    # 写入CSV表头
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # ================== 设备设置 ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================== 模型初始化 ==================
    net = UNet2D(1, 1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        verbose=True
    )
    
    
    criterion = CustomLoss(alpha=0.85, beta=0.15)

    # ================== 数据加载 ==================
    full_dataset = MyDataset(config["image_path"], config["label_path"])
    
    # 更合理的数据集划分
    total_size = len(full_dataset)
    val_size = test_size = int(total_size * config["validation_split"])
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
    )

    # ================== 训练循环 ==================
    best_val_loss = float("inf")
    
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        
        # 训练阶段
        net.train()
        train_loss = 0.0
        total_samples = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", unit="batch") as pbar:
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.float32)
                
                # 前向传播
                output = net(batch_x)
                # loss = criterion(output, (batch_x - batch_y))
                loss = criterion(output, batch_y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计损失
                train_loss += loss.item()
                total_samples += batch_x.size(0)
                pbar.set_postfix({"loss": f"{loss.item()/batch_x.size(0):.6f}"})

        # 计算平均训练损失（按样本数平均）
        avg_train_loss = train_loss / total_samples

        # 验证阶段
        net.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch") as pbar:
            for val_x, val_y in pbar:
                val_x = val_x.to(device, dtype=torch.float32)
                val_y = val_y.to(device, dtype=torch.float32)
                
                output = net(val_x)
                loss = criterion(output, batch_y)
                
                val_loss += loss.item()
                total_val_samples += val_x.size(0)
                pbar.set_postfix({"val_loss": f"{loss.item()/val_x.size(0):.6f}"})

        avg_val_loss = val_loss / total_val_samples
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ================== 记录和保存 ==================
        # 写入日志
        log_data = [
            epoch+1,
            f"{avg_train_loss:.6f}",
            f"{avg_val_loss:.6f}",
            f"{current_lr:.2e}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
            }, os.path.join(config["model_dir"], "best_model.pth"))

        # 定期保存模型
        if (epoch+1) % config["save_interval"] == 0:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
            }, os.path.join(config["model_dir"], f"model_epoch_{epoch+1}.pth"))

        # 打印统计信息
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")

if __name__ == "__main__":
    main()

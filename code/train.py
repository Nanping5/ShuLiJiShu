"""
模型训练主脚本
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from pathlib import Path

from config import TrainConfig, ModelConfig, PROCESSED_DATA_PATH, MODEL_DIR, FIGURE_DIR, SUMMARY_DIR
from models import create_model
from data import load_dataset
from utils import get_logger, LoggerContext
from utils.trainer import Trainer
from utils.visualizer import plot_training_history


def main():
    """主训练流程"""
    # 设置日志
    logger = get_logger("train")
    logger.info("=" * 70)
    logger.info("开始训练流程")
    logger.info("=" * 70)
    
    # 检查设备
    device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载数据（优化：添加归一化和数据增强）
    with LoggerContext(logger, "加载数据集"):
        from config import DataConfig
        datasets = load_dataset(
            str(PROCESSED_DATA_PATH),
            normalize_labels=DataConfig.NORMALIZE_LABELS,
            normalize_params=DataConfig.NORMALIZE_PARAMS,
            use_augmentation=DataConfig.USE_AUGMENTATION
        )
        
        train_loader = DataLoader(
            datasets['train'],
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Windows兼容
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            datasets['val'],
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"训练集: {len(datasets['train'])} 样本")
        logger.info(f"验证集: {len(datasets['val'])} 样本")
        logger.info(f"批大小: {TrainConfig.BATCH_SIZE}")
        logger.info(f"标签归一化: {DataConfig.NORMALIZE_LABELS}")
        logger.info(f"参数归一化: {DataConfig.NORMALIZE_PARAMS}")
        logger.info(f"数据增强: {DataConfig.USE_AUGMENTATION}")
    
    # 创建模型
    with LoggerContext(logger, "初始化模型"):
        model = create_model(ModelConfig(), device=str(device))
        params = model.get_num_parameters()
        logger.info(f"总参数量: {params['total']:,}")
        logger.info(f"可训练参数: {params['trainable']:,}")
    
    # 定义损失函数和优化器（优化版）
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY,
        betas=TrainConfig.BETAS
    )
    
    # 学习率调度器（优化：使用ReduceLROnPlateau）
    scheduler = None
    if TrainConfig.USE_SCHEDULER:
        if TrainConfig.SCHEDULER_TYPE == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=TrainConfig.SCHEDULER_FACTOR,
                patience=TrainConfig.SCHEDULER_PATIENCE,
                min_lr=TrainConfig.SCHEDULER_MIN_LR,
                verbose=True
            )
            logger.info(f"学习率调度: ReduceLROnPlateau (patience={TrainConfig.SCHEDULER_PATIENCE}, factor={TrainConfig.SCHEDULER_FACTOR})")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=0.5
            )
            logger.info(f"学习率调度: StepLR")
    
    logger.info(f"损失函数: {TrainConfig.LOSS_FUNCTION}")
    logger.info(f"优化器: {TrainConfig.OPTIMIZER} (lr={TrainConfig.LEARNING_RATE}, weight_decay={TrainConfig.WEIGHT_DECAY})")
    logger.info(f"梯度裁剪: {TrainConfig.USE_GRAD_CLIP} (value={TrainConfig.GRAD_CLIP_VALUE if TrainConfig.USE_GRAD_CLIP else 'N/A'})")
    
    # 创建训练器（优化：添加梯度裁剪）
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        logger=logger,
        scheduler=scheduler,
        grad_clip=TrainConfig.GRAD_CLIP_VALUE if TrainConfig.USE_GRAD_CLIP else None
    )
    
    # 训练模型
    logger.info("-" * 70)
    logger.info("开始训练...")
    logger.info("-" * 70)
    
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=TrainConfig.EPOCHS,
        patience=TrainConfig.PATIENCE if TrainConfig.EARLY_STOPPING else None,
        min_delta=TrainConfig.MIN_DELTA,
        log_interval=TrainConfig.LOG_INTERVAL
    )
    total_time = time.time() - start_time
    
    logger.info("-" * 70)
    logger.info(f"训练完成！总耗时: {total_time/60:.2f} 分钟")
    logger.info(f"最佳验证损失: {history['best_val_loss']:.6f} (第 {history['best_epoch']+1} 轮)")
    
    # 保存训练历史图表
    with LoggerContext(logger, "保存训练历史图表"):
        fig_path = FIGURE_DIR / "training_history.png"
        plot_training_history(history, save_path=fig_path)
        logger.info(f"图表已保存: {fig_path}")
    
    # 保存训练摘要
    with LoggerContext(logger, "保存训练摘要"):
        summary_path = SUMMARY_DIR / "training_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"""
{'=' * 70}
训练摘要
{'=' * 70}

训练配置:
  - 训练轮数: {history['epochs_trained']}
  - 批大小: {TrainConfig.BATCH_SIZE}
  - 学习率: {TrainConfig.LEARNING_RATE}
  - 优化器: {TrainConfig.OPTIMIZER}
  - 早停: {TrainConfig.EARLY_STOPPING} (耐心={TrainConfig.PATIENCE})

模型信息:
  - 总参数量: {params['total']:,}
  - 可训练参数: {params['trainable']:,}

训练结果:
  - 最佳验证损失: {history['best_val_loss']:.6f}
  - 最佳轮次: {history['best_epoch']+1}
  - 训练时间: {total_time/60:.2f} 分钟
  - 最终训练损失: {history['train_losses'][-1]:.6f}
  - 最终验证损失: {history['val_losses'][-1]:.6f}

输出文件:
  - 最佳模型: {MODEL_DIR / 'best_model.pth'}
  - 最终模型: {MODEL_DIR / 'final_model.pth'}
  - 训练历史: {fig_path}
""")
        logger.info(f"摘要已保存: {summary_path}")
    
    logger.info("=" * 70)
    logger.info("训练流程完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

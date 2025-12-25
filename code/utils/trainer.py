"""
训练器类 - 封装训练逻辑
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import logging

from config import MODEL_DIR


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: logging.Logger,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.grad_clip = grad_clip  # 优化：梯度裁剪
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch（优化版）"""
        self.model.train()
        total_loss = 0.0
        
        for img, param, label in train_loader:
            img = img.to(self.device)
            param = param.to(self.device)
            label = label.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(img, param)
            loss = self.criterion(output, label)
            
            # 反向传播
            loss.backward()
            
            # 优化：梯度裁剪
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for img, param, label in val_loader:
                img = img.to(self.device)
                param = param.to(self.device)
                label = label.to(self.device)
                
                output = self.model(img, param)
                loss = self.criterion(output, label)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: Optional[int] = None,
        min_delta: float = 1e-6,
        log_interval: int = 5
    ) -> dict:
        """
        完整训练流程（优化版）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            patience: 早停耐心值（None表示不使用早停）
            min_delta: 最小改进阈值
            log_interval: 日志打印间隔
        
        Returns:
            训练历史字典
        """
        train_losses = []
        val_losses = []
        learning_rates = []  # 优化：记录学习率变化
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 日志
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )
            
            # 优化：保存最佳模型（考虑min_delta）
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                self.save_checkpoint(
                    MODEL_DIR / 'best_model.pth',
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
                self.logger.info(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
            
            # 优化：学习率调度（支持ReduceLROnPlateau）
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 早停检查
            if patience is not None and patience_counter >= patience:
                self.logger.info(f"早停触发！在第 {epoch+1} 轮停止训练")
                self.logger.info(f"  最佳模型在第 {best_epoch+1} 轮 (Val Loss: {best_val_loss:.6f})")
                break
        
        # 保存最终模型
        self.save_checkpoint(
            MODEL_DIR / 'final_model.pth',
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epochs_trained': epoch + 1
        }
    
    def save_checkpoint(self, path, **kwargs):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

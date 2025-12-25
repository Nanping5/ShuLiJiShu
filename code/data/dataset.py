"""
数据集类定义（优化版）
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms


class AirfoilDataset(Dataset):
    """
    翼型数据集（优化版）
    
    Args:
        images: (N, H, W, 1) numpy数组
        params: (N,) AoA参数
        labels: (N,) Cd标签
        transform: 数据增强变换
        normalize_labels: 是否归一化标签
        normalize_params: 是否归一化参数
        label_stats: 标签统计信息（用于归一化）
        param_stats: 参数统计信息（用于归一化）
    """
    
    def __init__(
        self, 
        images: np.ndarray, 
        params: np.ndarray, 
        labels: np.ndarray,
        transform=None,
        normalize_labels: bool = False,
        normalize_params: bool = False,
        label_stats: dict = None,
        param_stats: dict = None
    ):
        # 转换为PyTorch张量并调整维度
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 1, H, W)
        self.params = torch.tensor(params, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        
        self.transform = transform
        
        # 优化：标签归一化
        if normalize_labels and label_stats is not None:
            self.labels = (self.labels - label_stats['mean']) / label_stats['std']
            self.label_normalized = True
            self.label_stats = label_stats
        else:
            self.label_normalized = False
            self.label_stats = None
        
        # 优化：参数归一化
        if normalize_params and param_stats is not None:
            self.params = (self.params - param_stats['mean']) / param_stats['std']
            self.param_normalized = True
            self.param_stats = param_stats
        else:
            self.param_normalized = False
            self.param_stats = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        param = self.params[idx]
        label = self.labels[idx]
        
        # 优化：数据增强
        if self.transform is not None:
            img = self.transform(img)
        
        return img, param, label
    
    def denormalize_label(self, normalized_label):
        """反归一化标签"""
        if self.label_normalized and self.label_stats is not None:
            return normalized_label * self.label_stats['std'] + self.label_stats['mean']
        return normalized_label
    
    def get_statistics(self):
        """获取数据集统计信息"""
        return {
            'n_samples': len(self),
            'label_mean': self.labels.mean().item(),
            'label_std': self.labels.std().item(),
            'label_min': self.labels.min().item(),
            'label_max': self.labels.max().item(),
            'param_unique': torch.unique(self.params).tolist(),
            'label_normalized': self.label_normalized,
            'param_normalized': self.param_normalized
        }


def load_dataset(data_path: str, normalize_labels: bool = True, normalize_params: bool = True, use_augmentation: bool = False):
    """
    加载预处理后的数据集（优化版）
    
    Args:
        data_path: npz文件路径
        normalize_labels: 是否归一化标签
        normalize_params: 是否归一化参数
        use_augmentation: 是否使用数据增强
    
    Returns:
        字典，包含训练/验证/测试数据
    """
    data = np.load(data_path)
    
    # 优化：计算归一化统计信息（基于训练集）
    label_stats = None
    param_stats = None
    
    if normalize_labels:
        label_mean = data['y_cd_train'].mean()
        label_std = data['y_cd_train'].std()
        label_stats = {'mean': label_mean, 'std': label_std}
    
    if normalize_params:
        param_mean = data['p_train'].mean()
        param_std = data['p_train'].std()
        param_stats = {'mean': param_mean, 'std': param_std}
    
    # 优化：数据增强（仅用于训练集）
    train_transform = None
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(5),  # 随机旋转±5度
            transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),  # 随机缩放
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)  # 添加噪声
        ])
    
    datasets = {
        'train': AirfoilDataset(
            data['X_train'],
            data['p_train'],
            data['y_cd_train'],
            transform=train_transform,
            normalize_labels=normalize_labels,
            normalize_params=normalize_params,
            label_stats=label_stats,
            param_stats=param_stats
        ),
        'val': AirfoilDataset(
            data['X_val'],
            data['p_val'],
            data['y_cd_val'],
            transform=None,  # 验证集不使用数据增强
            normalize_labels=normalize_labels,
            normalize_params=normalize_params,
            label_stats=label_stats,
            param_stats=param_stats
        ),
        'test': AirfoilDataset(
            data['X_test'],
            data['p_test'],
            data['y_cd_test'],
            transform=None,  # 测试集不使用数据增强
            normalize_labels=normalize_labels,
            normalize_params=normalize_params,
            label_stats=label_stats,
            param_stats=param_stats
        )
    }
    
    return datasets


if __name__ == "__main__":
    # 测试数据集加载
    from pathlib import Path
    
    data_path = Path(__file__).parent / "dataset.npz"
    if data_path.exists():
        datasets = load_dataset(str(data_path))
        print("数据集加载成功:")
        for split, dataset in datasets.items():
            stats = dataset.get_statistics()
            print(f"\n{split}集:")
            print(f"  样本数: {stats['n_samples']}")
            print(f"  标签均值: {stats['label_mean']:.6f}")
            print(f"  标签范围: [{stats['label_min']:.6f}, {stats['label_max']:.6f}]")

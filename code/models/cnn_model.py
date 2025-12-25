"""
CNN模型定义 - 改进版
使用FiLM (Feature-wise Linear Modulation) 融合AoA参数
"""
import torch
import torch.nn as nn
from config import ModelConfig


class FiLMLayer(nn.Module):
    """
    FiLM层：用条件参数调制特征
    gamma * feature + beta
    """
    def __init__(self, num_features, condition_dim=32):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, num_features)
        self.beta_fc = nn.Linear(condition_dim, num_features)
        
        # 初始化：gamma接近1，beta接近0
        nn.init.ones_(self.gamma_fc.weight.data * 0.1)
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)
    
    def forward(self, x, condition):
        """
        x: (B, C, H, W) 特征图
        condition: (B, condition_dim) 条件向量
        """
        gamma = self.gamma_fc(condition).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_fc(condition).unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * x + beta


class ConvBlock(nn.Module):
    """卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class AirfoilCNN(nn.Module):
    """
    翼型阻力系数预测CNN模型（改进版）
    
    改进点：
    1. 使用FiLM融合AoA - 让AoA调制图像特征
    2. 简化网络结构 - 减少过拟合
    3. 更好的初始化
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        self.config = config
        
        # AoA嵌入网络
        self.aoa_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 卷积层
        self.conv_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        
        in_ch = config.INPUT_CHANNELS
        for out_ch in config.CONV_CHANNELS:
            self.conv_blocks.append(ConvBlock(in_ch, out_ch, config.USE_BATCH_NORM))
            if config.AOA_FUSION == 'film':
                self.film_layers.append(FiLMLayer(out_ch, 32))
            in_ch = out_ch
        
        # 计算展平后的尺寸
        n_pools = len(config.CONV_CHANNELS)
        flat_size = config.CONV_CHANNELS[-1] * (config.IMG_SIZE // (2 ** n_pools)) ** 2
        
        # 全连接层
        fc_layers = []
        if config.AOA_FUSION == 'concat':
            in_features = flat_size + 32  # 拼接AoA嵌入
        else:
            in_features = flat_size
        
        for hidden_dim in config.FC_HIDDEN_DIMS:
            fc_layers.append(nn.Linear(in_features, hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(config.DROPOUT_RATE))
            in_features = hidden_dim
        
        fc_layers.append(nn.Linear(in_features, config.OUTPUT_DIM))
        self.fc = nn.Sequential(*fc_layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, img, param):
        """
        img: (B, 1, H, W)
        param: (B, 1) AoA
        """
        # AoA嵌入
        aoa_emb = self.aoa_embed(param)  # (B, 32)
        
        # 卷积 + FiLM调制
        x = img
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            if self.config.AOA_FUSION == 'film' and i < len(self.film_layers):
                x = self.film_layers[i](x, aoa_emb)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 如果是concat模式，拼接AoA嵌入
        if self.config.AOA_FUSION == 'concat':
            x = torch.cat([x, aoa_emb], dim=1)
        
        # 全连接预测
        return self.fc(x)
    
    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def create_model(config: ModelConfig = None, device: str = 'cpu'):
    model = AirfoilCNN(config)
    return model.to(device)


if __name__ == "__main__":
    model = create_model()
    print(model)
    print(f"\n参数量: {model.get_num_parameters()}")
    
    # 测试
    img = torch.randn(4, 1, 128, 128)
    param = torch.randn(4, 1)
    out = model(img, param)
    print(f"输入: img={img.shape}, param={param.shape}")
    print(f"输出: {out.shape}")

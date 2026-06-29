import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# 胶囊网络中的Squash激活函数
def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (torch.sqrt(squared_norm) + 1e-8)


class CapsuleLayer(nn.Module):
    """基本胶囊层"""

    def __init__(self, num_capsules, in_channels, out_channels, routing_iters=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.routing_iters = routing_iters

        # 转换矩阵
        self.W = nn.Parameter(torch.randn(num_capsules, in_channels, out_channels))

    def forward(self, x):
        # x: [batch_size, num_input_capsules, in_channels]
        batch_size = x.size(0)

        # 扩展输入和权重
        x = x.unsqueeze(1).repeat(1, self.num_capsules, 1, 1)
        W = self.W.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # u_hat: [batch_size, num_capsules, num_input_capsules, out_channels]
        u_hat = torch.matmul(x, W)

        # 初始化路由系数
        b_ij = torch.zeros(batch_size, self.num_capsules, x.size(2), 1, device=x.device)

        # 动态路由
        for _ in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=1)
            s_j = (c_ij * u_hat).sum(dim=2)
            v_j = squash(s_j)

            if _ < self.routing_iters - 1:
                # 更新路由系数
                a_ij = torch.matmul(u_hat, v_j.unsqueeze(-1))
                b_ij = b_ij + a_ij

        return v_j  # [batch_size, num_capsules, out_channels]


class AttributeCapsules(nn.Module):
    """特定属性的胶囊层，每个胶囊编码一个可解释的视觉属性"""

    def __init__(self, num_attributes, input_dim, capsule_dim):
        super().__init__()
        self.num_attributes = num_attributes
        self.capsule_dim = capsule_dim

        # 从特征提取器到属性胶囊的映射
        self.primary_caps = nn.Conv2d(input_dim, num_attributes * capsule_dim, kernel_size=1)
        self.attribute_names = None  # 可以在这里存储属性名称以便于解释

    def forward(self, x):
        # x: [batch_size, input_dim, height, width]
        batch_size = x.size(0)

        # 生成初级胶囊
        primary = self.primary_caps(x)

        # 重塑为胶囊格式
        primary = primary.view(batch_size, self.num_attributes, self.capsule_dim, -1)
        primary = primary.permute(0, 1, 3, 2).contiguous()
        primary = primary.view(batch_size, self.num_attributes, -1, self.capsule_dim)

        # 对每个位置的胶囊应用squash
        capsules = squash(primary, dim=-1)

        # 聚合空间维度，得到每个属性的单一胶囊
        capsules = capsules.mean(dim=2)  # [batch_size, num_attributes, capsule_dim]

        return capsules


class DiagnosisCapsules(nn.Module):
    """诊断胶囊，基于属性胶囊进行分类"""

    def __init__(self, num_attributes, attribute_dim, num_classes, routing_iters=3):
        super().__init__()
        self.capsule_layer = CapsuleLayer(
            num_capsules=num_classes,
            in_channels=attribute_dim,
            out_channels=16,  # 诊断胶囊的维度
            routing_iters=routing_iters
        )

    def forward(self, x):
        # x: [batch_size, num_attributes, attribute_dim]
        return self.capsule_layer(x)  # [batch_size, num_classes, 16]


class AttributeDecoder(nn.Module):
    """从属性胶囊重建原始属性分数"""

    def __init__(self, num_attributes, capsule_dim, hidden_dim=64):
        super().__init__()
        self.num_attributes = num_attributes

        self.decoder = nn.Sequential(
            nn.Linear(num_attributes * capsule_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_attributes),
            nn.Sigmoid()  # 假设属性分数已归一化为0-1
        )

    def forward(self, x):
        # x: [batch_size, num_attributes, capsule_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.decoder(x)  # [batch_size, num_attributes]


class HiFuseCapsNet(nn.Module):
    """结合HiFuse和CapsNet的多任务可解释网络"""

    def __init__(self, base_model, num_attributes, attribute_dim=8, num_classes=2):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_classes = num_classes

        # 使用HiFuse作为特征提取器
        self.feature_extractor = base_model

        # 移除原有的分类头
        self.feature_extractor.conv_head = nn.Identity()

        # 特征维度 (取决于HiFuse的输出维度)
        feature_dim = 768  # 对于HiFuse_Small是384，根据具体模型调整

        # 属性胶囊层
        self.attribute_capsules = AttributeCapsules(
            num_attributes=num_attributes,
            input_dim=feature_dim,
            capsule_dim=attribute_dim
        )

        # 诊断胶囊层
        self.diagnosis_capsules = DiagnosisCapsules(
            num_attributes=num_attributes,
            attribute_dim=attribute_dim,
            num_classes=num_classes
        )

        # 属性解码器 (用于监督学习属性)
        self.attribute_decoder = AttributeDecoder(
            num_attributes=num_attributes,
            capsule_dim=attribute_dim
        )

        # 用于预测最终恶性概率的附加层
        self.malignancy_predictor = nn.Sequential(
            nn.Linear(num_classes * 16, 32),  # 16是诊断胶囊的维度
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)

        # 将特征重塑为2D特征图以适应胶囊网络
        batch_size = features.size(0)
        features = features.view(batch_size, -1, 1, 1)

        # 提取属性胶囊
        attribute_caps = self.attribute_capsules(features)

        # 生成诊断胶囊
        diagnosis_caps = self.diagnosis_capsules(attribute_caps)

        # 计算类别概率 (胶囊长度)
        class_lengths = torch.sqrt((diagnosis_caps ** 2).sum(dim=-1))
        class_probs = F.softmax(class_lengths, dim=-1)

        # 预测属性分数
        attribute_scores = self.attribute_decoder(attribute_caps)

        # 预测恶性程度
        diagnosis_flat = diagnosis_caps.view(batch_size, -1)
        malignancy_score = self.malignancy_predictor(diagnosis_flat)

        return {
            'class_probs': class_probs,  # [batch_size, num_classes]
            'malignancy_score': malignancy_score,  # [batch_size, 1]
            'attribute_scores': attribute_scores,  # [batch_size, num_attributes]
            'attribute_capsules': attribute_caps  # [batch_size, num_attributes, attribute_dim]
        }

    def loss_function(self, outputs, targets, attribute_targets=None, lambda_attr=0.5, lambda_recon=0.1):
        """多任务损失函数"""
        # 分类损失
        classification_loss = F.cross_entropy(outputs['class_probs'], targets)

        # 恶性程度预测损失 (假设targets也包含恶性程度标签)
        malignancy_loss = F.binary_cross_entropy(
            outputs['malignancy_score'].squeeze(),
            targets.float() if targets.dim() == 1 else targets.float().squeeze()
        )

        total_loss = classification_loss + malignancy_loss

        # 如果提供了属性标签，增加属性预测损失
        if attribute_targets is not None:
            attribute_loss = F.mse_loss(outputs['attribute_scores'], attribute_targets)
            total_loss += lambda_attr * attribute_loss

        return total_loss


def create_hifuse_capsnet(base_model, num_attributes, attribute_names=None):
    """创建HiFuseCapsNet模型

    Args:
        base_model: 预训练的HiFuse模型
        num_attributes: 要编码的视觉属性数量
        attribute_names: 属性名称列表，用于模型解释

    Returns:
        配置好的HiFuseCapsNet模型
    """
    model = HiFuseCapsNet(base_model, num_attributes)

    # 设置属性名称以便于解释
    if attribute_names:
        model.attribute_capsules.attribute_names = attribute_names

    return model

# 使用示例
# from your_hifuse_module import HiFuse_Small

# # 肺结节相关的可解释视觉属性
# attribute_names = [
#     "Spiculation", "Lobulation", "Margin", "Sphericity",
#     "Subtlety", "Texture", "Calcification", "Internal Structure"
# ]

# # 创建基础HiFuse模型
# base_model = HiFuse_Small(num_classes=2)  # 二分类问题:良性/恶性

# # 创建HiFuseCapsNet
# model = create_hifuse_capsnet(
#     base_model=base_model,
#     num_attributes=len(attribute_names),
#     attribute_names=attribute_names
# )

# # 模型训练和评估代码...
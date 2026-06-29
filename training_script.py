import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from dataset_loader import get_data_loaders
# 假设HiFuse模型的导入路径
from hifuse_model import HiFuse_Small, HiFuse_Mini

# 导入胶囊网络
# 这里应该导入您之前定义的HiFuseCapsNet类

from HiFuse_CapsNet import create_hifuse_capsnet

# 为简化，这里直接定义相关代码
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_class_weights(csv_file):
    """计算类别权重"""
    df = pd.read_csv(csv_file)
    class_counts = df['class'].value_counts().to_dict()

    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)

    # 计算每个类别的权重 (样本越少权重越高)
    weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}

    # 转换为张量
    weight_tensor = torch.FloatTensor([weights[i] for i in range(n_classes)])

    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights}")

    return weight_tensor


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
        # 或 [batch_size, input_dim, 1, 1] 如果已经是全局特征
        batch_size = x.size(0)

        # 对于HiFuse模型，输出可能已经是全局特征
        # 检查输入尺寸
        if x.size(2) == 1 and x.size(3) == 1:
            # 如果已经是全局特征，直接使用全连接层
            x = x.view(batch_size, -1)
            # 创建每个属性的胶囊
            capsules = torch.zeros(batch_size, self.num_attributes, self.capsule_dim, device=x.device)
            for i in range(self.num_attributes):
                # 创建属性特定的线性层
                if not hasattr(self, f'attr_fc_{i}'):
                    setattr(self, f'attr_fc_{i}', nn.Linear(x.size(1), self.capsule_dim).to(x.device))

                attr_fc = getattr(self, f'attr_fc_{i}')
                capsules[:, i] = squash(attr_fc(x))

            return capsules
        else:
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
        # 对于HiFuse_Small是384，HiFuse_Mini是256
        if hasattr(base_model, 'conv_norm') and hasattr(base_model.conv_norm, 'normalized_shape'):
            feature_dim = base_model.conv_norm.normalized_shape[0]
        else:
            # 默认值，可能需要调整
            feature_dim = 384

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

    def loss_function(self, outputs, targets, attribute_targets=None, class_weights=None, lambda_attr=0.5,
                      lambda_recon=0.1):
        """多任务损失函数"""
        # 分类损失，添加类别权重
        if class_weights is not None and class_weights.device != targets.device:
            class_weights = class_weights.to(targets.device)

        # 分类损失 - 确保这是标量
        classification_loss = F.cross_entropy(
            outputs['class_probs'],
            targets,
            weight=class_weights
        )
        classification_loss = torch.mean(classification_loss)  # 确保是标量

        # 恶性程度预测损失 - 确保这是标量
        mal_targets = targets.float()
        if mal_targets.dim() > 1:
            mal_targets = mal_targets.squeeze()
        if outputs['malignancy_score'].dim() > 1:
            mal_scores = outputs['malignancy_score'].squeeze()
        else:
            mal_scores = outputs['malignancy_score']

        # 确保维度匹配
        if mal_targets.dim() == 0:
            mal_targets = mal_targets.unsqueeze(0)
        if mal_scores.dim() == 0:
            mal_scores = mal_scores.unsqueeze(0)

        malignancy_loss = F.binary_cross_entropy(mal_scores, mal_targets)
        malignancy_loss = torch.mean(malignancy_loss)  # 确保是标量

        # 合并损失 - 确保每个组件都是标量
        total_loss = classification_loss + malignancy_loss

        # 如果提供了属性标签，增加属性预测损失
        if attribute_targets is not None:
            attribute_loss = F.mse_loss(outputs['attribute_scores'], attribute_targets)
            attribute_loss = torch.mean(attribute_loss)  # 确保是标量
            total_loss = total_loss + lambda_attr * attribute_loss

        return total_loss


class AttrExplainer:
    """用于解释模型预测的工具类"""

    def __init__(self, model, attribute_names):
        self.model = model
        self.attribute_names = attribute_names
        self.caps_dim = model.attribute_capsules.capsule_dim

    def explain_prediction(self, image, true_label=None, true_attrs=None):
        """
        解释单个图像的预测结果

        Args:
            image: 输入图像张量 [1, C, H, W]
            true_label: 真实标签，可选
            true_attrs: 真实属性值，可选

        Returns:
            explanation_dict: 包含解释信息的字典
        """
        # 确保模型处于评估模式
        self.model.eval()

        # 前向传播
        with torch.no_grad():
            outputs = self.model(image)

        # 获取预测结果
        class_probs = outputs['class_probs'][0].cpu().numpy()
        pred_class = class_probs.argmax()
        pred_attrs = outputs['attribute_scores'][0].cpu().numpy()
        concept_contrib = outputs.get('concept_contributions')
        if concept_contrib is not None:
            concept_contrib = concept_contrib[0].cpu().numpy()
        complement_scores = outputs.get('complement_scores')
        if complement_scores is not None:
            complement_scores = complement_scores[0].cpu().numpy()
        attr_heatmaps = outputs.get('attr_heatmaps')
        if attr_heatmaps is not None:
            attr_heatmaps = attr_heatmaps[0].cpu().numpy()

        # 获取属性胶囊
        attr_caps = outputs['attribute_capsules'][0].cpu().numpy()

        # 计算每个属性的重要性（胶囊向量的长度）
        attr_importance = np.sqrt((attr_caps ** 2).sum(axis=1))

        # 归一化重要性分数
        attr_importance = attr_importance / (attr_importance.sum() + 1e-8)

        # 构建解释字典
        explanation = {
            'prediction': 'Malignant' if pred_class == 1 else 'Benign',
            'confidence': float(class_probs[pred_class]),
            'malignancy_score': float(outputs['malignancy_score'].view(-1)[0].cpu().item()),
            'concept_malignancy_score': float(outputs['concept_malignancy_score'].view(-1)[0].cpu().item()),
            'attribute_scores': {name: float(score) for name, score in zip(self.attribute_names, pred_attrs)},
            'attribute_importance': {name: float(imp) for name, imp in zip(self.attribute_names, attr_importance)},
            'concept_contributions': {
                name: float(value) for name, value in zip(self.attribute_names, concept_contrib)
            } if concept_contrib is not None else None,
            'complement_scores': {
                f'complement_{i + 1}': float(value) for i, value in enumerate(complement_scores)
            } if complement_scores is not None else None,
            'attribute_heatmaps': {
                name: attr_heatmaps[i] for i, name in enumerate(self.attribute_names)
            } if attr_heatmaps is not None else None
        }

        # 如果提供了真实标签
        if true_label is not None:
            explanation['true_label'] = 'Malignant' if true_label == 1 else 'Benign'
            explanation['correct'] = (pred_class == true_label)

        # 如果提供了真实属性
        if true_attrs is not None:
            explanation['true_attributes'] = {name: float(val) for name, val in zip(self.attribute_names, true_attrs)}
            explanation['attribute_error'] = {
                name: float(abs(pred - true))
                for name, pred, true in zip(self.attribute_names, pred_attrs, true_attrs)
            }

        return explanation

    def visualize_explanation(self, explanation, save_path=None):
        """可视化解释结果"""
        # 创建图表
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # 绘制属性评分
        attr_names = list(explanation['attribute_scores'].keys())
        attr_scores = list(explanation['attribute_scores'].values())
        attr_importance = list(explanation['attribute_importance'].values())

        # 按重要性排序
        sorted_idx = np.argsort(attr_importance)[::-1]
        sorted_names = [attr_names[i] for i in sorted_idx]
        sorted_scores = [attr_scores[i] for i in sorted_idx]
        sorted_importance = [attr_importance[i] for i in sorted_idx]

        # 属性评分条形图
        bars = axs[0].barh(sorted_names, sorted_scores, color='skyblue')
        axs[0].set_title('Predicted Attribute Scores')
        axs[0].set_xlim(0, 1)

        # 如果有真实属性，同时显示
        if 'true_attributes' in explanation:
            true_scores = [explanation['true_attributes'][name] for name in sorted_names]
            axs[0].barh(sorted_names, true_scores, color='lightcoral', alpha=0.6)
            axs[0].legend(['Predicted', 'True'])

        # 属性重要性条形图
        cmap = plt.cm.get_cmap('YlOrRd')
        colors = [cmap(imp) for imp in sorted_importance]

        bars = axs[1].barh(sorted_names, sorted_importance, color=colors)
        axs[1].set_title('Attribute Importance for Prediction')
        axs[1].set_xlim(0, max(sorted_importance) * 1.1)

        # 添加预测结果文本
        pred_text = f"Prediction: {explanation['prediction']} (Confidence: {explanation['confidence']:.2f})\n"
        pred_text += f"Malignancy Score: {explanation['malignancy_score']:.2f}"

        if 'true_label' in explanation:
            pred_text += f"\nTrue Label: {explanation['true_label']}"
            if explanation['correct']:
                pred_text += " ✓"
            else:
                pred_text += " ✗"

        fig.suptitle(pred_text, fontsize=14)
        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path)

        plt.show()

        return fig


def train_epoch(model, train_loader, optimizer, device, epoch, class_weights=None):
    """执行一个训练周期"""
    model.train()
    running_loss = 0.0
    running_components = {
        'classification_loss': 0.0,
        'attribute_loss': 0.0,
        'ce_loss': 0.0,
        'malignancy_loss': 0.0,
        'weighted_classification_loss': 0.0,
        'weighted_attribute_loss': 0.0,
        'concept_cls_loss': 0.0,
        'complement_cls_loss': 0.0,
        'consistency_loss': 0.0,
    }
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        attributes = batch['attributes'].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        # 计算损失
        loss, loss_components = model.loss_function(
            outputs, labels, attributes, class_weights, return_components=True
        )

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 更新统计
        running_loss += loss.item()
        for name in running_components:
            running_components[name] += loss_components[name].item()

        # 计算准确率
        _, predicted = outputs['class_probs'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'cls': running_components['classification_loss'] / (batch_idx + 1),
            'attr': running_components['attribute_loss'] / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    avg_components = {name: value / len(train_loader) for name, value in running_components.items()}
    return running_loss / len(train_loader), 100. * correct / total, avg_components


def validate(model, val_loader, device,class_weights=None):
    """在验证集上评估模型"""
    model.eval()
    val_loss = 0
    val_components = {
        'classification_loss': 0.0,
        'attribute_loss': 0.0,
        'ce_loss': 0.0,
        'malignancy_loss': 0.0,
        'weighted_classification_loss': 0.0,
        'weighted_attribute_loss': 0.0,
        'concept_cls_loss': 0.0,
        'complement_cls_loss': 0.0,
        'consistency_loss': 0.0,
    }
    correct = 0
    total = 0

    # 用于计算属性预测精度
    # attr_pred_error = torch.zeros(val_loader.dataset.dataset.available_attributes).to(device)
    attr_pred_error = torch.zeros(len(val_loader.attribute_names)).to(device)
    attr_count = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            attributes = batch['attributes'].to(device)

            outputs = model(images)
            loss, loss_components = model.loss_function(
                outputs, labels, attributes, class_weights, return_components=True
            )

            val_loss += loss.item()
            for name in val_components:
                val_components[name] += loss_components[name].item()

            # 计算分类准确率
            _, predicted = outputs['class_probs'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 收集属性预测误差
            attr_pred_error += torch.sum(torch.abs(outputs['attribute_scores'] - attributes), dim=0)
            attr_count += attributes.size(0)

            # 收集标签和预测，用于计算精确率、召回率等
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算平均每个属性的预测绝对误差
    avg_attr_error = attr_pred_error.cpu().numpy() / attr_count

    # 计算额外的性能指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_components = {name: value / len(val_loader) for name, value in val_components.items()}

    return {
        'loss': val_loss / len(val_loader),
        'accuracy': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'attr_error': avg_attr_error,
        **avg_components
    }


def train_model(csv_file, model_type='small', num_epochs=50, batch_size=16, learning_rate=0.0001,
                image_size=128, save_dir='./models', device=None, use_sampler=False,
                use_class_weights=False, lambda_attr=0.05, lambda_concept_cls=0.05,
                lambda_consistency=0.02, lambda_complement=0.0):
    """
    训练HiFuseCapsNet模型

    Args:
        csv_file: 数据集CSV文件
        model_type: 'mini'或'small'，选择HiFuse的基础模型大小
        num_epochs: 训练周期数
        batch_size: 批次大小
        learning_rate: 学习率
        image_size: 图像大小
        save_dir: 模型保存目录
        device: 训练设备
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    train_loader, val_loader, test_loader, attribute_names = get_data_loaders(
        csv_file, batch_size=batch_size, image_size=image_size, use_sampler=use_sampler
    )

    num_attributes = len(attribute_names)
    print(f"Number of attributes: {num_attributes}")
    print(f"Attribute names: {attribute_names}")

    # 计算类别权重
    if use_class_weights:
        class_weights = calculate_class_weights(csv_file).to(device)
    else:
        class_weights = None
        print("Class weights disabled.")

    # 创建基础模型
    if model_type.lower() == 'mini':
        base_model = HiFuse_Mini(num_classes=2, patch_size=2, window_size=2)
    else:
        base_model = HiFuse_Small(num_classes=2, patch_size=2, window_size=2)

    # 创建HiFuseCapsNet模型
    # 使用 HiFuse_CapsNet.py 中的新版可解释模型，保留空间热力图和严格概念瓶颈。
    model = create_hifuse_capsnet(
        base_model,
        num_attributes,
        attribute_names,
        lambda_attr=lambda_attr,
        lambda_concept_cls=lambda_concept_cls,
        lambda_consistency=lambda_consistency,
        lambda_complement=lambda_complement,
    )

    # 创建HiFuseCapsNet模型
    model.to(device)

    # 设置优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练循环
    best_val_acc = 0
    best_val_loss = float('inf')

    # 存储训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_attr_error': [],
        'train_cls_loss': [],
        'train_attr_loss': [],
        'train_ce_loss': [],
        'train_malignancy_loss': [],
        'train_weighted_cls_loss': [],
        'train_weighted_attr_loss': [],
        'train_concept_cls_loss': [],
        'train_complement_cls_loss': [],
        'train_consistency_loss': [],
        'val_cls_loss': [],
        'val_attr_loss': [],
        'val_ce_loss': [],
        'val_malignancy_loss': [],
        'val_weighted_cls_loss': [],
        'val_weighted_attr_loss': [],
        'val_concept_cls_loss': [],
        'val_complement_cls_loss': [],
        'val_consistency_loss': []
    }

    for epoch in range(1, num_epochs + 1):
        # 训练一个周期
        train_loss, train_acc, train_components = train_epoch(
            model, train_loader, optimizer, device, epoch, class_weights
        )

        # 验证
        val_metrics = validate(model, val_loader, device, class_weights)

        # 打印进度
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"Train Components | cls(raw): {train_components['classification_loss']:.4f}, "
            f"attr(raw): {train_components['attribute_loss']:.4f}, "
            f"CE: {train_components['ce_loss']:.4f}, "
            f"malignancy BCE: {train_components['malignancy_loss']:.4f}, "
            f"comp BCE: {train_components['complement_cls_loss']:.4f}, "
            f"consistency: {train_components['consistency_loss']:.4f}, "
            f"cls(weighted): {train_components['weighted_classification_loss']:.4f}, "
            f"attr(weighted): {train_components['weighted_attribute_loss']:.4f}"
        )
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(
            f"Val Components   | cls(raw): {val_metrics['classification_loss']:.4f}, "
            f"attr(raw): {val_metrics['attribute_loss']:.4f}, "
            f"CE: {val_metrics['ce_loss']:.4f}, "
            f"malignancy BCE: {val_metrics['malignancy_loss']:.4f}, "
            f"comp BCE: {val_metrics['complement_cls_loss']:.4f}, "
            f"consistency: {val_metrics['consistency_loss']:.4f}, "
            f"cls(weighted): {val_metrics['weighted_classification_loss']:.4f}, "
            f"attr(weighted): {val_metrics['weighted_attribute_loss']:.4f}"
        )
        print(
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Avg. Attribute Error: {np.mean(val_metrics['attr_error']):.4f}")

        # 更新学习率
        scheduler.step(val_metrics['loss'])

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_attr_error'].append(np.mean(val_metrics['attr_error']))
        history['train_cls_loss'].append(train_components['classification_loss'])
        history['train_attr_loss'].append(train_components['attribute_loss'])
        history['train_ce_loss'].append(train_components['ce_loss'])
        history['train_malignancy_loss'].append(train_components['malignancy_loss'])
        history['train_weighted_cls_loss'].append(train_components['weighted_classification_loss'])
        history['train_weighted_attr_loss'].append(train_components['weighted_attribute_loss'])
        history['train_concept_cls_loss'].append(train_components['concept_cls_loss'])
        history['train_complement_cls_loss'].append(train_components['complement_cls_loss'])
        history['train_consistency_loss'].append(train_components['consistency_loss'])
        history['val_cls_loss'].append(val_metrics['classification_loss'])
        history['val_attr_loss'].append(val_metrics['attribute_loss'])
        history['val_ce_loss'].append(val_metrics['ce_loss'])
        history['val_malignancy_loss'].append(val_metrics['malignancy_loss'])
        history['val_weighted_cls_loss'].append(val_metrics['weighted_classification_loss'])
        history['val_weighted_attr_loss'].append(val_metrics['weighted_attribute_loss'])
        history['val_concept_cls_loss'].append(val_metrics['concept_cls_loss'])
        history['val_complement_cls_loss'].append(val_metrics['complement_cls_loss'])
        history['val_consistency_loss'].append(val_metrics['consistency_loss'])

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(save_dir, 'best_acc_model.pth'))
            print("Saved best accuracy model!")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best_loss_model.pth'))
            print("Saved best loss model!")

        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, os.path.join(save_dir, 'final_model.pth'))

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    # 绘制训练历史
    plot_training_history(history, save_dir)

    # 在测试集上进行评估
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Avg. Attribute Error: {np.mean(test_metrics['attr_error']):.4f}")

    # 保存测试结果
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.2f}%\n")
        f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Test F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"Avg. Attribute Error: {np.mean(test_metrics['attr_error']):.4f}\n")
        for i, attr in enumerate(attribute_names):
            f.write(f"{attr} Error: {test_metrics['attr_error'][i]:.4f}\n")

    # 创建解释器
    explainer = AttrExplainer(model, attribute_names)

    # 随机选择一些测试样本来生成解释
    explanation_dir = os.path.join(save_dir, 'explanations')
    os.makedirs(explanation_dir, exist_ok=True)

    print("\nGenerating explanations for test samples...")
    generate_explanations(model, test_loader, explainer, attribute_names, explanation_dir, num_samples=10)

    return model, explainer, history


def plot_training_history(history, save_dir):
    """绘制训练历史"""
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 绘制损失
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # 绘制准确率
    axs[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axs[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].legend()

    # 绘制精确率、召回率和F1
    axs[1, 0].plot(history['val_precision'], label='Precision')
    axs[1, 0].plot(history['val_recall'], label='Recall')
    axs[1, 0].plot(history['val_f1'], label='F1 Score')
    axs[1, 0].set_title('Classification Metrics')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()

    # 绘制属性预测误差
    axs[1, 1].plot(history['val_attr_error'], label='Attr Prediction Error')
    axs[1, 1].set_title('Attribute Prediction Error')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Mean Absolute Error')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def generate_explanations(model, data_loader, explainer, attribute_names, save_dir, num_samples=10):
    """为测试样本生成可视化解释"""
    model.eval()

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 随机选择样本
    all_samples = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image']
            labels = batch['label']
            attributes = batch['attributes']

            for i in range(len(images)):
                all_samples.append({
                    'image': images[i],
                    'label': labels[i].item(),
                    'attributes': attributes[i]
                })

            if len(all_samples) >= num_samples * 5:  # 采集足够多的样本以便随机选择
                break

    # 随机选择样本
    selected_samples = np.random.choice(all_samples, num_samples, replace=False)

    device = next(model.parameters()).device

    # 为每个样本生成解释
    for i, sample in enumerate(selected_samples):
        image = sample['image'].unsqueeze(0).to(device)
        label = sample['label']
        attributes = sample['attributes']

        # 获取解释
        explanation = explainer.explain_prediction(
            image,
            true_label=label,
            true_attrs=attributes
        )

        # 可视化解释
        fig = explainer.visualize_explanation(
            explanation,
            save_path=os.path.join(save_dir, f'explanation_{i}.png')
        )
        plt.close(fig)

        # 保存解释数据
        with open(os.path.join(save_dir, f'explanation_{i}.txt'), 'w') as f:
            f.write(f"Prediction: {explanation['prediction']}\n")
            f.write(f"Confidence: {explanation['confidence']:.4f}\n")
            f.write(f"Malignancy Score: {explanation['malignancy_score']:.4f}\n")
            f.write(f"True Label: {explanation['true_label']}\n")
            f.write("\nAttribute Scores:\n")
            for attr, score in explanation['attribute_scores'].items():
                f.write(f"  {attr}: {score:.4f}\n")

            f.write("\nAttribute Importance:\n")
            for attr, imp in explanation['attribute_importance'].items():
                f.write(f"  {attr}: {imp:.4f}\n")

            f.write("\nTrue Attributes:\n")
            for attr, val in explanation['true_attributes'].items():
                f.write(f"  {attr}: {val:.4f}\n")

            f.write("\nAttribute Errors:\n")
            for attr, err in explanation['attribute_error'].items():
                f.write(f"  {attr}: {err:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HiFuseCapsNet model')
    parser.add_argument('--csv_file', type=str, default='nodule_dataset_normalized.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--model_type', type=str, default='small', choices=['mini', 'small'],
                        help='HiFuse base model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=128, help='Input image size')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--use_sampler', action='store_true',
                        help='Enable WeightedRandomSampler. Disabled by default to recover classification accuracy.')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Enable class-weighted classification loss. Disabled by default.')
    parser.add_argument('--lambda_attr', type=float, default=0.05, help='Weight for attribute regression loss')
    parser.add_argument('--lambda_concept_cls', type=float, default=0.05,
                        help='Weight for attribute-only concept classification loss')
    parser.add_argument('--lambda_consistency', type=float, default=0.02,
                        help='Weight for consistency between direct and concept classifiers')
    parser.add_argument('--lambda_complement', type=float, default=0.0,
                        help='Weight for complement concept classification loss')

    args = parser.parse_args()

    print(f"Using CSV file: {args.csv_file}")
    print(f"Model type: {args.model_type}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.image_size}")
    print(f"Saving models to: {args.save_dir}")
    print(f"Use sampler: {args.use_sampler}")
    print(f"Use class weights: {args.use_class_weights}")
    print(
        "Loss weights: "
        f"lambda_attr={args.lambda_attr}, "
        f"lambda_concept_cls={args.lambda_concept_cls}, "
        f"lambda_consistency={args.lambda_consistency}, "
        f"lambda_complement={args.lambda_complement}"
    )

    # 检查CSV文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found!")
        print("Please specify a valid CSV file with --csv_file option")
        sys.exit(1)

    # 训练模型
    model, explainer, history = train_model(
        csv_file=args.csv_file,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        save_dir=args.save_dir,
        use_sampler=args.use_sampler,
        use_class_weights=args.use_class_weights,
        lambda_attr=args.lambda_attr,
        lambda_concept_cls=args.lambda_concept_cls,
        lambda_consistency=args.lambda_consistency,
        lambda_complement=args.lambda_complement
    )

    print("Training completed!")

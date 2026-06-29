import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
# 导入自定义模块
from data_preprocessing import preprocess_nodule_dataset, normalize_attributes
from dataset_loader import NoduleAttributeDataset, get_data_loaders
from training_script import HiFuseCapsNet, AttrExplainer, train_model

# 导入HiFuse模型
# 注意：这里假设您已经有定义好的HiFuse模型
from hifuse_model import HiFuse_Mini, HiFuse_Small


def setup_environment():
    """设置环境和随机种子"""
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    return device


def preprocess_data(args):
    """预处理数据"""
    print("Preprocessing data...")

    # 检查输出CSV是否已存在
    if os.path.exists(args.output_csv):
        print(f"Found existing CSV file: {args.output_csv}")
        # 读取现有CSV
        import pandas as pd
        df = pd.read_csv(args.output_csv)
    else:
        # 处理数据集
        df = preprocess_nodule_dataset(args.data_dir, args.output_csv)
        # 归一化属性
        df = normalize_attributes(df)
        # 保存归一化后的数据
        df.to_csv(args.output_csv, index=False)

    # 打印数据分布
    print("\nData Distribution:")
    print(f"Total nodules: {len(df)}")
    if 'class' in df.columns:
        print(f"Benign nodules: {len(df[df['class'] == 0])}")
        print(f"Malignant nodules: {len(df[df['class'] == 1])}")

    # 检查哪些属性可用
    attribute_cols = [
        "subtlety", "internal_structure", "calcification",
        "sphericity", "margin", "lobulation", "spiculation", "texture"
    ]
    available_attrs = [col for col in attribute_cols if col in df.columns]

    print("\nAvailable attributes:")
    for attr in available_attrs:
        print(f"- {attr}")

    return df, available_attrs


def train_hifuse_capsnet(args, device):
    """训练HiFuseCapsNet模型"""
    print("\nTraining HiFuseCapsNet model...")

    # 训练模型
    model, explainer, history = train_model(
        csv_file=args.output_csv,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        save_dir=args.save_dir,
        device=device
    )

    return model, explainer, history


def evaluate_interpretability(model, explainer, test_loader, save_dir, device,attribute_names=None):
    """评估模型的可解释性"""
    print("\nEvaluating model interpretability...")

    if attribute_names is None:
        attribute_names = explainer.attribute_names

    # 创建保存目录
    interp_dir = os.path.join(save_dir, 'interpretability')
    os.makedirs(interp_dir, exist_ok=True)

    # 在测试集上计算属性预测准确性
    model.eval()
    attr_errors = []
    attr_importance = []
    correct_preds = []
    incorrect_preds = []


    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            attributes = batch['attributes'].to(device)

            outputs = model(images)

            # 计算属性预测误差
            attr_err = torch.abs(outputs['attribute_scores'] - attributes).cpu().numpy()
            attr_errors.append(attr_err)

            # 获取属性重要性
            attr_caps = outputs['attribute_capsules'].cpu().numpy()
            importance = np.sqrt((attr_caps ** 2).sum(axis=-1))
            importance = importance / importance.sum(axis=1, keepdims=True)
            attr_importance.append(importance)

            # 检查预测正确性
            _, predicted = outputs['class_probs'].max(1)
            correct = predicted.eq(labels)

            # 收集正确和错误的预测
            for i in range(len(images)):
                sample = {
                    'image': images[i].cpu(),
                    'label': labels[i].item(),
                    'pred': predicted[i].item(),
                    'attributes': attributes[i].cpu().numpy(),
                    'pred_attributes': outputs['attribute_scores'][i].cpu().numpy(),
                    'attr_importance': importance[i]
                }

                if correct[i]:
                    correct_preds.append(sample)
                else:
                    incorrect_preds.append(sample)

    # 计算平均属性误差
    attr_errors = np.concatenate(attr_errors, axis=0)
    mean_attr_errors = attr_errors.mean(axis=0)

    # 计算平均属性重要性
    attr_importance = np.concatenate(attr_importance, axis=0)
    mean_attr_importance = attr_importance.mean(axis=0)

    # 保存统计结果
    # attribute_names = test_loader.dataset.dataset.available_attributes

    with open(os.path.join(interp_dir, 'attribute_statistics.txt'), 'w') as f:
        f.write("Mean Attribute Prediction Error:\n")
        for i, attr in enumerate(attribute_names):
            f.write(f"{attr}: {mean_attr_errors[i]:.4f}\n")

        f.write("\nMean Attribute Importance:\n")
        for i, attr in enumerate(attribute_names):
            f.write(f"{attr}: {mean_attr_importance[i]:.4f}\n")

    # 可视化属性重要性
    plt.figure(figsize=(10, 6))
    plt.bar(attribute_names, mean_attr_importance)
    plt.title('Mean Attribute Importance')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(interp_dir, 'attribute_importance.png'))

    # attribute_names = test_loader.dataset.dataset.available_attributes
    # 可视化属性预测误差
    plt.figure(figsize=(10, 6))
    plt.bar(attribute_names, mean_attr_errors)
    plt.title('Mean Attribute Prediction Error')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(interp_dir, 'attribute_error.png'))

    # 分析属性对分类的影响
    print("\nAnalyzing attribute influence on classification...")

    # 随机选择一些正确和错误的预测样本进行可视化
    num_samples = min(5, len(correct_preds), len(incorrect_preds))

    # 正确预测的样本
    for i in range(num_samples):
        sample = correct_preds[i]
        image = sample['image'].unsqueeze(0).to(device)

        # 获取解释
        explanation = explainer.explain_prediction(
            image,
            true_label=sample['label'],
            true_attrs=sample['attributes']
        )

        # 可视化解释
        fig = explainer.visualize_explanation(
            explanation,
            save_path=os.path.join(interp_dir, f'correct_explanation_{i}.png')
        )
        plt.close(fig)

    # 错误预测的样本
    for i in range(num_samples):
        sample = incorrect_preds[i]
        image = sample['image'].unsqueeze(0).to(device)

        # 获取解释
        explanation = explainer.explain_prediction(
            image,
            true_label=sample['label'],
            true_attrs=sample['attributes']
        )

        # 可视化解释
        fig = explainer.visualize_explanation(
            explanation,
            save_path=os.path.join(interp_dir, f'incorrect_explanation_{i}.png')
        )
        plt.close(fig)

    # 属性扰动分析
    print("\nPerforming attribute perturbation analysis...")

    # 选择一个样本
    if len(correct_preds) > 0:
        sample = correct_preds[0]
    else:
        sample = incorrect_preds[0]

    image = sample['image'].unsqueeze(0).to(device)
    label = sample['label']
    attributes = torch.tensor(sample['attributes']).unsqueeze(0).to(device)

    # 原始预测
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        orig_probs = outputs['class_probs'][0].cpu().numpy()
        attribute_caps = outputs['attribute_capsules'][0].cpu().numpy()

    # 对每个属性进行扰动
    perturbation_results = []

    for i, attr_name in enumerate(attribute_names):
        # 屏蔽这个属性的胶囊
        with torch.no_grad():
            # 获取属性胶囊
            perturbed_caps = model.attribute_capsules(model.feature_extractor(image).view(1, -1, 1, 1))

            # 屏蔽特定属性
            perturbed_caps[:, i] = 0

            # 获取诊断结果
            diagnosis_caps = model.diagnosis_capsules(perturbed_caps)
            class_lengths = torch.sqrt((diagnosis_caps ** 2).sum(dim=-1))
            perturbed_probs = F.softmax(class_lengths, dim=-1)[0].cpu().numpy()

            # 计算概率变化
            prob_change = orig_probs - perturbed_probs

            perturbation_results.append({
                'attribute': attr_name,
                'original_probs': orig_probs.copy(),
                'perturbed_probs': perturbed_probs,
                'prob_change': prob_change
            })

    # 可视化扰动结果
    plt.figure(figsize=(12, 8))

    for i, result in enumerate(perturbation_results):
        # 计算对正确类别的概率影响
        correct_class_change = result['prob_change'][label]

        plt.bar(i, correct_class_change)

    plt.xticks(range(len(attribute_names)), attribute_names, rotation=45)
    plt.title('Effect of Removing Each Attribute on Classification Probability')
    plt.ylabel('Change in Correct Class Probability')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(interp_dir, 'attribute_perturbation.png'))

    print(f"Interpretability evaluation results saved to {interp_dir}")
    return interp_dir


def create_demo(model, explainer, test_loader, save_dir, device):
    """创建简单的解释演示"""
    print("\nCreating explanation demo...")

    # 创建保存目录
    demo_dir = os.path.join(save_dir, 'demo')
    os.makedirs(demo_dir, exist_ok=True)

    # 从测试集选择一些样本
    demo_samples = []

    # 确保同时包含良性和恶性样本
    benign_count = 0
    malignant_count = 0
    target_count = 3  # 每类样本的目标数量

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            attributes = batch['attributes'].to(device)

            for i in range(len(images)):
                if labels[i].item() == 0 and benign_count < target_count:
                    demo_samples.append({
                        'image': images[i].cpu(),
                        'label': labels[i].item(),
                        'attributes': attributes[i].cpu().numpy()
                    })
                    benign_count += 1
                elif labels[i].item() == 1 and malignant_count < target_count:
                    demo_samples.append({
                        'image': images[i].cpu(),
                        'label': labels[i].item(),
                        'attributes': attributes[i].cpu().numpy()
                    })
                    malignant_count += 1

            if benign_count >= target_count and malignant_count >= target_count:
                break

    # 为每个样本生成解释
    for i, sample in enumerate(demo_samples):
        image = sample['image'].unsqueeze(0).to(device)
        label = sample['label']
        attributes = sample['attributes']

        # 保存原始图像
        img_pil = T.ToPILImage()(sample['image'])
        img_pil.save(os.path.join(demo_dir, f'sample_{i}_image.png'))

        # 获取模型预测和解释
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            pred_class = outputs['class_probs'][0].argmax().item()
            pred_attrs = outputs['attribute_scores'][0].cpu().numpy()

            # 获取属性胶囊
            attr_caps = outputs['attribute_capsules'][0].cpu().numpy()

            # 计算属性重要性
            attr_importance = np.sqrt((attr_caps ** 2).sum(axis=1))
            attr_importance = attr_importance / attr_importance.sum()

        # 生成完整解释
        explanation = explainer.explain_prediction(
            image,
            true_label=label,
            true_attrs=attributes
        )

        # 可视化解释
        fig = explainer.visualize_explanation(
            explanation,
            save_path=os.path.join(demo_dir, f'sample_{i}_explanation.png')
        )
        plt.close(fig)

        # 保存属性信息
        attribute_names = explainer.attribute_names

        with open(os.path.join(demo_dir, f'sample_{i}_details.txt'), 'w') as f:
            f.write(f"Sample {i}\n")
            f.write(f"True Label: {'Malignant' if label == 1 else 'Benign'}\n")
            f.write(f"Predicted Label: {'Malignant' if pred_class == 1 else 'Benign'}\n")
            f.write(f"Prediction {'Correct' if pred_class == label else 'Incorrect'}\n\n")

            f.write("Attribute Values (True vs Predicted):\n")
            for j, attr in enumerate(attribute_names):
                f.write(f"{attr}: {attributes[j]:.2f} vs {pred_attrs[j]:.2f} (Importance: {attr_importance[j]:.2f})\n")

    print(f"Demo materials saved to {demo_dir}")
    return demo_dir


def main():
    parser = argparse.ArgumentParser(description='HiFuseCapsNet for Explainable Nodule Classification')
    parser.add_argument('--data_dir', type=str, default='/remote-home/cs_cs_xyh/classified_nodules',
                        help='Directory containing nodule data')
    parser.add_argument('--output_csv', type=str, default='nodule_dataset.csv',
                        help='Path to save the processed dataset CSV')
    parser.add_argument('--model_type', type=str, default='small', choices=['mini', 'small'],
                        help='HiFuse base model type')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=64, help='Input image size')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--skip_training', action='store_true',default=True, help='Skip training and load a saved model')
    parser.add_argument('--model_path', type=str, default='', help='Path to saved model (when skip_training is True)')

    args = parser.parse_args()

    # 设置环境
    device = setup_environment()

    # 预处理数据
    df, available_attrs = preprocess_data(args)

    # 获取数据加载器
    train_loader, val_loader, test_loader, attribute_names = get_data_loaders(
        args.output_csv,
        batch_size=args.batch_size,
        image_size=args.image_size
    )


    print(f"Available attributes: {attribute_names}")

    if args.skip_training and args.model_path:
        print(f"Loading saved model from {args.model_path}")

        # 创建基础模型
        if args.model_type.lower() == 'mini':
            base_model = HiFuse_Mini(num_classes=2, patch_size=2, window_size=2)
        else:
            base_model = HiFuse_Small(num_classes=2, patch_size=2, window_size=2)

        # 创建HiFuseCapsNet模型
        model = HiFuseCapsNet(base_model, len(attribute_names))

        # 加载保存的模型权重
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'],strict = False)
        model.to(device)

        # 创建解释器
        explainer = AttrExplainer(model, attribute_names)

    else:
        # 训练模型
        model, explainer, history = train_hifuse_capsnet(args, device)

    # 评估可解释性
    interp_dir = evaluate_interpretability(model, explainer, test_loader, args.save_dir, device, attribute_names)

    # 创建示例演示
    demo_dir = create_demo(model, explainer, test_loader, args.save_dir, device)

    print("\nProcess completed successfully!")
    print(f"Results saved to {args.save_dir}")
    print(f"Interpretability analysis: {interp_dir}")
    print(f"Explanation demo: {demo_dir}")


if __name__ == "__main__":
    main()

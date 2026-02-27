#!/usr/bin/env python
# 可视化MoE模型在实际数据上的特征分布

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from sklearn.manifold import TSNE
import seaborn as sns

from mmfewshot.detection.models import build_detector
from mmfewshot.detection.datasets import build_dataset, build_dataloader
from mmdet.core import bbox2roi


def parse_args():
    parser = argparse.ArgumentParser(
        description='可视化MoE模型在实际数据上的特征分布')
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('checkpoint', help='检查点文件路径')
    parser.add_argument(
        '--work-dir',
        help='保存可视化结果的目录路径')
    parser.add_argument(
        '--vis-class-names',
        type=str,
        nargs='+',
        help='要可视化的类别名称，如果不指定则可视化所有类别')
    parser.add_argument(
        '--num-samples-per-class',
        type=int,
        default=50,
        help='每个类别要提取的样本数量')
    parser.add_argument(
        '--vis-proto-feats',
        action='store_true',
        help='是否可视化原型与特征之间的关系')
    parser.add_argument(
        '--vis-gate-activations',
        action='store_true',
        help='是否可视化gate激活模式')
    parser.add_argument(
        '--fig-size',
        type=float,
        nargs='+',
        default=[12, 10],
        help='图像大小')
    parser.add_argument(
        '--dpi', type=int, default=300, help='DPI设置')
    parser.add_argument(
        '--seed', type=int, default=42, help='随机种子')
    parser.add_argument(
        '--perplexity', type=int, default=30, help='t-SNE的困惑度参数')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置文件中的一些设置')
    args = parser.parse_args()
    return args


def extract_roi_features(model, data_loader, max_samples_per_class=50):
    """提取验证集中的RoI特征。
    
    Args:
        model (nn.Module): 已加载的检测模型。
        data_loader (DataLoader): 数据加载器。
        max_samples_per_class (int): 每个类别最多提取的样本数。
        
    Returns:
        tuple: (features, labels, bbox_preds)，分别是特征、标签和边界框预测。
    """
    # 初始化结果容器
    all_features = []
    all_labels = []
    all_bbox_preds = []
    
    # 每个类别的样本计数
    class_counts = {}
    
    # 设置模型为评估模式
    model.eval()
    
    # 临时保存bbox_head的前向函数
    original_forward = model.roi_head.bbox_head.forward
    
    # 定义提取特征的新前向函数
    def forward_with_feature_extraction(x):
        """自定义前向函数，用于保存特征。"""
        # 使用原始forward函数获取结果
        cls_score, bbox_pred = original_forward(x)
        # 将归一化后的特征添加到结果中
        return cls_score, bbox_pred, F.normalize(x, dim=1)
    
    # 替换前向函数
    model.roi_head.bbox_head.forward = forward_with_feature_extraction
    
    # 提取特征
    with torch.no_grad():
        for data in mmcv.track_iter_progress(data_loader):
            # 准备数据
            img = data['img']
            img_metas = data['img_metas']
            gt_bboxes = data['gt_bboxes']
            gt_labels = data['gt_labels']
            
            # 前向传播获取特征图
            x = model.extract_feat(img)
            
            # 为每个样本处理
            for i in range(len(img_metas)):
                # 获取当前图像的标签和边界框
                img_gt_labels = gt_labels[i].to(x[0].device)
                img_gt_bboxes = gt_bboxes[i].to(x[0].device)
                
                # 如果没有标签，跳过
                if img_gt_labels.numel() == 0:
                    continue
                
                # 检查每个类别的样本数量
                current_labels = img_gt_labels.cpu().numpy()
                should_skip = False
                
                for label in np.unique(current_labels):
                    if label not in class_counts:
                        class_counts[label] = 0
                    if class_counts[label] >= max_samples_per_class:
                        should_skip = True
                        break
                
                if should_skip:
                    continue
                
                # 将GT边界框转换为RoI
                rois = bbox2roi([img_gt_bboxes])
                
                # 获取单个图像的特征图
                img_x = [feat[i:i+1] for feat in x]
                
                # 提取RoI特征
                roi_feats = model.roi_head.bbox_roi_extractor(
                    img_x[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
                
                # 获取分类分数、边界框预测和特征
                _, _, roi_norm_feats = model.roi_head.bbox_head(roi_feats)
                
                # 保存特征和标签
                all_features.append(roi_norm_feats.cpu())
                all_labels.append(img_gt_labels.cpu())
                
                # 更新类别计数
                for label in np.unique(current_labels):
                    count = np.sum(current_labels == label)
                    class_counts[label] += count
                
                # 检查是否所有类别都达到了最大样本数
                if all(count >= max_samples_per_class for count in class_counts.values()):
                    break
    
    # 恢复原始前向函数
    model.roi_head.bbox_head.forward = original_forward
    
    # 如果没有特征，返回空结果
    if not all_features:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # 合并所有特征和标签
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


def plot_tsne(features, labels, class_names, prototypes=None, num_experts=None, 
              save_path=None, figsize=(12, 10), dpi=300, perplexity=30):
    """使用t-SNE可视化特征分布。
    
    Args:
        features (torch.Tensor): 特征向量，形状为(N, D)。
        labels (torch.Tensor): 标签，形状为(N)。
        class_names (list): 类别名称列表。
        prototypes (torch.Tensor, 可选): 原型，形状为(C, K, D)。
        num_experts (int, 可选): 每个类别的专家数量。
        save_path (str, 可选): 保存图像的路径。
        figsize (tuple): 图像大小。
        dpi (int): 图像DPI。
        perplexity (int): t-SNE的困惑度参数。
    """
    # 转为numpy数组
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 准备t-SNE输入数据
    tsne_input = features_np
    
    # 如果有原型，加入到输入中
    proto_labels = None
    if prototypes is not None and num_experts is not None:
        proto_np = prototypes.cpu().detach().numpy()
        num_classes = proto_np.shape[0]
        flat_protos = proto_np.reshape(-1, proto_np.shape[2])
        tsne_input = np.vstack([features_np, flat_protos])
        
        # 生成原型标签（与对应类别相同但有特殊标记）
        proto_labels = np.array([i for i in range(num_classes) for _ in range(num_experts)])
        
        # 创建特征与原型标记的掩码
        is_proto = np.zeros(len(tsne_input), dtype=bool)
        is_proto[len(features_np):] = True
    
    # 使用t-SNE进行降维
    print(f"正在使用t-SNE对{len(tsne_input)}个样本进行降维...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed)
    result = tsne.fit_transform(tsne_input)
    
    # 分离特征和原型的t-SNE结果
    if prototypes is not None:
        feat_tsne = result[:len(features_np)]
        proto_tsne = result[len(features_np):]
    else:
        feat_tsne = result
    
    # 设置颜色
    unique_labels = np.unique(labels_np)
    num_classes_to_plot = len(unique_labels)
    palette = sns.color_palette("husl", num_classes_to_plot)
    
    # 绘制t-SNE图
    plt.figure(figsize=figsize, dpi=dpi)
    
    # 绘制特征点
    for i, label in enumerate(unique_labels):
        idx = (labels_np == label)
        plt.scatter(
            feat_tsne[idx, 0], 
            feat_tsne[idx, 1],
            label=class_names[label] if label < len(class_names) else f"类别 {label}",
            color=palette[i],
            marker='o',
            s=30,
            alpha=0.6
        )
    
    # 如果有原型，用星形标记绘制
    if prototypes is not None:
        for i, label in enumerate(unique_labels):
            if label >= len(class_names):
                continue
                
            # 找到对应类别的原型
            proto_idx = (proto_labels == label)
            if not np.any(proto_idx):
                continue
                
            plt.scatter(
                proto_tsne[proto_idx, 0], 
                proto_tsne[proto_idx, 1],
                label=f"{class_names[label]}原型",
                color=palette[i],
                marker='*',
                s=200,
                alpha=1.0,
                edgecolors='black'
            )
    
    # 添加图例和标题
    if prototypes is not None:
        plt.title(f"RoI特征与MoE原型的t-SNE可视化", fontsize=16)
    else:
        plt.title(f"RoI特征的t-SNE可视化", fontsize=16)
    
    plt.xlabel("t-SNE维度1", fontsize=14)
    plt.ylabel("t-SNE维度2", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"特征分布图已保存至 {save_path}")
    
    plt.close()


def visualize_gate_activations(model, features, labels, class_names, save_path=None,
                              figsize=(12, 10), dpi=300):
    """可视化不同样本在gate网络上的激活模式。
    
    Args:
        model (nn.Module): 包含gate网络的模型。
        features (torch.Tensor): 特征向量，形状为(N, D)。
        labels (torch.Tensor): 标签，形状为(N)。
        class_names (list): 类别名称列表。
        save_path (str, 可选): 保存图像的路径。
        figsize (tuple): 图像大小。
        dpi (int): 图像DPI。
    """
    # 确保模型处于评估模式
    model.eval()
    
    with torch.no_grad():
        # 将特征放到正确的设备上
        device = next(model.parameters()).device
        features = features.to(device)
        
        # 获取bbox_head
        bbox_head = model.roi_head.bbox_head
        
        # 获取gate激活值
        gate_weight = bbox_head.gate_net(features)  # [N, C*K]
        
        # 重塑成 [N, C, K]
        N = features.size(0)
        C = bbox_head.num_classes
        K = bbox_head.num_experts
        gate_weight = gate_weight.view(N, C, K)
        gate_weight = F.softmax(gate_weight, dim=-1)  # 对每个类别的K个专家进行softmax
        
        # 转为numpy
        gate_weight_np = gate_weight.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # 获取唯一的类别标签
        unique_labels = np.unique(labels_np)
        
        # 对每个类别可视化gate激活
        for class_id in unique_labels:
            if class_id >= len(class_names):
                class_name = f"类别 {class_id}"
            else:
                class_name = class_names[class_id]
                
            # 获取该类别的样本
            class_indices = (labels_np == class_id)
            
            if np.sum(class_indices) == 0:
                print(f"跳过类别 {class_name} - 没有样本")
                continue
                
            # 获取该类别的gate权重
            class_gate = gate_weight_np[class_indices, class_id, :]  # [n_samples, K]
            
            # 计算平均激活和标准差
            mean_activation = np.mean(class_gate, axis=0)
            std_activation = np.std(class_gate, axis=0)
            
            # 创建两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
            
            # 绘制热力图
            sns.heatmap(
                class_gate[:min(50, class_gate.shape[0])],  # 限制最多显示50个样本
                cmap="YlGnBu",
                xticklabels=[f"专家{j+1}" for j in range(K)],
                yticklabels=False,
                ax=ax1
            )
            ax1.set_title(f"{class_name}类的Gate激活模式", fontsize=16)
            ax1.set_xlabel("专家编号", fontsize=14)
            ax1.set_ylabel("样本", fontsize=14)
            
            # 绘制平均激活条形图
            ax2.bar(
                range(K), 
                mean_activation, 
                yerr=std_activation,
                alpha=0.7, 
                capsize=5
            )
            ax2.set_title(f"{class_name}类的平均Gate激活", fontsize=16)
            ax2.set_xlabel("专家编号", fontsize=14)
            ax2.set_ylabel("平均激活强度", fontsize=14)
            ax2.set_xticks(range(K))
            ax2.set_xticklabels([f"专家{j+1}" for j in range(K)])
            
            plt.tight_layout()
            
            if save_path:
                gate_save_path = save_path.replace('.png', f'_gate_{class_name}.png')
                plt.savefig(gate_save_path)
                print(f"{class_name}类的gate激活可视化已保存至 {gate_save_path}")
            
            plt.close()


def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 设置工作目录
    if args.work_dir is None:
        args.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(args.work_dir, exist_ok=True)
    
    # 构建模型
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 获取类别名称
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(cfg.data, 'val') and cfg.data.val.get('CLASSES', None) is not None:
        model.CLASSES = cfg.data.val.CLASSES
    elif hasattr(cfg.data, 'test') and cfg.data.test.get('CLASSES', None) is not None:
        model.CLASSES = cfg.data.test.CLASSES
    
    # 筛选指定类别
    class_names = model.CLASSES
    if args.vis_class_names:
        # 找出指定类别的索引
        selected_indices = [class_names.index(name) for name in args.vis_class_names if name in class_names]
        # 如果没有找到指定类别，则使用所有类别
        if not selected_indices:
            print(f"警告：未找到指定的类别 {args.vis_class_names}，将使用所有类别")
        else:
            class_names = [class_names[i] for i in selected_indices]
    
    # 将模型设置为评估模式并移动到GPU
    model.eval()
    model = MMDataParallel(model, device_ids=[0])
    
    # 构建数据集和数据加载器
    # 优先使用验证集，如果没有则使用测试集
    if hasattr(cfg.data, 'val'):
        dataset = build_dataset(cfg.data.val)
    else:
        dataset = build_dataset(cfg.data.test)
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    try:
        # 提取RoI特征
        print("正在提取RoI特征...")
        features, labels = extract_roi_features(
            model, data_loader, max_samples_per_class=args.num_samples_per_class)
        
        if features.numel() == 0:
            print("没有提取到有效特征，请检查数据集和模型")
            return
        
        print(f"成功提取到 {features.size(0)} 个特征向量")
        
        # 保存路径
        feat_tsne_path = os.path.join(args.work_dir, 'roi_features_tsne.png')
        feat_proto_tsne_path = os.path.join(args.work_dir, 'roi_features_with_prototypes_tsne.png')
        gate_vis_path = os.path.join(args.work_dir, 'gate_activations.png')
        
        # 获取原型
        bbox_head = model.module.roi_head.bbox_head
        prototypes = None
        num_experts = None
        
        if hasattr(bbox_head, 'prototypes'):
            prototypes = bbox_head.prototypes
            num_experts = bbox_head.num_experts
            print(f"成功获取到原型，形状：{prototypes.shape}")
        
        # 可视化特征分布
        plot_tsne(
            features, 
            labels, 
            class_names, 
            prototypes=None,  # 先不包含原型
            save_path=feat_tsne_path,
            figsize=tuple(args.fig_size),
            dpi=args.dpi,
            perplexity=args.perplexity
        )
        
        # 可视化特征与原型的分布关系
        if args.vis_proto_feats and prototypes is not None:
            plot_tsne(
                features, 
                labels, 
                class_names, 
                prototypes=prototypes,
                num_experts=num_experts,
                save_path=feat_proto_tsne_path,
                figsize=tuple(args.fig_size),
                dpi=args.dpi,
                perplexity=args.perplexity
            )
        
        # 可视化gate激活模式
        if args.vis_gate_activations and hasattr(bbox_head, 'gate_net'):
            visualize_gate_activations(
                model.module,
                features,
                labels,
                class_names,
                save_path=gate_vis_path,
                figsize=tuple(args.fig_size),
                dpi=args.dpi
            )
        
        print("可视化完成！")
        
    except Exception as e:
        print(f"可视化过程中出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
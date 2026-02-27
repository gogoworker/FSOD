#!/usr/bin/env python
# 可视化MoE原型的t-SNE分布

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import seaborn as sns

from mmfewshot.detection.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='可视化MoE原型的t-SNE分布')
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


def plot_tsne(prototypes, class_names, num_experts, save_path=None, 
              figsize=(12, 10), dpi=300, perplexity=30):
    """使用t-SNE可视化原型的分布。
    
    Args:
        prototypes (torch.Tensor): 形状为(num_classes, num_experts, feat_dim)的原型张量。
        class_names (list): 类别名称列表。
        num_experts (int): 每个类别的专家数量。
        save_path (str, 可选): 保存图像的路径。如果为None则不保存。
        figsize (tuple): 图像大小。
        dpi (int): 图像DPI。
        perplexity (int): t-SNE的困惑度参数。
    """
    # 转为numpy数组并展平类别和专家维度
    proto_np = prototypes.cpu().detach().numpy()
    num_classes = proto_np.shape[0]
    feat_dim = proto_np.shape[2]
    
    # 准备数据
    flat_protos = proto_np.reshape(-1, feat_dim)
    
    # 生成标签（每个类别的每个原型都用同一个标签）
    labels = np.array([i for i in range(num_classes) for _ in range(num_experts)])
    
    # 使用t-SNE进行降维
    print(f"正在使用t-SNE对{flat_protos.shape[0]}个原型进行降维...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed)
    proto_tsne = tsne.fit_transform(flat_protos)
    
    # 设置颜色
    palette = sns.color_palette("husl", num_classes)
    
    # 绘制t-SNE图
    plt.figure(figsize=figsize, dpi=dpi)
    
    # 为每个类别绘制所有专家
    for i in range(num_classes):
        class_indices = (labels == i)
        plt.scatter(
            proto_tsne[class_indices, 0], 
            proto_tsne[class_indices, 1],
            label=class_names[i],
            color=palette[i],
            marker='o',
            s=100,
            alpha=0.7
        )
        
        # 计算类中心并标注
        class_center = proto_tsne[class_indices].mean(axis=0)
        plt.annotate(
            class_names[i],
            xy=(class_center[0], class_center[1]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=12,
            fontweight='bold',
            ha='center'
        )
    
    # 添加图例和标题
    plt.title(f"MoE原型的t-SNE可视化 (每类{num_experts}个专家)", fontsize=16)
    plt.xlabel("t-SNE维度1", fontsize=14)
    plt.ylabel("t-SNE维度2", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存至 {save_path}")
    
    plt.close()


def plot_inter_prototype_distances(prototypes, class_names, save_path=None, 
                                  figsize=(12, 10), dpi=300):
    """可视化原型间的余弦相似度。
    
    Args:
        prototypes (torch.Tensor): 形状为(num_classes, num_experts, feat_dim)的原型张量。
        class_names (list): 类别名称列表。
        save_path (str, 可选): 保存图像的路径。如果为None则不保存。
        figsize (tuple): 图像大小。
        dpi (int): 图像DPI。
    """
    # 归一化原型
    normed_protos = F.normalize(prototypes, dim=-1)
    num_classes, num_experts, _ = normed_protos.shape
    
    # 准备计算类间余弦相似度
    class_avg_protos = []
    
    # 对每个类别计算平均原型
    for i in range(num_classes):
        class_avg = normed_protos[i].mean(dim=0, keepdim=True)  # (1, feat_dim)
        class_avg = F.normalize(class_avg, dim=-1)
        class_avg_protos.append(class_avg)
    
    # 将列表转为张量 (num_classes, 1, feat_dim)
    class_avg_protos = torch.cat(class_avg_protos, dim=0)
    
    # 计算类间余弦相似度
    sim_matrix = torch.mm(
        class_avg_protos.squeeze(1), 
        class_avg_protos.squeeze(1).t()
    )
    
    # 绘制热力图
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        sim_matrix.cpu().numpy(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("类别间原型余弦相似度", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"类别间相似度矩阵已保存至 {save_path}")
    
    plt.close()
    
    # 计算并绘制类内专家间的余弦相似度
    for i in range(num_classes):
        class_protos = normed_protos[i]  # (num_experts, feat_dim)
        intra_sim = torch.mm(class_protos, class_protos.t())
        
        plt.figure(figsize=(8, 6), dpi=dpi)
        sns.heatmap(
            intra_sim.cpu().numpy(),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=[f"Expert {j+1}" for j in range(num_experts)],
            yticklabels=[f"Expert {j+1}" for j in range(num_experts)]
        )
        plt.title(f"{class_names[i]}类内专家间余弦相似度", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            intra_save_path = save_path.replace('.png', f'_intra_{class_names[i]}.png')
            plt.savefig(intra_save_path)
            print(f"{class_names[i]}类内专家相似度矩阵已保存至 {intra_save_path}")
        
        plt.close()


def visualize_gate_activations(model, features, labels, class_names, save_path=None,
                               figsize=(12, 10), dpi=300):
    """可视化不同样本在gate网络上的激活模式。
    
    Args:
        model (nn.Module): 包含gate网络的模型。
        features (torch.Tensor): 用于测试的特征，形状为(N, feat_dim)。
        labels (torch.Tensor): 特征对应的标签，形状为(N,)。
        class_names (list): 类别名称列表。
        save_path (str, 可选): 保存图像的路径。如果为None则不保存。
        figsize (tuple): 图像大小。
        dpi (int): 图像DPI。
    """
    # 确保模型处于评估模式
    model.eval()
    
    with torch.no_grad():
        # 获取bbox_head (MoEPrototypeHead)
        bbox_head = model.roi_head.bbox_head
        
        # 获取gate激活值
        roi_feats = F.normalize(features, dim=-1)
        gate_weight = bbox_head.gate_net(roi_feats)  # [N, C*K]
        
        # 重塑成 [N, C, K]
        N = roi_feats.size(0)
        C = bbox_head.num_classes
        K = bbox_head.num_experts
        gate_weight = gate_weight.view(N, C, K)
        gate_weight = F.softmax(gate_weight, dim=-1)  # 对每个类别的K个专家进行softmax
        
        # 转为numpy
        gate_weight_np = gate_weight.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # 对每个类别可视化gate激活
        for class_id in range(C):
            # 获取该类别的样本
            class_indices = (labels_np == class_id)
            
            if np.sum(class_indices) == 0:
                print(f"跳过类别 {class_names[class_id]} - 没有样本")
                continue
                
            # 获取该类别的gate权重
            class_gate = gate_weight_np[class_indices, class_id, :]  # [n_samples, K]
            
            # 绘制激活热力图
            plt.figure(figsize=figsize, dpi=dpi)
            sns.heatmap(
                class_gate,
                cmap="YlGnBu",
                xticklabels=[f"Expert {j+1}" for j in range(K)],
                yticklabels=False
            )
            plt.title(f"{class_names[class_id]}类的Gate激活模式", fontsize=16)
            plt.xlabel("专家编号", fontsize=14)
            plt.ylabel("样本", fontsize=14)
            plt.tight_layout()
            
            if save_path:
                gate_save_path = save_path.replace('.png', f'_gate_{class_names[class_id]}.png')
                plt.savefig(gate_save_path)
                print(f"{class_names[class_id]}类的gate激活热力图已保存至 {gate_save_path}")
            
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
    
    # 构建模型
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 获取类别名称
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif cfg.data.train.get('CLASSES', None) is not None:
        model.CLASSES = cfg.data.train.CLASSES
    
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
    
    # 将模型设置为评估模式
    model.eval()
    
    # 获取原型
    try:
        # 确保能够获取到原型，检查模型结构
        bbox_head = model.roi_head.bbox_head
        if not hasattr(bbox_head, 'prototypes'):
            raise AttributeError("模型不包含原型属性，请确认是否使用了MoE原型头")
        
        prototypes = bbox_head.prototypes
        num_experts = bbox_head.num_experts
        print(f"成功获取到原型，形状：{prototypes.shape}")
        
        # 保存路径
        tsne_save_path = os.path.join(args.work_dir, 'moe_prototypes_tsne.png')
        sim_save_path = os.path.join(args.work_dir, 'moe_prototypes_similarity.png')
        
        # 绘制t-SNE可视化
        plot_tsne(
            prototypes, 
            class_names, 
            num_experts,
            save_path=tsne_save_path,
            figsize=tuple(args.fig_size),
            dpi=args.dpi,
            perplexity=args.perplexity
        )
        
        # 绘制原型间的相似度
        plot_inter_prototype_distances(
            prototypes,
            class_names,
            save_path=sim_save_path,
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
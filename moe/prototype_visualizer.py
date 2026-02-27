import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


class PrototypeVisualizer:
    """原型分布可视化工具类"""
    
    def __init__(self, save_dir: str = "prototype_vis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_prototype_distribution(self, 
                                       prototypes: torch.Tensor,
                                       class_names: Optional[List[str]] = None,
                                       epoch: int = 0,
                                       method: str = 'tsne') -> Dict[str, str]:
        """可视化原型分布
        
        Args:
            prototypes: 原型张量，形状为 [num_classes, num_experts, feat_dim]
            class_names: 类别名称列表
            epoch: 当前epoch
            method: 降维方法，'tsne' 或 'pca'
            
        Returns:
            Dict[str, str]: 保存的文件路径字典
        """
        saved_files = {}
        
        # 归一化原型
        protos = F.normalize(prototypes, dim=-1)
        num_classes, num_experts, feat_dim = protos.shape
        
        # 1. 原型相似度热力图
        similarity_heatmap_path = self._plot_similarity_heatmap(
            protos, class_names, epoch)
        saved_files['similarity_heatmap'] = similarity_heatmap_path
        
        # 2. 类内专家相似度分布
        intra_similarity_path = self._plot_intra_class_similarity(
            protos, class_names, epoch)
        saved_files['intra_similarity'] = intra_similarity_path
        
        # 3. 类间专家相似度分布
        inter_similarity_path = self._plot_inter_class_similarity(
            protos, class_names, epoch)
        saved_files['inter_similarity'] = inter_similarity_path
        
        # 4. 降维可视化
        if method == 'tsne':
            dim_reduction_path = self._plot_tsne_visualization(
                protos, class_names, epoch)
        else:
            dim_reduction_path = self._plot_pca_visualization(
                protos, class_names, epoch)
        saved_files['dim_reduction'] = dim_reduction_path
        
        # 5. 原型统计信息
        stats_path = self._plot_prototype_statistics(
            protos, class_names, epoch)
        saved_files['statistics'] = stats_path
        
        return saved_files
    
    def _plot_similarity_heatmap(self, 
                                protos: torch.Tensor,
                                class_names: Optional[List[str]] = None,
                                epoch: int = 0) -> str:
        """绘制原型相似度热力图"""
        plt.figure(figsize=(12, 10))
        
        # 计算所有原型间的相似度
        protos_flat = protos.view(-1, protos.shape[-1])  # [C*K, D]
        similarity_matrix = torch.mm(protos_flat, protos_flat.T)
        
        # 转换为numpy
        sim_np = similarity_matrix.detach().cpu().numpy()
        
        # 创建标签
        num_classes, num_experts = protos.shape[:2]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        labels = []
        for i in range(num_classes):
            for j in range(num_experts):
                labels.append(f'{class_names[i]}_E{j}')
        
        # 绘制热力图
        sns.heatmap(sim_np, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   cmap='viridis',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(f'Prototype Similarity Matrix (Epoch {epoch})')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'similarity_heatmap_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_intra_class_similarity(self,
                                    protos: torch.Tensor,
                                    class_names: Optional[List[str]] = None,
                                    epoch: int = 0) -> str:
        """绘制类内专家相似度分布"""
        plt.figure(figsize=(15, 8))
        
        num_classes, num_experts = protos.shape[:2]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        # 计算每个类内的专家相似度
        intra_similarities = []
        class_labels = []
        
        for c in range(num_classes):
            pc = protos[c]  # [K, D]
            cos_sim = torch.mm(pc, pc.T)  # [K, K]
            
            # 获取上三角矩阵（排除对角线）
            upper_tri = torch.triu(cos_sim, diagonal=1)
            valid_sims = upper_tri[upper_tri != 0]
            
            if len(valid_sims) > 0:
                intra_similarities.extend(valid_sims.detach().cpu().numpy())
                class_labels.extend([class_names[c]] * len(valid_sims))
        
        # 绘制箱线图
        df = pd.DataFrame({
            'Similarity': intra_similarities,
            'Class': class_labels
        })
        
        sns.boxplot(data=df, x='Class', y='Similarity')
        plt.title(f'Intra-class Expert Similarity Distribution (Epoch {epoch})')
        plt.xticks(rotation=45)
        plt.ylabel('Cosine Similarity')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'intra_similarity_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_inter_class_similarity(self,
                                   protos: torch.Tensor,
                                   class_names: Optional[List[str]] = None,
                                   epoch: int = 0) -> str:
        """绘制类间专家相似度分布"""
        plt.figure(figsize=(15, 8))
        
        num_classes, num_experts = protos.shape[:2]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        # 计算类间专家相似度
        inter_similarities = []
        pair_labels = []
        
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                pi = protos[i]  # [K, D]
                pj = protos[j]  # [K, D]
                
                # 计算所有专家间的相似度
                inter_cos = torch.mm(pi, pj.T)  # [K, K]
                sims = inter_cos.flatten().detach().cpu().numpy()
                
                inter_similarities.extend(sims)
                pair_labels.extend([f'{class_names[i]}-{class_names[j]}'] * len(sims))
        
        # 绘制箱线图
        df = pd.DataFrame({
            'Similarity': inter_similarities,
            'Class Pair': pair_labels
        })
        
        sns.boxplot(data=df, x='Class Pair', y='Similarity')
        plt.title(f'Inter-class Expert Similarity Distribution (Epoch {epoch})')
        plt.xticks(rotation=45)
        plt.ylabel('Cosine Similarity')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'inter_similarity_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_tsne_visualization(self,
                                protos: torch.Tensor,
                                class_names: Optional[List[str]] = None,
                                epoch: int = 0) -> str:
        """使用t-SNE可视化原型分布"""
        plt.figure(figsize=(12, 10))
        
        # 展平原型
        protos_flat = protos.view(-1, protos.shape[-1])  # [C*K, D]
        protos_np = protos_flat.detach().cpu().numpy()
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(protos_np)-1))
        protos_2d = tsne.fit_transform(protos_np)
        
        # 绘制散点图
        num_classes, num_experts = protos.shape[:2]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        
        for c in range(num_classes):
            start_idx = c * num_experts
            end_idx = (c + 1) * num_experts
            
            plt.scatter(protos_2d[start_idx:end_idx, 0], 
                       protos_2d[start_idx:end_idx, 1],
                       c=[colors[c]], 
                       label=class_names[c],
                       s=100, alpha=0.7)
            
            # 添加专家编号
            for k in range(num_experts):
                plt.annotate(f'E{k}', 
                           (protos_2d[start_idx + k, 0], protos_2d[start_idx + k, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
        
        plt.title(f'Prototype Distribution (t-SNE) - Epoch {epoch}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'tsne_visualization_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_pca_visualization(self,
                               protos: torch.Tensor,
                               class_names: Optional[List[str]] = None,
                               epoch: int = 0) -> str:
        """使用PCA可视化原型分布"""
        plt.figure(figsize=(12, 10))
        
        # 展平原型
        protos_flat = protos.view(-1, protos.shape[-1])  # [C*K, D]
        protos_np = protos_flat.detach().cpu().numpy()
        
        # PCA降维
        pca = PCA(n_components=2)
        protos_2d = pca.fit_transform(protos_np)
        
        # 绘制散点图
        num_classes, num_experts = protos.shape[:2]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        
        for c in range(num_classes):
            start_idx = c * num_experts
            end_idx = (c + 1) * num_experts
            
            plt.scatter(protos_2d[start_idx:end_idx, 0], 
                       protos_2d[start_idx:end_idx, 1],
                       c=[colors[c]], 
                       label=class_names[c],
                       s=100, alpha=0.7)
            
            # 添加专家编号
            for k in range(num_experts):
                plt.annotate(f'E{k}', 
                           (protos_2d[start_idx + k, 0], protos_2d[start_idx + k, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
        
        plt.title(f'Prototype Distribution (PCA) - Epoch {epoch}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'pca_visualization_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_prototype_statistics(self,
                                  protos: torch.Tensor,
                                  class_names: Optional[List[str]] = None,
                                  epoch: int = 0) -> str:
        """绘制原型统计信息"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        num_classes, num_experts = protos.shape[:2]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        # 1. 原型范数分布
        proto_norms = torch.norm(protos, dim=-1)  # [C, K]
        axes[0, 0].hist(proto_norms.flatten().detach().cpu().numpy(), bins=20, alpha=0.7)
        axes[0, 0].set_title('Prototype Norm Distribution')
        axes[0, 0].set_xlabel('Norm')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 每个类别的专家相似度均值
        intra_sim_means = []
        for c in range(num_classes):
            pc = protos[c]  # [K, D]
            cos_sim = torch.mm(pc, pc.T)  # [K, K]
            upper_tri = torch.triu(cos_sim, diagonal=1)
            valid_sims = upper_tri[upper_tri != 0]
            if len(valid_sims) > 0:
                intra_sim_means.append(valid_sims.mean().item())
            else:
                intra_sim_means.append(0.0)
        
        axes[0, 1].bar(range(num_classes), intra_sim_means)
        axes[0, 1].set_title('Average Intra-class Similarity')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Average Similarity')
        axes[0, 1].set_xticks(range(num_classes))
        axes[0, 1].set_xticklabels(class_names, rotation=45)
        
        # 3. 类间相似度矩阵
        inter_sim_matrix = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    pi = protos[i]  # [K, D]
                    pj = protos[j]  # [K, D]
                    inter_cos = torch.mm(pi, pj.T)  # [K, K]
                    inter_sim_matrix[i, j] = inter_cos.mean()
        
        im = axes[1, 0].imshow(inter_sim_matrix.detach().cpu().numpy(), 
                               cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Inter-class Similarity Matrix')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Class')
        axes[1, 0].set_xticks(range(num_classes))
        axes[1, 0].set_yticks(range(num_classes))
        axes[1, 0].set_xticklabels(class_names, rotation=45)
        axes[1, 0].set_yticklabels(class_names)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. 专家数量分布（每个类别的专家激活情况）
        expert_activation = torch.norm(protos, dim=-1)  # [C, K]
        expert_activation = (expert_activation > 0.1).float()  # 简单的激活阈值
        activation_counts = expert_activation.sum(dim=1)  # [C]
        
        axes[1, 1].bar(range(num_classes), activation_counts.detach().cpu().numpy())
        axes[1, 1].set_title('Active Experts per Class')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Number of Active Experts')
        axes[1, 1].set_xticks(range(num_classes))
        axes[1, 1].set_xticklabels(class_names, rotation=45)
        
        plt.suptitle(f'Prototype Statistics (Epoch {epoch})')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'prototype_statistics_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_gate_distribution(self,
                                  gate_weights: torch.Tensor,
                                  class_names: Optional[List[str]] = None,
                                  epoch: int = 0) -> str:
        """可视化门控权重分布
        
        Args:
            gate_weights: 门控权重，形状为 [N, C, K]
            class_names: 类别名称列表
            epoch: 当前epoch
            
        Returns:
            str: 保存的文件路径
        """
        plt.figure(figsize=(15, 10))
        
        N, C, K = gate_weights.shape
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(C)]
        
        # 计算每个类别的平均门控权重
        mean_gate_weights = gate_weights.mean(dim=0)  # [C, K]
        
        # 绘制热力图
        sns.heatmap(mean_gate_weights.detach().cpu().numpy(),
                   xticklabels=[f'E{k}' for k in range(K)],
                   yticklabels=class_names,
                   cmap='viridis',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Average Gate Weight'})
        
        plt.title(f'Average Gate Weight Distribution (Epoch {epoch})')
        plt.xlabel('Expert')
        plt.ylabel('Class')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'gate_distribution_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path 
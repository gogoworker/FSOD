# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.bbox_heads import MetaBBoxHead
from torch import Tensor
from mmdet.models.losses import accuracy
from typing import Optional, Dict, List
import os
import numpy as np

class TopKGating(nn.Module):
    """Top-K Gating机制，选择K个专家"""
    
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 k: int = 1, 
                 noisy_gating: bool = True,
                 expert_capacity: Optional[int] = None):
        super().__init__()
        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        self.k = k
        self.noisy_gating = noisy_gating
        self.expert_capacity = expert_capacity or num_experts
        
    def forward(self, x: torch.Tensor):
        """计算门控输出和专家权重"""
        batch_size = x.shape[0]
        
        # 基础门控分数
        clean_gate_logits = self.w_gate(x)
        
        if self.noisy_gating and self.training:
            # 添加噪声以增加探索性
            noise = torch.randn_like(clean_gate_logits) * 0.1
            gate_logits = clean_gate_logits + noise
        else:
            gate_logits = clean_gate_logits
            
        # 计算softmax得到门控概率
        gates = F.softmax(gate_logits, dim=1)
        
        # 选择top-k专家
        indices_sorted = torch.argsort(gates, dim=1, descending=True)
        top_k_indices = indices_sorted[:, :self.k]
        top_k_gates = torch.gather(gates, 1, top_k_indices)
        
        # 专家容量限制
        if self.expert_capacity < batch_size:
            # 实现专家容量限制逻辑
            pass
            
        return top_k_indices, top_k_gates


@HEADS.register_module()
class HybridMoEHead(MetaBBoxHead):
    """混合头，结合Meta-RCNN原始分类器和MoE原型分类器。
    
    Args:
        num_classes (int): 类别数量。
        in_channels (int): 输入特征通道数。
        num_experts (int): 每个类别的专家数量。
        feat_dim (int): RoI特征维度。
        use_bias (bool): 是否在gate网络中使用偏置。
        prototype_init_std (float): 原型初始化的标准差。
        ema_decay (float): 原型EMA更新的衰减因子。
        moe_weight (float): MoE分类器的权重，范围[0,1]。
        loss_proto_diversity_weight (float): 原型多样性损失权重。
        loss_proto_inter_weight (float): 原型类间距离损失权重。
    """

    def __init__(self,
                 num_classes=80,
                 in_channels=2048,
                 num_experts=4,
                 feat_dim=2048,
                 use_bias=False,
                 prototype_init_std=0.01,
                 ema_decay=0.99,
                 moe_weight=0.5,
                 use_proto_regularization=True,
                 loss_proto_diversity_weight=0.2,
                 loss_proto_inter_weight=0.5,
                 enable_visualization=False,
                 vis_save_dir="prototype_vis",
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            **kwargs)
        
        self.feat_dim = feat_dim
        self.num_experts = num_experts
        self.moe_weight = moe_weight
        self.use_proto_regularization = use_proto_regularization
        self.loss_proto_diversity_weight = loss_proto_diversity_weight
        self.loss_proto_inter_weight = loss_proto_inter_weight
        
        # 可视化相关
        self.enable_visualization = enable_visualization
        self.vis_save_dir = vis_save_dir
        if self.enable_visualization:
            from .prototype_visualizer import PrototypeVisualizer
            self.visualizer = PrototypeVisualizer(save_dir=vis_save_dir)

        
        # 原型存储：C x K x D，每个类别有K个专家原型
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, num_experts, feat_dim) * prototype_init_std,
            requires_grad=True
        )
        # gate 网络：输入 D → 输出 C x K 分配权重
        self.gate_net = nn.Linear(feat_dim, num_classes * num_experts, bias=use_bias)
        
        # EMA 更新相关
        # self.register_buffer('ema_decay', torch.tensor(ema_decay))
        # self.register_buffer('initialized', torch.tensor(0))
    
    def forward(self, x):
        """前向函数。
        
        Args:
            x (Tensor): 形状为 (num_rois, in_channels)。
            
        Returns:
            tuple: 分类分数和边界框预测的元组。
                cls_score (Tensor): 所有类别的分类分数。
                bbox_pred (Tensor): 边界框回归增量。
        """
        # 原始Meta-RCNN分类
        meta_cls_score = self.fc_cls(x)
        
        # MoE原型分类
        moe_cls_score = self.forward_moe_cls(x)
        
        # 融合两种分类结果
        cls_score = (1 - self.moe_weight) * meta_cls_score + self.moe_weight * moe_cls_score
        
        # 回归分支保持不变
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        
        return cls_score, bbox_pred
    
    def forward_moe_cls(self, roi_feats):
        """MoE分类前向函数。
        
        Args:
            roi_feats (Tensor): RoI特征，形状为 (N, D)。
            
        Returns:
            Tensor: 分类分数，形状为 (N, num_classes + 1)。
        """
        N, D = roi_feats.size()
        C, K = self.num_classes, self.num_experts
        
        # 特征归一化
        roi_feats = F.normalize(roi_feats, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)  # [C, K, D]
        
        # gate 分配：输出权重 [N, C*K] → reshape [N, C, K]
        gate_weight = self.gate_net(roi_feats)  # [N, C*K]
        gate_weight = gate_weight.view(N, C, K)
        gate_weight = F.softmax(gate_weight, dim=-1)  # 对每个类别的K个专家进行soft assign
        
        # 计算与原型的余弦相似度: [N, D] · [C, K, D] → [N, C, K]
        sim = torch.einsum('nd,ckd->nck', roi_feats, proto)
        
        # 加权融合多个专家的相似度：∑k sim * gate_weight
        logits = (gate_weight * sim).sum(dim=-1)  # [N, C]
        
        # 添加背景类得分（与所有原型的最小相似度）
        bg_score = -torch.max(sim, dim=2)[0].mean(dim=1, keepdim=True)  # [N, 1]
        cls_score = torch.cat([logits, bg_score], dim=1)  # [N, C+1]
        
        # 保存门控权重用于可视化
        if self.enable_visualization:
            self.last_gate_weights = gate_weight.detach()
        
        return cls_score
    
    def compute_prototype_loss(self):
        """计算原型多样性和类间距离正则化损失。
        
        Returns:
            dict: 包含原型多样性和类间距离损失的字典。
        """
        # 归一化原型 [C, K, D]
        protos = F.normalize(self.prototypes, dim=-1)
        
        # 类内多样性：鼓励同类原型方向不同
        intra_loss = 0.
        for c in range(self.num_classes):
            pc = protos[c]  # [K, D]
            cos_sim = torch.mm(pc, pc.T)  # [K, K]
            # 排除对角线（自身相似度）
            mask = torch.ones_like(cos_sim) - torch.eye(self.num_experts, device=cos_sim.device)
            # 计算平均相似度，越小越好
            intra_loss += (cos_sim * mask).sum() / (self.num_experts * (self.num_experts - 1))
        intra_loss = intra_loss / self.num_classes
        
        # 类间分离性：鼓励不同类原型方向远离
        inter_loss = 0.
        num_pairs = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                pi = protos[i]  # [K, D]
                pj = protos[j]  # [K, D]
                # 计算所有专家间的相似度
                inter_cos = torch.mm(pi, pj.T)  # [K, K]
                inter_loss += inter_cos.mean()
                num_pairs += 1
        
        if num_pairs > 0:
            inter_loss = inter_loss / num_pairs
        
        # 最终损失（多样性损失越小越好，类间距离损失越大越好）
        loss_div = self.loss_proto_diversity_weight * intra_loss
        loss_inter = self.loss_proto_inter_weight * (-inter_loss)  # 负号使得类间距离越大越好
        
        return {'loss_proto_diversity': loss_div, 'loss_proto_inter': loss_inter}
    
    def visualize_prototypes(self, 
                           epoch: int = 0,
                           class_names: Optional[List[str]] = None,
                           method: str = 'tsne') -> Dict[str, str]:
        """可视化原型分布
        
        Args:
            epoch: 当前epoch
            class_names: 类别名称列表
            method: 降维方法，'tsne' 或 'pca'
            
        Returns:
            Dict[str, str]: 保存的文件路径字典
        """
        if not self.enable_visualization:
            return {}
        
        # 可视化原型分布
        saved_files = self.visualizer.visualize_prototype_distribution(
            self.prototypes, class_names, epoch, method)
        
        # 可视化门控权重分布
        if hasattr(self, 'last_gate_weights'):
            gate_vis_path = self.visualizer.visualize_gate_distribution(
                self.last_gate_weights, class_names, epoch)
            saved_files['gate_distribution'] = gate_vis_path
        
        return saved_files
    
    def get_prototype_statistics(self) -> Dict[str, float]:
        """获取原型统计信息
        
        Returns:
            Dict[str, float]: 包含各种统计指标的字典
        """
        with torch.no_grad():
            protos = F.normalize(self.prototypes, dim=-1)
            
            # 计算类内多样性
            intra_similarities = []
            for c in range(self.num_classes):
                pc = protos[c]  # [K, D]
                cos_sim = torch.mm(pc, pc.T)  # [K, K]
                upper_tri = torch.triu(cos_sim, diagonal=1)
                valid_sims = upper_tri[upper_tri != 0]
                if len(valid_sims) > 0:
                    intra_similarities.extend(valid_sims.cpu().numpy())
            
            # 计算类间分离性
            inter_similarities = []
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    pi = protos[i]  # [K, D]
                    pj = protos[j]  # [K, D]
                    inter_cos = torch.mm(pi, pj.T)  # [K, K]
                    inter_similarities.extend(inter_cos.flatten().cpu().numpy())
            
            # 计算原型范数
            proto_norms = torch.norm(protos, dim=-1)  # [C, K]
            
            stats = {
                'avg_intra_similarity': np.mean(intra_similarities) if intra_similarities else 0.0,
                'avg_inter_similarity': np.mean(inter_similarities) if inter_similarities else 0.0,
                'intra_diversity': 1.0 - np.mean(intra_similarities) if intra_similarities else 0.0,
                'inter_separation': 1.0 - np.mean(inter_similarities) if inter_similarities else 0.0,
                'avg_proto_norm': proto_norms.mean().item(),
                'std_proto_norm': proto_norms.std().item(),
                'min_proto_norm': proto_norms.min().item(),
                'max_proto_norm': proto_norms.max().item(),
            }
            
            return stats
    

    
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets,
             bbox_weights, reduction_override=None):
        """计算损失。
        
        Args:
            cls_score (Tensor): 分类分数。
            bbox_pred (Tensor): 边界框预测。
            rois (Tensor): RoIs。
            labels (Tensor): 标签。
            label_weights (Tensor): 标签权重。
            bbox_targets (Tensor): 边界框目标。
            bbox_weights (Tensor): 边界框权重。
            reduction_override (str | None): 覆盖损失的reduction方式。
            
        Returns:
            dict: 包含损失组件的字典。
        """
        # 调用父类的损失计算
        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights,
                             bbox_targets, bbox_weights, reduction_override)
        
        # 添加原型相关损失
        if self.use_proto_regularization:
            proto_losses = self.compute_prototype_loss()
            losses.update(proto_losses)
        
        return losses
    
    def update_prototypes_with_support(self, support_feats, support_labels):
        """使用支持集特征更新原型。
        
        Args:
            support_feats (Tensor): 支持集特征，形状为 (N, D)。
            support_labels (Tensor): 支持集标签，形状为 (N)。
        """
        with torch.no_grad():
            for cls in range(self.num_classes):
                mask = support_labels == cls
                if mask.sum() == 0:
                    continue
                
                cls_feats = support_feats[mask]  # [M, D]
                if cls_feats.size(0) > 0:
                    new_proto = cls_feats.mean(dim=0, keepdim=True)  # [1, D]
                    new_proto = F.normalize(new_proto, dim=-1)  # unit norm
                    new_proto = new_proto.expand(self.num_experts, -1)  # [K, D]
                    
                    if self.initialized == 0:
                        self.prototypes.data[cls] = new_proto
                    else:
                        self.prototypes.data[cls] = (
                            self.ema_decay * self.prototypes.data[cls] + 
                            (1 - self.ema_decay) * new_proto
                        )
            
            self.initialized += 1
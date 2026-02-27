# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.bbox_heads import MetaBBoxHead
from torch import Tensor
from mmdet.models.losses import accuracy
from typing import Optional


@HEADS.register_module()
class GaussianMixtureHead(MetaBBoxHead):
    """基于高斯混合模型的头部网络，用于少样本目标检测。
    
    每个类别被建模为多个高斯分布的混合，每个高斯分布有自己的均值和协方差矩阵。
    这种方法可以更好地捕捉类内多样性和类间差异。
    
    Args:
        num_classes (int): 类别数量。
        in_channels (int): 输入特征通道数。
        num_components (int): 每个类别的高斯分布数量。
        feat_dim (int): RoI特征维度。
        use_bias (bool): 是否在gate网络中使用偏置。
        prototype_init_std (float): 原型初始化的标准差。
        cov_init_value (float): 协方差矩阵初始值。
        ema_decay (float): 原型EMA更新的衰减因子。
        meta_weight (float): Meta分类器的权重，范围[0,1]。
        use_proto_regularization (bool): 是否使用原型正则化。
        loss_proto_diversity_weight (float): 原型多样性损失权重。
        loss_proto_inter_weight (float): 原型类间距离损失权重。
    """

    def __init__(self,
                 num_classes=80,
                 in_channels=2048,
                 num_components=2,  # 根据论文，使用2个组件效果最佳
                 feat_dim=2048,
                 use_bias=False,
                 prototype_init_std=0.01,
                 cov_init_value=0.1,
                 ema_decay=0.99,
                 meta_weight=0.0,  # 完全依赖GMM分类器
                 use_proto_regularization=True,
                 loss_proto_diversity_weight=0.2,
                 loss_proto_inter_weight=1.0,
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            **kwargs)
        
        self.feat_dim = feat_dim
        self.num_components = num_components
        self.meta_weight = meta_weight
        self.use_proto_regularization = use_proto_regularization
        self.loss_proto_diversity_weight = loss_proto_diversity_weight
        self.loss_proto_inter_weight = loss_proto_inter_weight
        
        # 添加特征适配器，用于处理输入特征维度与GMM期望维度不匹配的情况
        if in_channels != feat_dim:
            self.feature_adapter = nn.Linear(in_channels, feat_dim)
        else:
            self.feature_adapter = None

        # 高斯混合模型参数
        # 均值：C x K x D，每个类别有K个高斯分布，每个分布有D维均值
        self.means = nn.Parameter(
            torch.randn(num_classes, num_components, feat_dim) * prototype_init_std,
            requires_grad=True
        )
        
        # 协方差矩阵：使用对角协方差矩阵简化计算
        # 使用log参数化确保协方差为正
        self.log_covs = nn.Parameter(
            torch.ones(num_classes, num_components, feat_dim) * math.log(cov_init_value),
            requires_grad=True
        )
        
        # 混合权重：C x K，每个类别有K个高斯分布的权重
        # 使用log参数化确保权重为正，并在前向传播中归一化
        self.log_weights = nn.Parameter(
            torch.zeros(num_classes, num_components),
            requires_grad=True
        )
        
        # EMA 更新相关
        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('initialized', torch.tensor(0))
    
    def forward(self, x, support_feat=None):
        """前向函数。
        
        Args:
            x (Tensor): 形状为 (num_rois, in_channels)。
            support_feat (Tensor, optional): 支持集特征，形状为 (1, in_channels)。
                用于Meta-RCNN分类器。
                
        Returns:
            tuple: 分类分数和边界框预测的元组。
                cls_score (Tensor): 所有类别的分类分数。
                bbox_pred (Tensor): 边界框回归增量。
        """
        # 使用特征适配器处理特征
        if self.feature_adapter is not None:
            adapted_x = self.feature_adapter(x)
        else:
            adapted_x = x
            
        # 高斯混合模型分类
        gmm_cls_score = self.forward_gmm_cls(adapted_x)
        
        # 原始Meta-RCNN分类
        if self.meta_weight > 0 and support_feat is not None:
            meta_cls_score = self.forward_meta_cls(x, support_feat)
        else:
            # 如果不使用meta分类器或没有提供支持集特征，使用普通分类器
            meta_cls_score = self.fc_cls(x)
        
        # 融合两种分类结果
        cls_score = (1 - self.meta_weight) * gmm_cls_score + self.meta_weight * meta_cls_score
        
        # 回归分支保持不变
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        
        return cls_score, bbox_pred
    
    def forward_gmm_cls(self, roi_feats):
        """高斯混合模型分类前向函数。
        
        计算特征属于每个类别GMM的概率。
        
        Args:
            roi_feats (Tensor): RoI特征，形状为 (N, D)。
            
        Returns:
            Tensor: 分类分数，形状为 (N, num_classes + 1)。
        """
        N, D = roi_feats.size()
        C = self.num_classes
        
        # 特征归一化
        roi_feats = F.normalize(roi_feats, dim=-1)
        means = F.normalize(self.means, dim=-1)  # [C, K, D]
        
        # 获取协方差矩阵和混合权重
        covs = torch.exp(self.log_covs)  # [C, K, D]
        weights = F.softmax(self.log_weights, dim=1)  # [C, K]
        
        # 计算每个样本属于每个类别的对数似然
        logits = torch.zeros(N, C, device=roi_feats.device)
        
        for c in range(C):
            # 计算样本在类别c的GMM中的对数似然
            log_likelihood = self._compute_gmm_log_likelihood(roi_feats, means[c], covs[c], weights[c])
            logits[:, c] = log_likelihood
        
        # 添加背景类得分
        # 背景类得分定义为所有类别GMM似然的负对数几何平均值
        bg_score = -torch.mean(logits, dim=1, keepdim=True)
        
        # 将类别得分和背景得分拼接
        cls_score = torch.cat([logits, bg_score], dim=1)  # [N, C+1]
        
        return cls_score
    
    def _compute_gmm_log_likelihood(self, x, means, covs, weights):
        """计算样本在高斯混合模型中的对数似然。
        
        Args:
            x (Tensor): 样本点，形状为 [N, D]
            means (Tensor): 高斯分布的均值，形状为 [K, D]
            covs (Tensor): 高斯分布的协方差矩阵对角线，形状为 [K, D]
            weights (Tensor): 混合权重，形状为 [K]
            
        Returns:
            Tensor: 每个样本在GMM中的对数似然，形状为 [N]
        """
        N, D = x.shape
        K = means.shape[0]
        
        # 计算每个样本到每个高斯分布的马氏距离
        # 扩展维度以便广播
        x_expanded = x.unsqueeze(1)  # [N, 1, D]
        means_expanded = means.unsqueeze(0)  # [1, K, D]
        covs_expanded = covs.unsqueeze(0)  # [1, K, D]
        
        # 计算马氏距离的平方：(x - μ)^T Σ^-1 (x - μ)
        # 对于对角协方差矩阵，这等价于 sum((x - μ)^2 / σ^2)
        mahalanobis_dist = ((x_expanded - means_expanded) ** 2 / covs_expanded).sum(dim=2)  # [N, K]
        
        # 计算每个高斯分布的对数似然
        # log N(x|μ,Σ) = -0.5 * [D*log(2π) + log|Σ| + (x-μ)^T Σ^-1 (x-μ)]
        log_det_cov = torch.sum(torch.log(covs), dim=1)  # [K]
        log_gaussian = -0.5 * (D * math.log(2 * math.pi) + log_det_cov.unsqueeze(0) + mahalanobis_dist)  # [N, K]
        
        # 添加混合权重
        log_weighted = log_gaussian + torch.log(weights.unsqueeze(0))  # [N, K]
        
        # 计算混合模型的对数似然：log(sum_k π_k N(x|μ_k,Σ_k))
        # 使用log-sum-exp技巧避免数值问题
        max_log_weighted = torch.max(log_weighted, dim=1, keepdim=True)[0]  # [N, 1]
        log_sum_exp = torch.log(torch.sum(torch.exp(log_weighted - max_log_weighted), dim=1, keepdim=True))  # [N, 1]
        log_likelihood = (max_log_weighted + log_sum_exp).squeeze(1)  # [N]
        return log_likelihood
    
    def compute_prototype_loss(self):
        """计算原型类间距离正则化损失。
        
        使用高斯混合模型表示每个类别，并使用KL散度衡量不同类别之间的距离。
        根据论文，只使用KL散度来衡量类间距离，不需要额外的类内多样性损失。
        
        Returns:
            dict: 包含原型类间距离损失的字典。
        """
        # 计算所有类别对之间的KL散度
        inter_kl_loss = self._compute_pairwise_kl_divergence()
        
        # 使用1/(1+KL)作为损失函数，KL越大，损失越小
        # 这样可以鼓励不同类别之间的分布差异更大
        loss_inter = self.loss_proto_inter_weight / (1 + inter_kl_loss)
        
        return {'loss_proto_inter': loss_inter}
    
    def _compute_pairwise_kl_divergence(self):
        """计算所有类别对之间的KL散度。
        
        根据论文方法，计算不同类别高斯混合模型之间的KL散度，
        用于衡量类间距离并促进不同类别特征的分离。
        
        Returns:
            Tensor: 平均KL散度
        """
        # 获取参数
        means = F.normalize(self.means, dim=-1)  # [C, K, D]
        covs = torch.exp(self.log_covs)  # [C, K, D]
        weights = F.softmax(self.log_weights, dim=1)  # [C, K]
        
        C = self.num_classes
        total_kl = 0.0
        num_pairs = 0
        
        # 计算所有类别对之间的KL散度
        for i in range(C):
            for j in range(i + 1, C):
                # 计算类别i和j之间的KL散度
                kl_ij = self._compute_gmm_kl_divergence_efficient(
                    means[i], means[j], covs[i], covs[j], weights[i], weights[j])
                
                # 由于KL散度不是对称的，我们计算双向KL散度并取平均
                kl_ji = self._compute_gmm_kl_divergence_efficient(
                    means[j], means[i], covs[j], covs[i], weights[j], weights[i])
                
                # 使用对称KL散度
                sym_kl = (kl_ij + kl_ji) / 2.0
                total_kl += sym_kl
                num_pairs += 1
        
        if num_pairs > 0:
            return total_kl / num_pairs
        return torch.tensor(0.0, device=means.device)
    
    def _compute_gmm_kl_divergence_efficient(self, means_i, means_j, covs_i, covs_j, weights_i, weights_j):
        """高效计算两个GMM之间的KL散度近似值。
        
        使用论文中提出的方法，通过直接计算组件间的KL散度并加权求和。
        
        Args:
            means_i (Tensor): 第一个GMM的均值 [K, D]
            means_j (Tensor): 第二个GMM的均值 [K, D]
            covs_i (Tensor): 第一个GMM的协方差 [K, D]
            covs_j (Tensor): 第二个GMM的协方差 [K, D]
            weights_i (Tensor): 第一个GMM的混合权重 [K]
            weights_j (Tensor): 第二个GMM的混合权重 [K]
            
        Returns:
            Tensor: 两个GMM之间的KL散度近似值
        """
        K_i = means_i.size(0)
        K_j = means_j.size(0)
        D = means_i.size(1)
        
        # 计算每对组件之间的KL散度
        kl_components = torch.zeros((K_i, K_j), device=means_i.device)
        
        for ki in range(K_i):
            for kj in range(K_j):
                # 计算均值差异
                mean_diff = means_i[ki] - means_j[kj]  # [D]
                
                # 计算协方差比率项
                cov_ratio = covs_j[kj] / covs_i[ki]  # [D]
                
                # 计算均值差异项
                mean_term = torch.sum((mean_diff ** 2) / covs_j[kj])
                
                # 计算协方差项
                cov_term = torch.sum(cov_ratio - 1.0 - torch.log(cov_ratio))
                
                # 组合KL散度
                kl_components[ki, kj] = 0.5 * (mean_term + cov_term)
        
        # 使用权重计算加权KL散度
        weighted_kl = torch.sum(weights_i.unsqueeze(1) * weights_j.unsqueeze(0) * kl_components)
        
        return weighted_kl
    
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
        
        根据论文中的方法，使用支持集特征更新高斯混合模型的参数，
        包括均值、协方差和混合权重。
        
        Args:
            support_feats (Tensor): 支持集特征，形状为 (N, D)。
            support_labels (Tensor): 支持集标签，形状为 (N)。
        """
        # 使用特征适配器处理支持集特征
        if self.feature_adapter is not None:
            support_feats = self.feature_adapter(support_feats)
            
        with torch.no_grad():
            for cls in range(self.num_classes):
                mask = support_labels == cls
                if mask.sum() == 0:
                    continue
                
                cls_feats = support_feats[mask]  # [M, D]
                if cls_feats.size(0) > 0:
                    # 归一化特征
                    cls_feats = F.normalize(cls_feats, dim=-1)
                    
                    # 使用K-means聚类更新均值
                    if cls_feats.size(0) >= self.num_components:
                        # 如果样本数足够，执行简化的K-means
                        # 1. 随机初始化聚类中心
                        indices = torch.randperm(cls_feats.size(0))[:self.num_components]
                        centroids = cls_feats[indices]  # [K, D]
                        
                        # 2. 执行几轮K-means迭代
                        for _ in range(3):  # 3轮迭代通常足够
                            # 计算每个样本到每个中心的距离
                            dists = torch.cdist(cls_feats, centroids)  # [M, K]
                            # 分配样本到最近的中心
                            assignments = torch.argmin(dists, dim=1)  # [M]
                            # 更新中心
                            new_centroids = centroids.clone()
                            for k in range(self.num_components):
                                cluster_samples = cls_feats[assignments == k]
                                if cluster_samples.size(0) > 0:
                                    new_centroids[k] = cluster_samples.mean(dim=0)
                            # 更新中心
                            centroids = new_centroids
                        
                        # 归一化中心
                        new_means = F.normalize(centroids, dim=-1)
                        
                        # 计算每个聚类的协方差
                        new_covs = torch.ones_like(self.log_covs[cls]) * math.log(0.1)  # 默认值
                        new_weights = torch.ones(self.num_components, device=cls_feats.device) / self.num_components
                        
                        for k in range(self.num_components):
                            cluster_samples = cls_feats[assignments == k]
                            if cluster_samples.size(0) > 0:
                                # 计算样本到中心的偏差
                                deviations = cluster_samples - new_means[k].unsqueeze(0)  # [Mk, D]
                                # 计算协方差（对角部分）
                                cluster_cov = torch.mean(deviations ** 2, dim=0)  # [D]
                                # 防止协方差过小
                                cluster_cov = torch.clamp(cluster_cov, min=0.01)
                                new_covs[k] = torch.log(cluster_cov)
                                # 更新权重为聚类中样本的比例
                                new_weights[k] = cluster_samples.size(0) / cls_feats.size(0)
                    else:
                        # 如果样本数不足，复制样本
                        new_means = cls_feats.repeat(self.num_components // cls_feats.size(0) + 1, 1)
                        new_means = new_means[:self.num_components]
                        new_means = F.normalize(new_means, dim=-1)
                        # 使用默认协方差和均匀权重
                        new_covs = torch.ones_like(self.log_covs[cls]) * math.log(0.1)
                        new_weights = torch.ones(self.num_components, device=cls_feats.device) / self.num_components
                    
                    # 更新模型参数
                    if self.initialized == 0:
                        self.means.data[cls] = new_means
                        self.log_covs.data[cls] = new_covs
                        self.log_weights.data[cls] = torch.log(new_weights)
                    else:
                        # 使用EMA更新
                        self.means.data[cls] = (
                            self.ema_decay * self.means.data[cls] + 
                            (1 - self.ema_decay) * new_means
                        )
                        self.log_covs.data[cls] = (
                            self.ema_decay * self.log_covs.data[cls] + 
                            (1 - self.ema_decay) * new_covs
                        )
                        self.log_weights.data[cls] = (
                            self.ema_decay * self.log_weights.data[cls] + 
                            (1 - self.ema_decay) * torch.log(new_weights)
                        )
            
            self.initialized += 1
    
    def visualize_gaussian_mixture(self, save_path=None):
        """可视化每个类别的高斯混合模型分布。
        
        使用t-SNE将高维特征降到2D进行可视化，显示每个类别的高斯分布均值和协方差椭圆。
        
        Args:
            save_path (str, optional): 保存可视化结果的路径。如果为None，则不保存。
            
        Returns:
            tuple: 包含图像和坐标的元组，可用于进一步分析。
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # 获取均值和协方差
        means = F.normalize(self.means, dim=-1).detach().cpu().numpy()
        C, K, D = means.shape
        
        covs = torch.exp(self.log_covs).detach().cpu().numpy()
        weights = F.softmax(self.log_weights, dim=1).detach().cpu().numpy()
        
        # 将所有均值展平为 [C*K, D]
        flat_means = means.reshape(-1, D)
        
        # 使用t-SNE将高维特征降到2D
        tsne = TSNE(n_components=2, random_state=42)
        means_2d = tsne.fit_transform(flat_means)
        
        # 重新整形为 [C, K, 2]
        means_2d = means_2d.reshape(C, K, 2)
        
        # 创建图像
        plt.figure(figsize=(12, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, C))
        
        for c in range(C):
            # 绘制类别c的高斯分布均值
            for k in range(K):
                plt.scatter(means_2d[c, k, 0], means_2d[c, k, 1], 
                           color=colors[c], label=f'Class {c}' if k == 0 else None, 
                           s=100 * weights[c, k] + 50, marker='o', alpha=0.8)
                
                # 为每个高斯分布绘制协方差椭圆
                # 在2D空间中，我们需要将协方差矩阵投影
                # 这里我们简化为使用t-SNE后的距离比例
                cov_scale = np.mean(covs[c, k]) * 0.1 * (weights[c, k] * 2 + 0.5)
                
                # 创建椭圆
                circle = plt.Circle(
                    (means_2d[c, k, 0], means_2d[c, k, 1]),
                    cov_scale,
                    fill=False,
                    color=colors[c],
                    linestyle='--',
                    alpha=0.5
                )
                plt.gca().add_patch(circle)
        
        plt.legend(loc='upper right')
        plt.title('Gaussian Mixture Model Visualization (t-SNE)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
        return plt.gcf(), means_2d 

    def forward_meta_cls(self, roi_feat, support_feat):
        """Meta-RCNN分类前向函数。
        
        Args:
            roi_feat (Tensor): ROI特征，形状为 (num_rois, in_channels)。
            support_feat (Tensor): 支持集特征，形状为 (1, in_channels)。
            
        Returns:
            Tensor: 分类分数，形状为 (num_rois, num_classes + 1)。
        """
        # 复用原始MetaBBoxHead中的元分类器逻辑
        if support_feat is not None:
            input_meta = roi_feat
            support_feat = support_feat.reshape(-1, self.in_channels)
            # 创建分类权重和偏置
            fc_cls_weight = self.fc_cls.weight.data
            fc_cls_bias = self.fc_cls.bias.data
            # 使用支持集特征进行分类
            class_logits = self.meta_cls_forward(input_meta, support_feat, 
                                               fc_cls_weight, fc_cls_bias)
            return class_logits
        else:
            # 若无支持集特征，回退到标准分类器
            return self.fc_cls(roi_feat) 
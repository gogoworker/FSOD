# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.bbox_heads import MetaBBoxHead
from torch import Tensor
from mmdet.models.losses import accuracy


@HEADS.register_module()
class MoEPrototypeHead(MetaBBoxHead):
    """BBoxHead with Multiple Expert Prototypes for few-shot object detection.
    
    Each class has multiple prototypes (experts), and each RoI feature is 
    assigned to these prototypes with a gate network.
    
    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        num_experts (int): Number of expert prototypes per class.
        feat_dim (int): Dimension of RoI feature.
        use_bias (bool): Whether to use bias in the gate network.
        prototype_init_std (float): Standard deviation for prototype initialization.
        ema_decay (float): Decay factor for EMA update of prototypes.
    """

    def __init__(self,
                 num_classes=80,
                 in_channels=2048,
                 num_experts=4,
                 feat_dim=2048,
                 use_bias=False,
                 prototype_init_std=0.01,
                 ema_decay=0.99,
                 with_meta_cls_loss=False,
                 **kwargs):
        kwargs.update(dict(
            num_classes=num_classes,
            in_channels=in_channels,
            with_meta_cls_loss=with_meta_cls_loss,
            num_meta_classes=num_classes  # 在kwargs中设置num_meta_classes
        ))
        super().__init__(**kwargs)
        
        self.feat_dim = feat_dim
        self.num_experts = num_experts
        
        # 原型存储：C x K x D，每个类别有K个专家原型
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, num_experts, feat_dim) * prototype_init_std,
            requires_grad=True
        )
        # gate 网络：输入 D → 输出 C x K 分配权重
        self.gate_net = nn.Linear(feat_dim, num_classes * num_experts, bias=use_bias)
        
        # EMA 更新相关
        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('initialized', torch.tensor(0))
        
        # 删除原始的分类器，使用原型分类器替代
        if hasattr(self, 'fc_cls'):
            self.fc_cls = None
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Shape of (num_rois, in_channels).
            
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                cls_score (Tensor): Classification scores for all classes.
                bbox_pred (Tensor): Box regression deltas.
        """
        # 使用原型进行分类
        cls_score = self.forward_cls(x)
        
        # 回归分支保持不变
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        
        return cls_score, bbox_pred
    
    def forward_cls(self, roi_feats):
        """Forward function for classification.
        
        Args:
            roi_feats (Tensor): RoI features with shape (N, D).
            
        Returns:
            Tensor: Classification scores with shape (N, num_classes + 1).
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
        
        return cls_score
    
    def update_prototypes_with_support(self, support_feats, support_labels):
        """Update prototypes with support features using EMA.
        
        Args:
            support_feats (Tensor): Support features with shape (N, D).
            support_labels (Tensor): Support labels with shape (N).
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
    
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function for MoEPrototypeHead.
        
        Args:
            cls_score (Tensor): Classification scores.
            bbox_pred (Tensor): BBox predictions.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Labels of proposals.
            label_weights (Tensor): Label weights of proposals.
            bbox_targets (Tensor): BBox regression targets.
            bbox_weights (Tensor): BBox regression loss weights.
            reduction_override (str, optional): Method to reduce losses.
                Default: None.
                
        Returns:
            dict: Computed losses.
                - loss_cls (Tensor): Classification loss.
                - acc (Tensor): Accuracy.
                - loss_bbox (Tensor): BBox regression loss.
        """
        losses = dict()
        
        # 分类损失
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                losses['acc'] = accuracy(cls_score, labels)
        
        # 回归损失
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        
        # 禁用meta分类损失
        if self.with_meta_cls_loss:
            losses['loss_meta_cls'] = torch.tensor(0.0, device=cls_score.device)
            losses['meta_acc'] = torch.tensor(100.0, device=cls_score.device)
        
        return losses 
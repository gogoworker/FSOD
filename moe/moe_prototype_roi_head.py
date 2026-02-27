# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead


@HEADS.register_module()
class MoEPrototypeRoIHead(MetaRCNNRoIHead):
    """Roi head for MoE Prototype-based few-shot object detection.
    
    This ROI head extends MetaRCNNRoIHead to work with MoEPrototypeHead.
    It provides functionality to update prototypes with support features.
    """

    def forward_train(self,
                      query_feats,
                      support_feats,
                      proposals,
                      query_img_metas,
                      query_gt_bboxes,
                      query_gt_labels,
                      support_gt_labels,
                      query_gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function for training.
        
        Args:
            query_feats (list[Tensor]): List of query features.
            support_feats (list[Tensor]): List of support features.
            proposals (list[Tensor]): List of region proposals.
            query_img_metas (list[dict]): List of query image info dict.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for query images.
            query_gt_labels (list[Tensor]): Ground truth labels for query images.
            support_gt_labels (list[Tensor]): Ground truth labels for support images.
            query_gt_bboxes_ignore (list[Tensor] | None): Bboxes to be ignored.
                Default: None.
                
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 提取支持集特征用于更新原型
        support_feat = self.extract_support_feats(support_feats)[0]
        
        # 更新原型（如果bbox_head支持）
        if hasattr(self.bbox_head, 'update_prototypes_with_support'):
            support_labels = torch.cat(support_gt_labels)
            self.bbox_head.update_prototypes_with_support(support_feat, support_labels)
        
        # 调用父类的forward_train进行常规训练
        losses = super().forward_train(
            query_feats, support_feats, proposals,
            query_img_metas, query_gt_bboxes, query_gt_labels,
            support_gt_labels, query_gt_bboxes_ignore, **kwargs
        )
        
        return losses
    
    def _bbox_forward(self, query_roi_feats, support_roi_feats):
        """Box head forward function used in both training and testing.
        
        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).
            
        Returns:
            dict: A dictionary of predicted results.
        """
        # 特征聚合
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        
        # 使用bbox_head进行分类和回归
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1))
        
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results 
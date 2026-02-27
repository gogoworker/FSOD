import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS, build_neck
from mmfewshot.detection.models.roi_heads import MetaRCNNRoIHead
from torch import Tensor

@HEADS.register_module()
class ProtoFusionRoIHead(MetaRCNNRoIHead):
    """RoI head that implements prototype feature fusion and momentum update.
    
    Args:
        momentum (float): The momentum coefficient for prototype updates.
            Default: 0.999.
        temperature (float): Temperature parameter for feature fusion.
            Default: 0.1.
        aggregation_layer (ConfigDict): Config of aggregation layer.
            Default: None.
        proto_weight (float): Weight of prototype similarity in classification.
            Default: 0.5.
    """
    def __init__(self,
                 momentum: float = 0.999,
                 temperature: float = 0.1, 
                 aggregation_layer: Optional[ConfigDict] = None,
                 is_base_training=True,
                 proto_weight: float = 0.5,
                 **kwargs) -> None:
        super().__init__(aggregation_layer=aggregation_layer, **kwargs)
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer('base_prototypes', None)
        self.is_base_training = is_base_training  # 标记当前是否为基类训练阶段
        self.proto_weight = proto_weight  # 原型相似度在分类中的权重
        
    @torch.no_grad()
    def _update_base_prototype(self, features: Tensor, labels: Tensor):
        """Update base class prototypes."""
        # Initialize base prototypes if not exist
        if self.base_prototypes is None:
            self.base_prototypes = torch.zeros(
                self.bbox_head.num_classes, features.shape[1],
                device=features.device)
            
        for label in torch.unique(labels):
            # Skip background class
            if label == self.bbox_head.num_classes:
                continue
            # Get features for current class
            class_feats = features[labels == label]
            class_proto = class_feats.mean(0) # 计算平均特征作为原型
            
            if self.is_base_training:
                # 基类训练阶段: 直接更新
                self.base_prototypes[label] = class_proto
            else:
                # 新类训练阶段: 动量更新
                if self.base_prototypes[label].sum() != 0:  # 只更新已有的基类原型
                    self.base_prototypes[label] = (
                        self.momentum * self.base_prototypes[label] + 
                        (1 - self.momentum) * class_proto)
                
    def _fuse_novel_prototype(self, novel_feats: Tensor, novel_labels: Tensor):
        """Fuse base and novel class prototypes."""
        # Get novel class prototypes
        novel_protos = []
        novel_labels_unique = []
        for label in torch.unique(novel_labels):
            if label == self.bbox_head.num_classes:
                continue
            class_feats = novel_feats[novel_labels == label]
            novel_protos.append(class_feats.mean(0))
            novel_labels_unique.append(label)
            
        if not novel_protos:  # 如果没有新类样本
            return None, None
            
        novel_protos = torch.stack(novel_protos)
        novel_labels_unique = torch.tensor(novel_labels_unique, 
                                         device=novel_protos.device)
        
        # Calculate attention between base and novel prototypes
        attention = torch.matmul(
            novel_protos, self.base_prototypes.t()) / self.temperature
        attention = F.softmax(attention, dim=1)
        
        # Fuse prototypes with attention
        fused_protos = novel_protos + torch.matmul(attention, self.base_prototypes)
        return fused_protos, novel_labels_unique

    def _bbox_forward(self, query_roi_feats: Tensor,
                     support_roi_feats: Tensor = None) -> Dict:
        """Box head forward function used in both training and testing."""
        # 获取原始分类结果
        cls_score_orig, bbox_pred = self.bbox_head(query_roi_feats)
        
        # 如果有原型且不是基类训练阶段，计算原型相似度
        if self.base_prototypes is not None and not self.is_base_training:
            # 计算query特征与所有原型的相似度
            similarity = F.cosine_similarity(
                query_roi_feats.unsqueeze(1),
                self.base_prototypes.unsqueeze(0),
                dim=2
            )
            # 缩放相似度
            proto_score = similarity / self.temperature
            
            # 处理背景类 - 为原型相似度添加背景类得分列
            # 使用较低的背景类得分，或者从原始分类头获取背景类得分
            if cls_score_orig.size(1) > proto_score.size(1):
                # 获取原始分类头的背景类得分
                bg_score = cls_score_orig[:, -1:] 
                # 将背景类得分添加到原型相似度中
                proto_score = torch.cat([proto_score, bg_score], dim=1)
            
            # 确保两个张量形状相同
            assert cls_score_orig.size() == proto_score.size(), \
                f"Shape mismatch: {cls_score_orig.size()} vs {proto_score.size()}"
                
            # 融合原始分类结果和原型相似度
            cls_score = (1 - self.proto_weight) * cls_score_orig + self.proto_weight * proto_score
        else:
            cls_score = cls_score_orig
            
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _bbox_forward_train(self, query_feats: List[Tensor],
                           support_feats: List[Tensor],
                           sampling_results: object,
                           query_img_metas: List[Dict],
                           query_gt_bboxes: List[Tensor],
                           query_gt_labels: List[Tensor],
                           support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.
        
        Args:
            query_feats (list[Tensor]): List of query features.
            support_feats (list[Tensor]): List of support features.
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes.
            query_gt_labels (list[Tensor]): Ground truth labels.
            support_gt_labels (list[Tensor]): Support set labels.
            
        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        
        # 获取分类和回归结果
        bbox_results = self._bbox_forward(query_roi_feats)
        
        # 计算损失
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, query_gt_bboxes, query_gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'], bbox_results['bbox_pred'], 
            query_rois, *bbox_targets)
            
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def forward_train(self,
                     query_feats: List[Tensor],
                     support_feats: List[Tensor],
                     proposals: List[Tensor],
                     query_img_metas: List[Dict],
                     query_gt_bboxes: List[Tensor],
                     query_gt_labels: List[Tensor],
                     support_gt_labels: List[Tensor],
                     query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                     **kwargs) -> Dict:
        """Forward function for training.
        
        Args:
            query_feats (list[Tensor]): List of query features.
            support_feats (list[Tensor]): List of support features.
            proposals (list[Tensor]): List of region proposals.
            query_img_metas (list[dict]): List of query image info.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes.
            query_gt_labels (list[Tensor]): Ground truth labels.
            support_gt_labels (list[Tensor]): Support set labels.
            query_gt_bboxes_ignore (list[Tensor] | None): GT bboxes to ignore.

        Returns:
            dict[str, Tensor]: Dict of loss components.
        """
        # Extract ROI features
        sampling_results = []
        num_imgs = len(query_img_metas)
        if query_gt_bboxes_ignore is None:
            query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposals[i], query_gt_bboxes[i],
                query_gt_bboxes_ignore[i], query_gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposals[i],
                query_gt_bboxes[i],
                query_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in query_feats])
            sampling_results.append(sampling_result)

        # Get query ROI features
        rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        
        # Get support features and update prototypes
        support_feat = self.extract_support_feats(support_feats)[0]
        self._update_base_prototype(support_feat, torch.cat(support_gt_labels))
        
        if not self.is_base_training:
            # 新类训练阶段: 融合新类原型
            novel_protos, novel_labels = self._fuse_novel_prototype(
                support_feat, torch.cat(support_gt_labels))
            if novel_protos is not None:
                # 更新分类头中的原型
                for proto, label in zip(novel_protos, novel_labels):
                    self.base_prototypes[label] = proto
        
        # Calculate losses
        bbox_results = self._bbox_forward_train(
            query_feats, support_feats, sampling_results,
            query_img_metas, query_gt_bboxes, query_gt_labels,
            support_gt_labels)
            
        losses = dict()
        losses.update(bbox_results['loss_bbox'])
        return losses
        
    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            proposal_list: List[Tensor],
            query_img_metas: List[Dict],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.
        
        Args:
            query_feats (list[Tensor]): Features of query image.
            support_feats_dict (dict[int, Tensor]): Dict of support features.
            proposal_list (list[Tensor]): Region proposals.
            query_img_metas (list[dict]): List of query image info.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
            
        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
        
        rois = bbox2roi(proposal_list)
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        
        # 获取分类和回归结果
        bbox_results = self._bbox_forward(query_roi_feats)
        
        # 分离每张图像的预测结果
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = bbox_results['cls_score'].split(num_proposals_per_img, 0)
        bbox_pred = bbox_results['bbox_pred'].split(num_proposals_per_img, 0)
        
        # 应用bbox后处理
        det_bboxes = []
        det_labels = []
        for i in range(len(proposal_list)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            
        return det_bboxes, det_labels
        
    def simple_test(self,
                   query_feats: List[Tensor],
                   support_feats_dict: Dict,
                   proposal_list: List[Tensor],
                   query_img_metas: List[Dict],
                   rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.
        
        Args:
            query_feats (list[Tensor]): Features of query image.
            support_feats_dict (dict[int, Tensor]): Dict of support features.
            proposal_list (list[Tensor]): Region proposals.
            query_img_metas (list[dict]): List of query image info.
            rescale (bool): Whether to rescale the results.
            
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feats_dict,
            proposal_list,
            query_img_metas,
            self.test_cfg,
            rescale=rescale)
            
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                       self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        
        return bbox_results
        
    def set_train_stage(self, is_base_training: bool):
        """Set the current training stage.
        
        Args:
            is_base_training (bool): Whether in base training stage.
        """
        self.is_base_training = is_base_training 
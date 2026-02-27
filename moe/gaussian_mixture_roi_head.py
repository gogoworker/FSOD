# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from mmdet.core.bbox.transforms import bbox2result
from mmcv.utils import ConfigDict
import numpy as np

@HEADS.register_module()
class GaussianMixtureRoIHead(MetaRCNNRoIHead):
    """用于高斯混合模型头部的ROI头。
    
    这个ROI头扩展了MetaRCNNRoIHead，以支持GaussianMixtureHead。
    它提供了使用支持集特征更新高斯混合模型参数的功能。
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
        """训练时的前向函数。
        
        Args:
            query_feats (list[Tensor]): 查询特征列表。
            support_feats (list[Tensor]): 支持集特征列表。
            proposals (list[Tensor]): 区域提议列表。
            query_img_metas (list[dict]): 查询图像元信息字典列表。
            query_gt_bboxes (list[Tensor]): 查询图像的真实边界框。
            query_gt_labels (list[Tensor]): 查询图像的真实标签。
            support_gt_labels (list[Tensor]): 支持集图像的真实标签。
            query_gt_bboxes_ignore (list[Tensor] | None): 要忽略的边界框。
                默认：None。
                
        Returns:
            dict[str, Tensor]: 损失组件字典。
        """
        # 提取支持集特征用于更新高斯混合模型参数
        support_feat = self.extract_support_feats(support_feats)[0]
        
        # 更新高斯混合模型参数（如果bbox_head支持）
        if hasattr(self.bbox_head, 'update_prototypes_with_support'):
            support_labels = torch.cat(support_gt_labels)
            self.bbox_head.update_prototypes_with_support(support_feat, support_labels)
        
        # 分配真实框并采样提议
        sampling_results = []
        if self.with_bbox:
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

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                query_feats, support_feats, sampling_results, query_img_metas,
                query_gt_bboxes, query_gt_labels, support_gt_labels)
            if bbox_results is not None:
                losses.update(bbox_results['loss_bbox'])

        return losses
        
    def _bbox_forward_train(self, query_feats, support_feats, sampling_results,
                          query_img_metas, query_gt_bboxes, query_gt_labels,
                          support_gt_labels):
        """重写_bbox_forward_train方法，以正确处理高斯混合模型损失。
        
        Args:
            query_feats (list[Tensor]): 查询特征列表。
            support_feats (list[Tensor]): 支持集特征列表。
            sampling_results (obj:`SamplingResult`): 采样结果。
            query_img_metas (list[dict]): 查询图像元信息字典列表。
            query_gt_bboxes (list[Tensor]): 查询图像的真实边界框。
            query_gt_labels (list[Tensor]): 查询图像的真实标签。
            support_gt_labels (list[Tensor]): 支持集图像的真实标签。
            
        Returns:
            dict: 预测结果和损失组件的字典。
        """
        # 提取支持集特征
        support_feat = self.extract_support_feats(support_feats)[0]
        
        # 计算ROIs和特征
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        
        # 获取标签等
        batch_size = len(query_img_metas)
        
        # 依据采样结果，计算出一批图像里所有样本的真实标签和边界框回归目标
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                 query_gt_bboxes,
                                                 query_gt_labels,
                                                 self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        
        # 初始化损失字典
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 
                    'loss_proto_inter': []}
        
        # 每张图片的样本数
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            
            # 从当前查询图像的标签中随机选择一个类别（元学习中的"任务类别"）
            if len(query_gt_labels[img_id]) > 0:
                random_index = torch.randint(
                    0, query_gt_labels[img_id].size(0), (1,))
                random_query_label = query_gt_labels[img_id][random_index]
            else:
                random_query_label = torch.zeros(1).long().to(
                    query_roi_feats.device)
            
            # 对每个支持集特征
            for i in range(support_feat.size(0)):
                # 如果支持集标签与随机查询标签匹配
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat[i].unsqueeze(0))
                    
                    # 计算当前"查询-支持"对的损失（cls\bbox\acc）
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], 
                        bbox_results['bbox_pred'],
                        query_rois[start:end], 
                        labels[start:end],
                        label_weights[start:end], 
                        bbox_targets[start:end],
                        bbox_weights[start:end])
                    
                    # 添加损失
                    for key in single_loss_bbox.keys():
                        if key in loss_bbox:
                            loss_bbox[key].append(single_loss_bbox[key])
        
        # 处理损失
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    if loss_bbox['acc']:  # 确保列表不为空
                        loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                    else:
                        loss_bbox[key] = torch.zeros(1).mean().to(query_roi_feats.device)
                else:
                    if loss_bbox[key]:  # 确保列表不为空
                        loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size
                    else:
                        loss_bbox[key] = torch.zeros(1).mean().to(query_roi_feats.device)
        
        # 添加原型损失（如果列表为空）
        if not loss_bbox.get('loss_proto_inter'):
            proto_losses = self.bbox_head.compute_prototype_loss()
            for key, value in proto_losses.items():
                loss_bbox[key] = value
        
        # 更新结果
        if bbox_results is None:
            # 如果没有匹配的支持类别，创建一个空的bbox_results
            bbox_results = dict(
                cls_score=torch.zeros(0, self.bbox_head.num_classes + 1).to(query_roi_feats.device),
                bbox_pred=torch.zeros(0, self.bbox_head.num_classes * 4).to(query_roi_feats.device)
            )
        
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def _bbox_forward(self, query_roi_feats, support_feat):
        """计算框回归和分类的结果。
        
        Args:
            query_roi_feats (Tensor): RoI特征，形状为 (num_rois, in_channels)。
            support_feat (Tensor): 支持集特征，形状为 (1, in_channels)。
            
        Returns:
            dict: 计算结果的字典。
                cls_score (Tensor): 分类分数，形状为 (num_rois, num_classes + 1)。
                bbox_pred (Tensor): 边界框回归预测，形状为 (num_rois, num_classes * 4)。
        """
        # 将支持集特征传递给bbox_head的forward方法
        cls_score, bbox_pred = self.bbox_head(query_roi_feats, support_feat)
        
        bbox_results = dict(
            cls_score=cls_score, 
            bbox_pred=bbox_pred
        )
        return bbox_results
    
    def simple_test(self,
                   query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # 参考meta_rcnn_roi_head.py中的实现
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        # 使用support_feats_dict进行推理，与Meta-RCNN保持一致
        # 同时利用支持集特征更新GMM参数
        if self.bbox_head.meta_weight > 0 and hasattr(self.bbox_head, 'update_prototypes_with_support'):
            # 从support_feats_dict中提取所有支持集特征和标签
            support_feats = []
            support_labels = []
            for class_id, feat in support_feats_dict.items():
                if class_id < self.bbox_head.num_classes:  # 忽略背景类
                    support_feats.append(feat)
                    support_labels.append(torch.tensor([class_id], device=feat.device))
            
            # 拼接所有支持集特征和标签
            if support_feats and support_labels:
                support_feat = torch.cat(support_feats, dim=0)
                support_label = torch.cat(support_labels, dim=0)
                # 更新GMM参数
                self.bbox_head.update_prototypes_with_support(support_feat, support_label)
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feats_dict,
            query_img_metas,
            proposal_list,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results
    
    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """测试边界框，参考meta_rcnn_roi_head.py的实现。
        
        Args:
            query_feats (list[Tensor]): 查询图像特征。
            support_feats_dict (dict[int, Tensor]): 支持集特征字典。
            query_img_metas (list[dict]): 查询图像元信息。
            proposals (list[Tensor]): 区域提议。
            rcnn_test_cfg (obj:`ConfigDict`): 测试配置。
            rescale (bool): 是否缩放结果。默认：False。
            
        Returns:
            tuple[list[Tensor], list[Tensor]]: 检测框和标签。
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
        
        # 检查并处理proposals
        valid_proposals = []
        for i, proposal in enumerate(proposals):
            # 处理空的proposal
            if proposal is None or (isinstance(proposal, torch.Tensor) and proposal.numel() == 0):
                # 创建一个空的tensor作为placeholder
                device = query_feats[0].device
                valid_proposals.append(torch.zeros((0, 4), device=device))
            # 处理非tensor的proposal (如int)
            elif not isinstance(proposal, torch.Tensor):
                device = query_feats[0].device
                valid_proposals.append(torch.zeros((0, 4), device=device))
            else:
                valid_proposals.append(proposal)
        
        # 转换proposals为rois
        rois = bbox2roi(valid_proposals)
        
        # 如果没有有效的ROIs，直接返回空结果
        if rois.size(0) == 0:
            det_bboxes = []
            det_labels = []
            for i in range(len(valid_proposals)):
                det_bbox = torch.zeros((0, 5), device=rois.device)
                det_label = torch.zeros((0,), dtype=torch.long, device=rois.device)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            return det_bboxes, det_labels
        
        # 提取ROI特征
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        
        # 为每个类别计算预测结果
        for class_id in support_feats_dict.keys():
            if not isinstance(class_id, int) or class_id >= num_classes + 1:
                continue
                
            support_feat = support_feats_dict[class_id]
            # 确保support_feat是tensor
            if not isinstance(support_feat, torch.Tensor):
                continue
                
            # 前向传播计算bbox结果
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            
            # 提取对应类别的分类和回归结果
            if class_id < num_classes:  # 非背景类
                cls_scores_dict[class_id] = \
                    bbox_results['cls_score'][:, class_id:class_id + 1]
                bbox_preds_dict[class_id] = \
                    bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
                
            # 处理背景类分数
            if class_id < num_classes:  # 只有非背景类才对背景分数做贡献
                if cls_scores_dict.get(num_classes, None) is None:
                    cls_scores_dict[num_classes] = \
                        bbox_results['cls_score'][:, -1:]
                else:
                    cls_scores_dict[num_classes] += \
                        bbox_results['cls_score'][:, -1:]
        
        # 计算背景类的平均分数
        if num_classes in cls_scores_dict and len(support_feats_dict.keys()) > 0:
            # 计算有贡献的类别数
            contributing_classes = sum(1 for k in support_feats_dict.keys() 
                                      if isinstance(k, int) and k < num_classes)
            if contributing_classes > 0:
                cls_scores_dict[num_classes] /= contributing_classes
        
        # 如果cls_scores_dict为空，则为所有类别创建零分数
        if not cls_scores_dict:
            # 创建一个大小为[N, 1]的零张量作为每个类别的分数
            zero_score = torch.zeros((query_roi_feats.size(0), 1), 
                                     device=query_roi_feats.device)
            for i in range(num_classes + 1):
                cls_scores_dict[i] = zero_score.clone()
            
            # 为每个前景类创建零回归预测
            zero_pred = torch.zeros((query_roi_feats.size(0), 4), 
                                    device=query_roi_feats.device)
            for i in range(num_classes):
                bbox_preds_dict[i] = zero_pred.clone()
        
        # 为每个类别创建得分列表和回归预测列表
        cls_scores = [
            cls_scores_dict.get(i, torch.zeros_like(list(cls_scores_dict.values())[0]))
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict.get(i, torch.zeros_like(list(bbox_preds_dict.values())[0]))
            for i in range(num_classes)
        ]
        
        # 拼接得分和回归预测
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # 分割每张图片的预测结果
        num_proposals_per_img = tuple(len(p) for p in valid_proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # 对每张图片应用后处理
        det_bboxes = []
        det_labels = []
        for i in range(len(valid_proposals)):
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
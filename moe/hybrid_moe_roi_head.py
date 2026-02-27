# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead


@HEADS.register_module()
class HybridMoERoIHead(MetaRCNNRoIHead):
    """用于混合MoE头的ROI头。
    
    这个ROI头扩展了MetaRCNNRoIHead，以支持HybridMoEHead。
    它提供了使用支持集特征更新原型的功能。
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
        # 提取支持集特征用于更新原型
        #support_feat = self.extract_support_feats(support_feats)[0]
        
        # 更新原型（如果bbox_head支持）
        # if hasattr(self.bbox_head, 'update_prototypes_with_support'):
        #     support_labels = torch.cat(support_gt_labels)
        #     self.bbox_head.update_prototypes_with_support(support_feat, support_labels)
        
        # 调用父类的forward_train进行常规训练
        # assign gts and sample proposals
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
        """重写_bbox_forward_train方法，以正确处理原型损失。
        
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
                    'loss_proto_diversity': [], 'loss_proto_inter': []}
        
        # 每张图片的样本数
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            
            # 从当前查询图像的标签中随机选择一个类别（元学习中的“任务类别”）
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
                    
                    # 计算当前“查询-支持”对的损失（cls\bbox\acc）
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
        
        # meta分类损失
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)
        
        # 添加原型损失（如果列表为空）
        if (not loss_bbox.get('loss_proto_diversity') or 
            not loss_bbox.get('loss_proto_inter')):
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
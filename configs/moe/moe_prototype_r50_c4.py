_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
]

# 自定义导入设置
custom_imports = dict(
    imports=['moe.moe_prototype_head', 'moe.moe_prototype_roi_head'],
    allow_failed_imports=False
)

# model settings
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    type='MetaRCNN',
    pretrained=pretrained,
    backbone=dict(type='ResNetWithMetaConv', frozen_stages=2, depth=101),
    rpn_head=dict(
        feat_channels=512, loss_cls=dict(use_sigmoid=False, loss_weight=1.0)),
    roi_head=dict(
        type='MoEPrototypeRoIHead',
        shared_head=dict(type='MetaRCNNResLayer', pretrained=pretrained),
        bbox_head=dict(
            type='MoEPrototypeHead',
            with_avg_pool=False,
            in_channels=2048,
            roi_feat_size=1,
            num_classes=80,
            num_experts=4,
            feat_dim=2048,
            use_bias=False,
            prototype_init_std=0.01,
            ema_decay=0.99,
            with_meta_cls_loss=False,  # 禁用meta分类损失
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DotProductAggregator',
                    in_channels=2048,
                    with_fc=False)
            ])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100))) 
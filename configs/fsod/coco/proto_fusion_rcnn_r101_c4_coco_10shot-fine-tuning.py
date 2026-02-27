_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../proto_fusion_rcnn_r101_c4.py',
    '../../_base_/default_runtime.py'
]

# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='ProtoFusion', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10)),
    val=dict(classes='ALL_CLASSES'),
    test=dict(classes='ALL_CLASSES'),
    model_init=dict(classes='ALL_CLASSES'))

evaluation = dict(interval=500, class_splits=['BASE_CLASSES', 'NOVEL_CLASSES'])
checkpoint_config = dict(interval=500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=2000)

# load_from = 'path of base training model'
load_from = 'work_dirs/proto_fusion_rcnn_r101_c4_coco_base-training/latest.pth'

# model settings
model = dict(
    roi_head=dict(
        is_base_training=False,
        proto_weight=0.6,
        momentum=0.998,
        temperature=0.08,
        bbox_head=dict(num_classes=80, num_meta_classes=80)),
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ]) 
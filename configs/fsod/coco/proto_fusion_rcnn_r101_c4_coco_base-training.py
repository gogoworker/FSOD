_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco.py',
    '../../_base_/schedules/schedule.py', '../proto_fusion_rcnn_r101_c4.py',
    '../../_base_/default_runtime.py'
]

# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES')),
    val=dict(classes='BASE_CLASSES'),
    test=dict(classes='BASE_CLASSES'),
    model_init=dict(classes='BASE_CLASSES'))

lr_config = dict(warmup_iters=500, step=[120000])
evaluation = dict(interval=10000)
checkpoint_config = dict(interval=10000)
runner = dict(max_iters=130000)
optimizer = dict(lr=0.005)

# model settings
model = dict(
    roi_head=dict(
        is_base_training=True,
        proto_weight=0.2,  # 基类训练阶段降低原型权重
        bbox_head=dict(num_classes=60, num_meta_classes=60))) 
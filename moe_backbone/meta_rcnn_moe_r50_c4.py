_base_ = [
    '../configs/_base_/meta-rcnn_r50_c4.py',
]

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# 修改模型配置以使用MoE backbone
model = dict(
    backbone=dict(
        type='ResNetWithMetaConvMoE',
        pretrained=pretrained,
        depth=101,
        MoE_Block_inds=[[0, 1], [0, 2], [0, 3, 5], [0, 2]],  # 在每个阶段中指定使用MoE的块索引
        num_experts=8,  # 专家数量
        top_k=2,        # 每次选择的专家数量
        gate='cosine',  # 门控机制类型
        noisy_gating=True,  # 是否使用噪声门控
        frozen_stages=2,  # 冻结前两个阶段
    )
)

# 训练配置
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]) 
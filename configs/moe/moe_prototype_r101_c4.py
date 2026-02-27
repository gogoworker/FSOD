_base_ = [
    '../fsod/meta-rcnn_r50_c4.py',
]

# 自定义导入设置
custom_imports = dict(
    imports=['moe.moe_prototype_head', 'moe.moe_prototype_roi_head'],
    allow_failed_imports=False
)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# 模型设置
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='MoEPrototypeRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='MoEPrototypeHead',
            num_experts=4,
            use_bias=False,
            prototype_init_std=0.01,
            ema_decay=0.99,
            reg_class_agnostic=True,
        )
    )
) 
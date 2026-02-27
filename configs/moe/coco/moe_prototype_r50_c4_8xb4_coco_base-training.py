_base_ = [
    '../../../mmfewshot/configs/_base_/datasets/nway_kshot/base_coco.py',
    '../../../mmfewshot/configs/_base_/schedules/schedule.py',
    '../moe_prototype_r50_c4.py',
    '../../../mmfewshot/configs/_base_/default_runtime.py'
]

# 训练设置
lr_config = dict(warmup=None, step=[110000])
checkpoint_config = dict(interval=10000)
runner = dict(max_iters=120000)
optimizer = dict(lr=0.005)

# 模型设置
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=60,  # COCO基础类别数为60
            num_experts=4
        )
    )
) 
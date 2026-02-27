_base_ = [
    '../../../mmfewshot/configs/_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../../mmfewshot/configs/_base_/schedules/schedule.py',
    '../moe_prototype_r50_c4.py',
    '../../../mmfewshot/configs/_base_/default_runtime.py'
]

# 数据集设置
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MoEPrototype', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
        )
    )
)

# 评估设置
evaluation = dict(interval=500, class_splits=['BASE_CLASSES', 'NOVEL_CLASSES'])

# 训练设置
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[5000])
runner = dict(max_iters=6000)
checkpoint_config = dict(interval=500)

# 模型设置
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=80, num_experts=4)  # COCO所有类别数为80
    ),
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ]
)

# 加载预训练模型
load_from = 'work_dirs/moe_prototype_r50_c4_8xb4_coco_base-training/latest.pth' 
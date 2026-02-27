_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../proto_fusion_rcnn_r101_c4.py',
    '../../../_base_/default_runtime.py'
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT1_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))

evaluation = dict(interval=50, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=50)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=1200)

# load_from = 'path of base training model'
load_from = 'work_dirs/test/proto_fusion_rcnn_r101_c4_voc-split1_base-training/latest.pth'

# model settings
model = dict(
    roi_head=dict(
        is_base_training=False,
        with_refine=True,
        proto_weight=0.62,  # 2-shot微调阶段的原型权重
        momentum=0.995,     # 样本增加，提高动量系数
        temperature=0.08,   # 调整温度参数
        bbox_head=dict(num_classes=20, num_meta_classes=20)),
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ]) 
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
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT1_1SHOT')],
            num_novel_shots=1,
            num_base_shots=1,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))

evaluation = dict(interval=50, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=50)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=400)

# load_from = 'path of base training model'
load_from = 'work_dirs/test/proto_fusion_rcnn_r101_c4_voc-split1_base-training/latest.pth'

# model settings
model = dict(
    roi_head=dict(
        is_base_training=False,
        with_refine=True,
        proto_weight=0.7,  # 少样本微调阶段提高原型权重
        momentum=0.99,     # 适当降低动量以加快新类原型更新
        temperature=0.07,  # 降低温度参数使相似度分布更加尖锐
        bbox_head=dict(num_classes=20, num_meta_classes=20)),
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ]) 
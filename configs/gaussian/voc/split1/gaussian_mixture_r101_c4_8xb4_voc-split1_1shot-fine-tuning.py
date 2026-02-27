_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc_ms.py',
    '../../../_base_/schedules/schedule.py',
    '../../gaussian_mixture_r101_c4.py',
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

evaluation = dict(
    interval=200, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=200)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=800)


load_from = 'work_dirs/gaussian_mixture_r101_c4_8xb4_voc-split1_base-training/latest.pth'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20, num_components=4)
    ),
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ]
) 
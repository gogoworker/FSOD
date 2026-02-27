_base_ = [
    '../fsod/meta-rcnn_r50_c4.py',
]

# 自定义导入设置
custom_imports = dict(
    imports=['moe.gaussian_mixture_head', 'moe.gaussian_mixture_roi_head'],
    allow_failed_imports=False
)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# 模型设置
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='GaussianMixtureRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='GaussianMixtureHead',
            num_components=2,  # 根据论文，使用2个高斯组件效果最佳
            in_channels=2048,  # 确保与backbone输出维度匹配
            feat_dim=2048,     # 设置为与in_channels相同
            use_bias=False,
            prototype_init_std=0.01,
            cov_init_value=0.1,  # 协方差矩阵初始值
            ema_decay=0.99,
            meta_weight=0.0,  # 完全依赖GMM分类器
            use_proto_regularization=True,
            loss_proto_inter_weight=1.0,  # 类间KL散度损失权重
            reg_class_agnostic=True,
        )
    )
) 
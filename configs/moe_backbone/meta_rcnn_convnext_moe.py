_base_ = [
    '../_base_/meta-rcnn_r50_c4.py',
]

pretrained = 'data/convnext_tiny_1k_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ConvNeXtMoE',
        arch='tiny',
        in_channels=3,
        # depths=[3, 3, 9, 3],
        # dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        moe_block_inds=[[0, 1], [0, 2], [0, 3, 5], [0, 2]],  # 可根据实验调整
        num_experts=4,
        top_k=2,
        gate='cosine',
        noisy_gating=True,
        pretrained=pretrained,
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    ),
)

custom_imports = dict(
    imports=['FSOD.moe_backbone.convnext_moe', 'FSOD.moe_backbone.moe_hooks'],
    allow_failed_imports=False)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MoELossLoggerHook', log_interval=10),
    ]) 
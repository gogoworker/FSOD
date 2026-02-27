_base_ = [
    './meta-rcnn_r50_c4.py',
]

custom_imports = dict(
    imports=[
        'fsod.roi_head',
        'fsod.utils'], 
    allow_failed_imports=False)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='ProtoFusionRoIHead',
        shared_head=dict(pretrained=pretrained),
        momentum=0.999,
        temperature=0.1,
        proto_weight=0.5)) 
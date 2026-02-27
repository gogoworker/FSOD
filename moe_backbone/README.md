# Meta-RCNN with MoE Backbone

本目录包含了基于Meta-RCNN的MoE backbone实现，用于少样本目标检测。

## 介绍

Meta-RCNN是一种有效的少样本目标检测方法，通过元学习提高模型在少样本场景下的泛化能力。本实现在Meta-RCNN的基础上，引入了Mixture of Experts (MoE)机制，通过多个专家网络提高特征提取能力。

主要特点：
- 基于ResNetWithMetaConv的MoE实现
- 支持可配置的专家数量和选择机制
- 保持与原始Meta-RCNN相同的输入输出接口
- 灵活的MoE块配置，可以指定在ResNet的哪些块使用MoE

## 文件说明

- `resnet_with_meta_conv_moe.py`: MoE backbone的实现
- `meta_rcnn_moe_r50_c4.py`: 使用MoE backbone的Meta-RCNN配置文件

## 使用方法

### 训练

```bash
# 使用单GPU训练
python tools/train.py FSOD/moe_backbone/meta_rcnn_moe_r50_c4.py

# 使用多GPU训练
./tools/dist_train.sh FSOD/moe_backbone/meta_rcnn_moe_r50_c4.py ${GPU_NUM}
```

### 测试

```bash
# 使用单GPU测试
python tools/test.py FSOD/moe_backbone/meta_rcnn_moe_r50_c4.py ${CHECKPOINT_FILE} --eval mAP

# 使用多GPU测试
./tools/dist_test.sh FSOD/moe_backbone/meta_rcnn_moe_r50_c4.py ${CHECKPOINT_FILE} ${GPU_NUM} --eval mAP
```

## 配置说明

在配置文件中，可以通过以下参数配置MoE backbone：

```python
model = dict(
    backbone=dict(
        type='ResNetWithMetaConvMoE',
        depth=50,  # ResNet深度，支持18/34/50/101/152
        MoE_Block_inds=[[0, 1], [0, 2], [0, 3, 5], [0, 2]],  # 在每个阶段中指定使用MoE的块索引
        num_experts=8,  # 专家数量
        top_k=2,        # 每次选择的专家数量
        gate='cosine',  # 门控机制类型，支持'linear'和'cosine'
        noisy_gating=True,  # 是否使用噪声门控
        frozen_stages=2,  # 冻结前两个阶段
    )
)
```

### MoE_Block_inds参数说明

`MoE_Block_inds`是一个包含4个列表的列表，对应ResNet的4个阶段。每个子列表中的数字表示在该阶段中哪些块使用MoE。例如：

```python
MoE_Block_inds=[[0, 1], [0, 2], [0, 3, 5], [0, 2]]
```

表示：
- 第1阶段：第0和第1个块使用MoE
- 第2阶段：第0和第2个块使用MoE
- 第3阶段：第0、第3和第5个块使用MoE
- 第4阶段：第0和第2个块使用MoE

## 实现细节

MoE层的实现基于以下组件：

1. **CosineTopKGate**: 基于余弦相似度的门控机制
2. **SparseDispatcher**: 负责将输入分发到各个专家并组合结果
3. **MoE_layer**: 核心MoE实现，包括门控、专家选择和输出组合
4. **MoEBasicBlock/MoEBottleneck**: 带有MoE的ResNet基本块和瓶颈块

## 注意事项

- MoE机制会增加模型参数量和计算复杂度，请根据实际硬件资源调整专家数量和MoE块的配置
- 建议从较小的配置开始（如少量专家和MoE块），然后根据性能逐步增加复杂度
- 训练初期可能会出现不稳定，可以考虑使用较小的学习率或更长的预热期 
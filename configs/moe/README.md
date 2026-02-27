# MoE原型方法 (Multiple Expert Prototypes)

## 简介

MoE原型方法是一种基于原型的少样本目标检测方法，它引入了"多个专家原型"的概念，每个类别有多个原型表示，每个RoI特征根据自己的特征软分配权重，决定由哪个原型来"指导分类"。

主要特点：
- 每类多个原型 (Multiple Expert Prototypes)：每个类别由多个原型表示，可以更好地捕捉类内多样性
- Gate软分配 (Soft Assignment)：动态分配每个RoI特征与各个原型的关联权重
- 基于余弦相似度的分类 (Cosine Similarity)：使用特征与原型的余弦相似度进行分类

## 模型架构

该方法基于Meta-RCNN架构，主要修改了以下部分：
1. 使用`MoEPrototypeHead`替换原始的分类头，实现多专家原型分类
2. 使用`MoEPrototypeRoIHead`替换原始的ROI Head，实现对原型的更新

## 使用方法

### 训练基础模型

```bash
# 在VOC split1上训练基础模型
python tools/train.py \
    FSOD/configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_base-training.py \
    --work-dir work_dirs/moe_prototype_r101_c4_8xb4_voc-split1_base-training

# 在COCO上训练基础模型
python tools/train.py \
    FSOD/configs/moe/coco/moe_prototype_r50_c4_8xb4_coco_base-training.py \
    --work-dir work_dirs/moe_prototype_r50_c4_8xb4_coco_base-training
```

### 微调模型

```bash
# 在VOC split1上进行1-shot微调
python tools/train.py \
    FSOD/configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py \
    --work-dir work_dirs/moe_prototype_r101_c4_8xb4_voc-split1_1shot-fine-tuning

# 在COCO上进行10-shot微调
python tools/train.py \
    FSOD/configs/moe/coco/moe_prototype_r50_c4_8xb4_coco_10shot-fine-tuning.py \
    --work-dir work_dirs/moe_prototype_r50_c4_8xb4_coco_10shot-fine-tuning
```

### 测试模型

```bash
# 测试VOC split1上的1-shot模型
python tools/test.py \
    FSOD/configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py \
    work_dirs/moe_prototype_r101_c4_8xb4_voc-split1_1shot-fine-tuning/latest.pth \
    --eval mAP

# 测试COCO上的10-shot模型
python tools/test.py \
    FSOD/configs/moe/coco/moe_prototype_r50_c4_8xb4_coco_10shot-fine-tuning.py \
    work_dirs/moe_prototype_r50_c4_8xb4_coco_10shot-fine-tuning/latest.pth \
    --eval bbox
```

## 参数配置

主要参数说明：
- `num_experts`: 每个类别的专家原型数量，默认为4
- `feat_dim`: 特征维度，通常与backbone输出的特征维度一致
- `use_bias`: 是否在gate网络中使用偏置项
- `prototype_init_std`: 原型初始化的标准差
- `ema_decay`: 原型更新的指数移动平均衰减系数

## 方法原理

1. **特征提取**：使用backbone和ROI提取器获取RoI特征
2. **原型表示**：每个类别由多个原型表示，通过支持集样本初始化和更新
3. **动态分配**：使用gate网络计算每个RoI特征与各个原型的关联权重
4. **相似度计算**：计算RoI特征与原型的余弦相似度
5. **加权融合**：根据gate权重对多个专家原型的相似度进行加权融合
6. **分类决策**：基于加权融合后的相似度进行分类 
# 高斯混合模型用于少样本目标检测

## 简介

本目录包含基于高斯混合模型(GMM)的少样本目标检测方法的配置文件。与传统的基于原型的分类器相比，高斯混合模型可以更好地捕捉类内多样性和类间差异，特别是在特征空间中表现为多模态分布的情况。

## 配置文件说明

### 主配置文件

- `gaussian_mixture_r101_c4.py`: 基于ResNet-101骨干网络的高斯混合模型配置

### VOC数据集配置

#### Split1

- `voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_base-training.py`: 基础类别训练配置
- `voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py`: 1-shot微调配置
- `voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py`: 2-shot微调配置
- `voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py`: 3-shot微调配置
- `voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py`: 5-shot微调配置
- `voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py`: 10-shot微调配置

## 训练流程

### 1. 基础类别训练

```bash
# 单GPU训练
python tools/train.py configs/gaussian/voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_base-training.py

# 多GPU训练
bash tools/dist_train.sh configs/gaussian/voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_base-training.py 8
```

### 2. 微调

```bash
# 单GPU微调 (以5-shot为例)
python tools/train.py configs/gaussian/voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py

# 多GPU微调
bash tools/dist_train.sh configs/gaussian/voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py 8
```

### 3. 测试

```bash
# 单GPU测试
python tools/test.py configs/gaussian/voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py work_dirs/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning/latest.pth --eval mAP

# 多GPU测试
bash tools/dist_test.sh configs/gaussian/voc/split1/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py work_dirs/gaussian_mixture_r101_c4_8xb4_voc-split1_5shot-fine-tuning/latest.pth 8 --eval mAP
```

## 模型特点

- **高斯混合建模**: 每个类别由多个高斯分布组成，可以更好地捕捉类内多样性
- **可学习的协方差矩阵**: 通过协方差矩阵建模特征的不确定性
- **基于KL散度的类别区分**: 通过KL散度最大化不同类别之间的差异
- **加权混合机制**: 自适应调整不同高斯分布的重要性 
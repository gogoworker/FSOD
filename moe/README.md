# 高斯混合模型用于少样本目标检测

## 简介

本项目实现了基于高斯混合模型(GMM)的少样本目标检测方法。与传统的基于原型的分类器相比，高斯混合模型可以更好地捕捉类内多样性和类间差异，特别是在特征空间中表现为多模态分布的情况。

## 模型结构

### GaussianMixtureHead

`GaussianMixtureHead`是一个基于高斯混合模型的分类头部，它将每个类别建模为多个高斯分布的混合。主要特点包括：

- 每个类别由多个高斯分布组成，每个高斯分布有自己的均值和协方差矩阵
- 使用可学习的混合权重来平衡不同高斯分布的重要性
- 通过KL散度衡量不同类别GMM之间的差异，促进类间区分
- 通过余弦相似度损失鼓励同类高斯分布的多样性

### GaussianMixtureRoIHead

`GaussianMixtureRoIHead`是配合`GaussianMixtureHead`使用的ROI头部，它负责：

- 从支持集中提取特征并更新高斯混合模型参数
- 处理查询图像的ROI特征并进行分类和回归
- 计算和优化各种损失函数

## 使用方法

### 配置文件示例

```python
# 模型配置
model = dict(
    type='FasterRCNN',
    ...
    roi_head=dict(
        type='GaussianMixtureRoIHead',
        bbox_roi_extractor=dict(...),
        bbox_head=dict(
            type='GaussianMixtureHead',
            num_classes=20,
            in_channels=1024,
            num_components=4,  # 每个类别的高斯分布数量
            feat_dim=1024,
            use_bias=True,
            prototype_init_std=0.01,
            cov_init_value=0.1,  # 协方差矩阵初始值
            meta_weight=0.5,  # 元分类器权重
            use_proto_regularization=True,
            loss_proto_diversity_weight=0.2,
            loss_proto_inter_weight=0.5,
        ),
        ...
    ),
    ...
)
```

### 训练

训练过程与标准的Meta-RCNN类似，但在损失计算中增加了高斯混合模型特有的损失项：

1. 类内多样性损失：鼓励同一类别的不同高斯分布表示不同的特征模式
2. 类间KL散度损失：最大化不同类别GMM之间的KL散度，增强类别区分性

### 可视化

模型提供了可视化方法`visualize_gaussian_mixture`，可以将高维的高斯混合模型通过t-SNE降维到2D空间进行可视化：

```python
# 可视化示例
model.roi_head.bbox_head.visualize_gaussian_mixture(save_path='gmm_vis.png')
```

## 优势

与传统的基于原型的方法相比，高斯混合模型有以下优势：

1. 更好地建模类内多样性：每个类别由多个高斯分布组成，可以捕捉不同的视角、姿态等变化
2. 更精确的不确定性估计：通过协方差矩阵建模特征的不确定性
3. 更强的类别区分能力：通过KL散度最大化不同类别之间的差异

## 参考文献

- Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks.
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning.
- Fort, S. (2017). Gaussian prototypical networks for few-shot learning on omniglot. 
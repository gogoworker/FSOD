# MoE原型可视化工具使用指南

这个目录包含两个可视化工具，用于分析MoE原型的区分度和特征分布情况：

1. `visualize_moe_prototypes.py` - 可视化模型中的MoE原型分布
2. `visualize_moe_features.py` - 提取并可视化验证集数据的特征及其与原型的关系

## 依赖安装

确保已安装以下依赖：

```bash
pip install sklearn matplotlib seaborn
```

## 1. 可视化MoE原型

这个工具用于直接从模型中提取MoE原型并可视化它们的分布情况，包括t-SNE降维可视化和类内/类间相似度分析。

### 使用方法

```bash
python tools/visualize_moe_prototypes.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [OPTIONS]
```

### 参数说明

- `CONFIG_FILE`: 模型配置文件路径
- `CHECKPOINT_FILE`: 模型检查点文件路径
- `--work-dir`: 保存可视化结果的目录，默认为配置文件同名目录
- `--vis-class-names`: 要可视化的类别名称，如果不指定则可视化所有类别
- `--fig-size`: 图像大小，默认为[12, 10]
- `--dpi`: 图像DPI，默认为300
- `--seed`: 随机种子，默认为42
- `--perplexity`: t-SNE的困惑度参数，默认为30

### 输出文件

- `moe_prototypes_tsne.png`: MoE原型的t-SNE可视化
- `moe_prototypes_similarity.png`: 类别间原型余弦相似度热力图
- 每个类别的类内原型相似度热力图: `moe_prototypes_similarity_intra_${CLASS_NAME}.png`

## 2. 可视化特征分布和Gate激活

这个工具用于从验证集提取RoI特征，并可视化它们的分布以及与原型的关系，还可以分析gate网络对不同样本的激活模式。

### 使用方法

```bash
python tools/visualize_moe_features.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [OPTIONS]
```

### 参数说明

- `CONFIG_FILE`: 模型配置文件路径
- `CHECKPOINT_FILE`: 模型检查点文件路径
- `--work-dir`: 保存可视化结果的目录，默认为配置文件同名目录
- `--vis-class-names`: 要可视化的类别名称，如果不指定则可视化所有类别
- `--num-samples-per-class`: 每个类别要提取的样本数量，默认为50
- `--vis-proto-feats`: 是否可视化原型与特征之间的关系
- `--vis-gate-activations`: 是否可视化gate激活模式
- `--fig-size`: 图像大小，默认为[12, 10]
- `--dpi`: 图像DPI，默认为300
- `--seed`: 随机种子，默认为42
- `--perplexity`: t-SNE的困惑度参数，默认为30

### 输出文件

- `roi_features_tsne.png`: RoI特征的t-SNE可视化
- `roi_features_with_prototypes_tsne.png`: RoI特征与原型共同的t-SNE可视化
- 每个类别的gate激活模式: `gate_activations_gate_${CLASS_NAME}.png`

## 可视化结果解读

### 1. 原型t-SNE分布

通过观察原型在t-SNE空间中的分布，我们可以判断：

- 不同类别的原型是否有明显的分离
- 同一类别的多个专家原型是否有合理的分散性
- 原型之间的相对距离关系

理想情况下，不同类别的原型应该形成明显分离的簇，同一类别的多个专家原型则应该在各自的区域内有一定分散性。

### 2. 原型相似度热力图

通过观察原型间的余弦相似度热力图，我们可以判断：

- 类间原型相似度：对角线外的值越小越好，表示不同类别的原型越不相似
- 类内原型相似度：同一类别的多个专家原型应该有一定的差异性，但不应该完全不相关

### 3. Gate激活模式

通过观察gate激活热力图和平均激活条形图，我们可以判断：

- 同一类别的不同样本是否有相似的专家分配模式
- 不同样本是否倾向于激活不同的专家（说明专家有特化性）
- 激活是否均匀分布在所有专家上，或是集中在少数专家上

理想情况下，gate网络应该根据样本的不同特点动态分配不同的专家权重，而不是对所有样本使用相同的专家组合。

## 示例用法

以下是一些典型的使用示例：

1. 可视化模型原型分布：

```bash
python tools/visualize_moe_prototypes.py configs/moe/voc/meta_rcnn_moe_r50_c4_8xb4_voc-split1_5shot-fine-tuning.py \
       work_dirs/meta_rcnn_moe_r50_c4_voc/latest.pth \
       --work-dir work_dirs/vis_results
```

2. 可视化特征和gate激活：

```bash
python tools/visualize_moe_features.py configs/moe/voc/meta_rcnn_moe_r50_c4_8xb4_voc-split1_5shot-fine-tuning.py \
       work_dirs/meta_rcnn_moe_r50_c4_voc/latest.pth \
       --work-dir work_dirs/vis_results \
       --vis-proto-feats --vis-gate-activations
```

3. 仅对特定类别进行可视化：

```bash
python tools/visualize_moe_prototypes.py configs/moe/voc/meta_rcnn_moe_r50_c4_8xb4_voc-split1_5shot-fine-tuning.py \
       work_dirs/meta_rcnn_moe_r50_c4_voc/latest.pth \
       --vis-class-names dog cat person
``` 
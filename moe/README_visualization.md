# 原型分布可视化功能

这个模块为`HybridMoEHead`提供了完整的原型分布可视化功能，帮助你在评估过程中分析和理解MoE模型的内部机制。

## 功能特性

### 1. 原型分布可视化
- **相似度热力图**: 显示所有原型之间的余弦相似度
- **类内专家相似度分布**: 分析同类专家的多样性
- **类间专家相似度分布**: 分析不同类专家的分离性
- **降维可视化**: 使用t-SNE或PCA将原型投影到2D空间
- **原型统计信息**: 包括范数分布、激活情况等

### 2. 门控权重可视化
- **门控权重分布**: 显示每个类别对专家的平均使用情况
- **门控权重统计**: 分析门控网络的稀疏性和分布

### 3. 演化分析
- **训练过程跟踪**: 记录每个epoch的原型统计信息
- **演化图表**: 绘制类内多样性、类间分离性等指标的变化趋势

## 安装依赖

```bash
pip install matplotlib seaborn scikit-learn pandas numpy
```

## 使用方法

### 1. 在模型配置中启用可视化

```python
# 在HybridMoEHead的配置中添加
model_config = {
    'num_classes': 80,
    'in_channels': 2048,
    'num_experts': 4,
    'feat_dim': 2048,
    'enable_visualization': True,  # 启用可视化
    'vis_save_dir': 'prototype_vis'  # 可视化结果保存目录
}
```

### 2. 在训练循环中使用

```python
from moe.evaluation_with_visualization import evaluate_with_prototype_visualization

# 在验证阶段调用可视化
def validate_epoch(model, val_loader, epoch):
    model.eval()
    
    # 获取MoE头
    moe_head = None
    for module in model.modules():
        if hasattr(module, 'prototypes') and hasattr(module, 'visualize_prototypes'):
            moe_head = module
            break
    
    if moe_head is not None:
        # 可视化原型分布
        vis_files = moe_head.visualize_prototypes(
            epoch=epoch,
            class_names=dataset.class_names,
            method='tsne'  # 或 'pca'
        )
        
        # 获取统计信息
        stats = moe_head.get_prototype_statistics()
        print(f"类内多样性: {stats['intra_diversity']:.4f}")
        print(f"类间分离性: {stats['inter_separation']:.4f}")
```

### 3. 在评估过程中使用

```python
from moe.evaluation_with_visualization import evaluate_with_prototype_visualization

# 在评估时调用
evaluate_with_prototype_visualization(
    model=model,
    data_loader=test_loader,
    epoch=0,
    class_names=dataset.class_names,
    save_dir="evaluation_vis"
)
```

### 4. 创建分析报告

```python
from moe.evaluation_with_visualization import create_prototype_analysis_report

# 创建详细的分析报告
create_prototype_analysis_report(
    model=model,
    class_names=dataset.class_names,
    save_dir="prototype_analysis"
)
```

### 5. 绘制演化过程

```python
from moe.evaluation_with_visualization import plot_prototype_evolution

# 假设你已经收集了训练过程中的统计信息
statistics_history = [
    {'intra_diversity': 0.8, 'inter_separation': 0.9, ...},
    {'intra_diversity': 0.85, 'inter_separation': 0.92, ...},
    # ...
]

plot_prototype_evolution(statistics_history, 'evolution_plots')
```

## 可视化输出

### 1. 相似度热力图 (`similarity_heatmap_epoch_X.png`)
- 显示所有原型之间的余弦相似度
- 颜色越深表示相似度越高
- 对角线为1（自身相似度）

### 2. 类内相似度分布 (`intra_similarity_epoch_X.png`)
- 箱线图显示每个类别内专家的相似度分布
- 相似度越低表示类内多样性越好

### 3. 类间相似度分布 (`inter_similarity_epoch_X.png`)
- 箱线图显示不同类别间专家的相似度分布
- 相似度越低表示类间分离性越好

### 4. 降维可视化 (`tsne_visualization_epoch_X.png` 或 `pca_visualization_epoch_X.png`)
- 将原型投影到2D空间
- 不同颜色代表不同类别
- 数字表示专家编号

### 5. 原型统计信息 (`prototype_statistics_epoch_X.png`)
- 原型范数分布
- 每个类别的平均类内相似度
- 类间相似度矩阵
- 每个类别的活跃专家数量

### 6. 门控权重分布 (`gate_distribution_epoch_X.png`)
- 热力图显示每个类别对专家的平均使用情况
- 数值表示平均门控权重

## 统计指标说明

### 类内多样性 (intra_diversity)
- 范围: [0, 1]
- 值越高表示同类专家的方向越不同
- 理想值: 接近1

### 类间分离性 (inter_separation)
- 范围: [0, 1]
- 值越高表示不同类专家的方向越不同
- 理想值: 接近1

### 平均原型范数 (avg_proto_norm)
- 表示原型的平均长度
- 归一化后应该接近1

### 门控权重稀疏性 (sparsity)
- 范围: [0, 1]
- 值越高表示门控网络越稀疏
- 理想值: 适中（既不过于稀疏也不过于密集）

## 高级用法

### 1. 自定义可视化参数

```python
# 自定义t-SNE参数
vis_files = moe_head.visualizer.visualize_prototype_distribution(
    prototypes=moe_head.prototypes,
    class_names=class_names,
    epoch=epoch,
    method='tsne'  # 或 'pca'
)
```

### 2. 批量处理可视化

```python
# 在训练过程中定期保存统计信息
statistics_history = []

for epoch in range(num_epochs):
    # 训练代码...
    
    if epoch % 10 == 0:
        stats = moe_head.get_prototype_statistics()
        statistics_history.append(stats)
        
        # 保存到文件
        with open(f'stats_epoch_{epoch}.json', 'w') as f:
            json.dump(stats, f, indent=2)

# 训练结束后绘制演化图
plot_prototype_evolution(statistics_history, 'evolution_plots')
```

### 3. 实时监控

```python
# 在训练循环中实时监控关键指标
for epoch in range(num_epochs):
    # 训练代码...
    
    if epoch % 5 == 0:  # 每5个epoch检查一次
        stats = moe_head.get_prototype_statistics()
        
        # 检查是否出现异常
        if stats['intra_diversity'] < 0.5:
            print("警告: 类内多样性过低!")
        if stats['inter_separation'] < 0.5:
            print("警告: 类间分离性过低!")
```

## 故障排除

### 1. 找不到MoE头
```python
# 确保模型包含HybridMoEHead
moe_head = None
for module in model.modules():
    if hasattr(module, 'prototypes') and hasattr(module, 'visualize_prototypes'):
        moe_head = module
        break

if moe_head is None:
    print("未找到MoE头，请检查模型结构")
```

### 2. 可视化文件未生成
- 检查`enable_visualization`是否设置为`True`
- 检查`vis_save_dir`目录是否有写入权限
- 确保已安装所有依赖包

### 3. 内存不足
- 对于大数据集，可以减少可视化的频率
- 使用`torch.no_grad()`减少内存使用
- 考虑使用较小的降维方法（如PCA而不是t-SNE）

## 性能优化

### 1. 减少可视化频率
```python
# 只在特定epoch进行可视化
if epoch % 20 == 0:  # 每20个epoch可视化一次
    vis_files = moe_head.visualize_prototypes(epoch=epoch)
```

### 2. 使用更快的降维方法
```python
# PCA比t-SNE更快
vis_files = moe_head.visualize_prototypes(
    epoch=epoch,
    method='pca'  # 使用PCA而不是t-SNE
)
```

### 3. 异步可视化
```python
import threading

def async_visualization(moe_head, epoch):
    vis_files = moe_head.visualize_prototypes(epoch=epoch)
    print(f"可视化完成: {vis_files}")

# 在训练循环中异步调用
if epoch % 10 == 0:
    thread = threading.Thread(
        target=async_visualization, 
        args=(moe_head, epoch)
    )
    thread.start()
```

## 扩展功能

### 1. 添加新的可视化类型
```python
# 在PrototypeVisualizer中添加新方法
def visualize_custom_analysis(self, prototypes, **kwargs):
    # 自定义可视化逻辑
    pass
```

### 2. 集成到TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/prototype_analysis')

# 在训练循环中记录统计信息
stats = moe_head.get_prototype_statistics()
writer.add_scalar('Prototype/IntraDiversity', stats['intra_diversity'], epoch)
writer.add_scalar('Prototype/InterSeparation', stats['inter_separation'], epoch)
```

这个可视化系统可以帮助你深入理解MoE模型的工作原理，优化模型性能，并发现潜在的问题。 
#!/usr/bin/env python3
"""
评估过程中原型分布可视化示例脚本
"""

import torch
import os
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# 假设你已经有了训练好的模型
def evaluate_with_prototype_visualization(model, 
                                        data_loader, 
                                        epoch: int = 0,
                                        class_names: Optional[List[str]] = None,
                                        save_dir: str = "evaluation_vis"):
    """在评估过程中可视化原型分布
    
    Args:
        model: 训练好的模型
        data_loader: 评估数据加载器
        epoch: 当前epoch
        class_names: 类别名称列表
        save_dir: 可视化结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保模型处于评估模式
    model.eval()
    
    # 获取MoE头
    moe_head = None
    for module in model.modules():
        if hasattr(module, 'prototypes') and hasattr(module, 'visualize_prototypes'):
            moe_head = module
            break
    
    if moe_head is None:
        print("未找到MoE头，无法进行可视化")
        return
    
    # 运行评估
    print("开始评估...")
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # 前向传播
            results = model(data)
            
            # 每处理一定数量的batch后进行一次可视化
            if batch_idx % 10 == 0:
                print(f"处理batch {batch_idx}, 生成可视化...")
                
                # 可视化原型分布
                vis_files = moe_head.visualize_prototypes(
                    epoch=epoch,
                    class_names=class_names,
                    method='tsne'  # 或 'pca'
                )
                
                # 获取原型统计信息
                stats = moe_head.get_prototype_statistics()
                
                # 保存统计信息到JSON文件
                stats_file = os.path.join(save_dir, f'prototype_stats_epoch_{epoch}_batch_{batch_idx}.json')
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"可视化文件已保存到: {list(vis_files.values())}")
                print(f"统计信息已保存到: {stats_file}")
                
                # 打印关键统计信息
                print(f"类内多样性: {stats['intra_diversity']:.4f}")
                print(f"类间分离性: {stats['inter_separation']:.4f}")
                print(f"平均原型范数: {stats['avg_proto_norm']:.4f}")
    
    print("评估完成！")


def create_prototype_analysis_report(model, 
                                   class_names: Optional[List[str]] = None,
                                   save_dir: str = "prototype_analysis"):
    """创建原型分析报告
    
    Args:
        model: 训练好的模型
        class_names: 类别名称列表
        save_dir: 报告保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取MoE头
    moe_head = None
    for module in model.modules():
        if hasattr(module, 'prototypes') and hasattr(module, 'visualize_prototypes'):
            moe_head = module
            break
    
    if moe_head is None:
        print("未找到MoE头，无法进行分析")
        return
    
    # 生成可视化
    vis_files = moe_head.visualize_prototypes(
        epoch=0,
        class_names=class_names,
        method='tsne'
    )
    
    # 获取统计信息
    stats = moe_head.get_prototype_statistics()
    
    # 创建分析报告
    report = f"""
# 原型分布分析报告

## 统计信息
- 类内多样性: {stats['intra_diversity']:.4f}
- 类间分离性: {stats['inter_separation']:.4f}
- 平均原型范数: {stats['avg_proto_norm']:.4f}
- 原型范数标准差: {stats['std_proto_norm']:.4f}
- 最小原型范数: {stats['min_proto_norm']:.4f}
- 最大原型范数: {stats['max_proto_norm']:.4f}

## 可视化文件
"""
    
    for vis_type, file_path in vis_files.items():
        report += f"- {vis_type}: {file_path}\n"
    
    # 保存报告
    report_file = os.path.join(save_dir, "prototype_analysis_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存统计信息
    stats_file = os.path.join(save_dir, "prototype_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"分析报告已保存到: {report_file}")
    print(f"统计信息已保存到: {stats_file}")


def plot_prototype_evolution(statistics_history: List[Dict], 
                           save_dir: str = "evolution_plots"):
    """绘制原型演化过程
    
    Args:
        statistics_history: 历史统计信息列表
        save_dir: 图片保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(len(statistics_history)))
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 类内多样性演化
    intra_diversity = [stats['intra_diversity'] for stats in statistics_history]
    axes[0, 0].plot(epochs, intra_diversity, 'b-', marker='o')
    axes[0, 0].set_title('Intra-class Diversity Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Diversity')
    axes[0, 0].grid(True)
    
    # 2. 类间分离性演化
    inter_separation = [stats['inter_separation'] for stats in statistics_history]
    axes[0, 1].plot(epochs, inter_separation, 'r-', marker='s')
    axes[0, 1].set_title('Inter-class Separation Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Separation')
    axes[0, 1].grid(True)
    
    # 3. 原型范数演化
    avg_norms = [stats['avg_proto_norm'] for stats in statistics_history]
    std_norms = [stats['std_proto_norm'] for stats in statistics_history]
    axes[1, 0].plot(epochs, avg_norms, 'g-', marker='^', label='Average')
    axes[1, 0].fill_between(epochs, 
                           [avg - std for avg, std in zip(avg_norms, std_norms)],
                           [avg + std for avg, std in zip(avg_norms, std_norms)],
                           alpha=0.3, label='±1 Std')
    axes[1, 0].set_title('Prototype Norm Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 相似度演化
    intra_sim = [stats['avg_intra_similarity'] for stats in statistics_history]
    inter_sim = [stats['avg_inter_similarity'] for stats in statistics_history]
    axes[1, 1].plot(epochs, intra_sim, 'b-', marker='o', label='Intra-class')
    axes[1, 1].plot(epochs, inter_sim, 'r-', marker='s', label='Inter-class')
    axes[1, 1].set_title('Similarity Evolution')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Similarity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "prototype_evolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"演化图已保存到: {save_path}")


# 使用示例
if __name__ == "__main__":
    # 示例：如何在训练循环中使用可视化
    
    # 1. 在模型配置中启用可视化
    model_config = {
        'enable_visualization': True,
        'vis_save_dir': 'prototype_vis'
    }
    
    # 2. 在评估过程中调用可视化
    # evaluate_with_prototype_visualization(model, val_loader, epoch=current_epoch)
    
    # 3. 创建分析报告
    # create_prototype_analysis_report(model, class_names=dataset.class_names)
    
    # 4. 绘制演化过程
    # plot_prototype_evolution(statistics_history)
    
    print("原型可视化功能已准备就绪！")
    print("请在训练/评估脚本中调用相应的函数。") 
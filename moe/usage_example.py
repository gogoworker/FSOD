#!/usr/bin/env python3
"""
原型可视化功能使用示例
"""

import torch
import os
from typing import List, Dict
import json

# 假设你已经有了训练好的模型和数据加载器
def example_usage():
    """使用示例"""
    
    # 1. 在模型配置中启用可视化
    model_config = {
        'num_classes': 80,
        'in_channels': 2048,
        'num_experts': 4,
        'feat_dim': 2048,
        'enable_visualization': True,  # 启用可视化
        'vis_save_dir': 'prototype_vis'  # 可视化结果保存目录
    }
    
    # 2. 在训练循环中定期可视化
    def train_with_visualization(model, train_loader, val_loader, num_epochs=100):
        """带可视化的训练循环"""
        
        # 获取MoE头
        moe_head = None
        for module in model.modules():
            if hasattr(module, 'prototypes') and hasattr(module, 'visualize_prototypes'):
                moe_head = module
                break
        
        if moe_head is None:
            print("未找到MoE头，无法进行可视化")
            return
        
        # 类别名称（根据你的数据集调整）
        class_names = [f'class_{i}' for i in range(80)]
        
        # 存储统计历史
        statistics_history = []
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs}")
            
            # 训练阶段
            model.train()
            for batch_idx, data in enumerate(train_loader):
                # 训练代码...
                pass
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    # 验证代码...
                    pass
            
            # 每10个epoch进行一次可视化
            if epoch % 10 == 0:
                print(f"生成第{epoch}个epoch的可视化...")
                
                # 可视化原型分布
                vis_files = moe_head.visualize_prototypes(
                    epoch=epoch,
                    class_names=class_names,
                    method='tsne'  # 或 'pca'
                )
                
                # 获取并保存统计信息
                stats = moe_head.get_prototype_statistics()
                statistics_history.append(stats)
                
                # 保存统计信息
                stats_file = f'prototype_vis/stats_epoch_{epoch}.json'
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"可视化文件: {list(vis_files.values())}")
                print(f"统计信息: {stats_file}")
                
                # 打印关键指标
                print(f"类内多样性: {stats['intra_diversity']:.4f}")
                print(f"类间分离性: {stats['inter_separation']:.4f}")
                print(f"平均原型范数: {stats['avg_proto_norm']:.4f}")
        
        # 训练结束后绘制演化图
        from evaluation_with_visualization import plot_prototype_evolution
        plot_prototype_evolution(statistics_history, 'evolution_plots')
    
    # 3. 在评估过程中使用可视化
    def evaluate_with_visualization(model, test_loader, class_names=None):
        """带可视化的评估过程"""
        
        # 获取MoE头
        moe_head = None
        for module in model.modules():
            if hasattr(module, 'prototypes') and hasattr(module, 'visualize_prototypes'):
                moe_head = module
                break
        
        if moe_head is None:
            print("未找到MoE头，无法进行可视化")
            return
        
        model.eval()
        
        # 收集所有batch的门控权重
        all_gate_weights = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # 前向传播
                results = model(data)
                
                # 收集门控权重（如果可用）
                if hasattr(moe_head, 'last_gate_weights'):
                    all_gate_weights.append(moe_head.last_gate_weights.cpu())
                
                # 每处理一定数量的batch后可视化
                if batch_idx % 20 == 0:
                    print(f"处理batch {batch_idx}, 生成可视化...")
                    
                    # 可视化原型分布
                    vis_files = moe_head.visualize_prototypes(
                        epoch=0,  # 评估时epoch为0
                        class_names=class_names,
                        method='tsne'
                    )
                    
                    # 获取统计信息
                    stats = moe_head.get_prototype_statistics()
                    
                    print(f"可视化文件: {list(vis_files.values())}")
                    print(f"统计信息: {stats}")
        
        # 分析门控权重分布
        if all_gate_weights:
            all_gates = torch.cat(all_gate_weights, dim=0)
            print(f"收集到 {len(all_gates)} 个样本的门控权重")
            
            # 计算门控权重的统计信息
            gate_stats = {
                'mean_gate_weight': all_gates.mean().item(),
                'std_gate_weight': all_gates.std().item(),
                'max_gate_weight': all_gates.max().item(),
                'min_gate_weight': all_gates.min().item(),
                'sparsity': (all_gates < 0.1).float().mean().item()  # 稀疏性
            }
            
            print(f"门控权重统计: {gate_stats}")
    
    # 4. 创建详细的分析报告
    def create_detailed_analysis(model, class_names=None):
        """创建详细的原型分析报告"""
        
        from evaluation_with_visualization import create_prototype_analysis_report
        create_prototype_analysis_report(model, class_names, 'detailed_analysis')
    
    return {
        'train_with_visualization': train_with_visualization,
        'evaluate_with_visualization': evaluate_with_visualization,
        'create_detailed_analysis': create_detailed_analysis
    }


# 在训练脚本中的集成示例
def integrate_with_training_script():
    """在训练脚本中集成的示例"""
    
    # 假设你已经有了模型和数据加载器
    # model = your_model
    # train_loader = your_train_loader
    # val_loader = your_val_loader
    # test_loader = your_test_loader
    
    # 1. 在模型初始化时启用可视化
    # model_config = {
    #     'enable_visualization': True,
    #     'vis_save_dir': 'prototype_vis'
    # }
    
    # 2. 在训练循环中调用可视化
    # functions = example_usage()
    # functions['train_with_visualization'](model, train_loader, val_loader)
    
    # 3. 在评估时调用可视化
    # functions['evaluate_with_visualization'](model, test_loader)
    
    # 4. 创建分析报告
    # functions['create_detailed_analysis'](model)
    
    print("集成示例完成！")


if __name__ == "__main__":
    print("原型可视化功能使用示例")
    print("请根据你的具体需求调整代码")
    
    # 运行示例
    example_usage()
    integrate_with_training_script() 
import torch
import torch.nn as nn
import torch.nn.functional as F

"""《Poseidon: A ViT-based Architecture for Multi-Frame Pose Estimation with Adaptive Frame Weighting and Multi-Scale Feature Fusion》 Arxiv 2025
人体姿势估计是计算机视觉中一项重要的任务，涉及检测和定位图像和视频中的人体关节。虽然单帧姿势估计取得了重大进展，但它往往无法捕捉时间动态以理解复杂、连续的运动。
我们提出了 Poseidon，这是一种新颖的多帧姿势估计架构，它通过集成时间信息来扩展 ViTPose 模型，以提高准确性和鲁棒性，以解决这些限制。
Poseidon 引入了关键创新：
(1) 自适应帧加权 (AFW) 机制，可根据帧的相关性动态地对帧进行优先级排序，确保模型专注于最具信息量的数据；
(2) 多尺度特征融合 (MSFF) 模块，可聚合来自不同骨干层的特征以捕获细粒度细节和高级语义；
(3) 交叉注意模块，用于在中心帧和上下文帧之间进行有效的信息交换，增强模型的时间连贯性。
所提出的架构提高了复杂视频场景中的性能，并提供了适合实际应用的可扩展性和计算效率。我们的方法在 PoseTrack21 和 PoseTrack18 数据集上实现了最先进的性能，分别实现了 88.3 和 87.8 的 mAP 分数，优于现有方法。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)

class AdaptiveFrameWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames):
        super(AdaptiveFrameWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, height, width]
        batch_size, num_frames, embed_dim, height, width = x.shape

        x_reshaped = x.view(batch_size * num_frames, embed_dim, height, width)

        # 通过帧质量估计器计算每一帧的质量得分
        quality_scores = self.frame_quality_estimator(x_reshaped).view(batch_size, num_frames)

        weights = F.softmax(quality_scores, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        weighted_x = x * weights

        return weighted_x, weights.squeeze()


if __name__ == '__main__':
    embed_dim = 32  # 示例嵌入维度
    num_frames = 10  # 示例帧数
    block = AdaptiveFrameWeighting(embed_dim, num_frames).to('cuda')

    #  [batch_size, num_frames, embed_dim, height, width]
    batch_size = 2
    height = 16
    width = 16
    input_tensor = torch.rand(batch_size, num_frames, embed_dim, height, width).to('cuda')

    output, weights = block(input_tensor)

    print(f"输入张量的形状: {input_tensor.size()}")
    print(f"输出张量的形状: {output.size()}")
    print(f"权重的形状: {weights.size()}")
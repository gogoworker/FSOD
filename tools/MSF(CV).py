import torch
import torch.nn as nn


"""《FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation》AAAI 2025
现有的小样本医学图像分割 （FSMIS） 模型无法解决医学成像中的一个实际问题：由不同成像技术引起的领域转移，这限制了对当前 FSMIS 任务的适用性。
为了克服这一限制，我们专注于跨域小样本医学图像分割 （CD-FSMIS） 任务，旨在开发一种通用模型，能够适应更广泛的医学图像分割场景，使用来自新目标域的有限标记数据。
受不同域之间频域相似性特征的启发，我们提出了一种频率感知匹配网络 （FAMNet），它包括两个关键组件：频率感知匹配 （FAM） 模块和多频谱融合 （MSF） 模块。
FAM 模块在元学习阶段解决了两个问题：1） 由于器官和病变的外观不同，由固有的支持-查询偏差引起的域内差异，以及 2） 由不同的医学成像技术引起的域间差异。
此外，我们设计了一个 MSF 模块来集成由 FAM 模块解耦的不同频率特征，并进一步减轻域间方差对模型分割性能的影响。
结合这两个模块，我们的 FAMNet 在三个跨域数据集上超越了现有的 FSMIS 模型和跨域 Few-shot 语义分割模型，在 CD-FSMIS 任务中实现了最先进的性能。
"""

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):
        B, C, N = Q_feature.shape

        Q_feature = Q_feature.permute(0, 2, 1)
        K_feature = K_feature.permute(0, 2, 1)

        Q = self.query(Q_feature)  # shape: [B, N, C]
        K = self.key(K_feature)  # shape: [B, N, C]
        V = self.value(K_feature)  # shape: [B, N, C]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)  # shape: [B, N, N]

        attended_features = torch.matmul(attention_weights, V)  # shape: [B, N, C]
        attended_features = attended_features.permute(0, 2, 1)

        return attended_features


class MSF(nn.Module):  # Attention-based Feature Fusion Module
    def __init__(self, feature_dim):
        super(MSF, self).__init__()
        self.CA1 = CrossAttentionFusion(feature_dim)
        self.CA2 = CrossAttentionFusion(feature_dim)
        self.relu = nn.ReLU()

    def forward(self, low, mid, high):
        low_new = self.CA1(mid, low)
        high_new = self.CA2(mid, high)
        fused_features = self.relu(low_new + mid + high_new)
        return fused_features


if __name__ == '__main__':
    feature_dim = 64

    block = MSF(feature_dim)

    #  (batch_size, channels, feature_length)
    low = torch.rand(1, feature_dim, 128)
    mid = torch.rand(1, feature_dim, 128)
    high = torch.rand(1, feature_dim, 128)

    output = block(low, mid, high)

    print(f"Low input size: {low.size()}")
    print(f"Mid input size: {mid.size()}")
    print(f"High input size: {high.size()}")
    print(f"Output size: {output.size()}")
import torch
import torch.nn as nn


"""《ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation》 WACV 2024
大规模预训练视觉语言模型 (VLM)（例如 CLIP）已在零样本分类中表现出色，例如在 ImageNet 上无需查看任何示例即可实现 76.3% 的 top-1 准确率，这为许多没有标记数据的任务带来了潜在优势。
然而，在将 CLIP 应用于下游目标域时，视觉和文本域差距以及跨模态错位的存在会极大地影响模型性能。为了解决这些挑战，我们提出了 ReCLIP，一种用于视觉语言模型的新型无源域自适应方法，它不需要任何源数据或目标标记数据。
ReCLIP 首先学习投影空间以减轻未对齐的视觉文本嵌入并学习伪标签，然后使用伪标签部署跨模态自训练，以迭代地更新视觉和文本编码器、细化标签并减少域差距和错位。
通过大量实验，我们证明 ReCLIP 的表现显著优于所有基线，并在 22 个图像分类基准上将 CLIP 的平均准确率从 69.83% 提高到 74.94%。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)


class UpdateProjection(nn.Module):
    def __init__(self, feature_dim=768, cut_dim=768, dtype=torch.float32):
        """
        SVD投影矩阵生成器
        Args:
            feature_dim (int): 输入特征维度（默认CLIP特征维度768）
            cut_dim (int): 保留的投影维度（默认768，若类别数>500可设为250）
            dtype: 计算精度（默认float32以保持数值稳定）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.cut_dim = cut_dim
        self.dtype = dtype
        self.projection_matrix = nn.Parameter(
            torch.eye(feature_dim, dtype=dtype),
            requires_grad=False
        )  # 初始化为单位矩阵

    def forward(self, classification_weight):
        """
        输入分类权重矩阵，更新投影矩阵
        Args:
            classification_weight: [feature_dim, num_classes] 的文本嵌入矩阵
        Returns:
            projection_matrix: [feature_dim, feature_dim] 的投影矩阵
        """
        # SVD分解
        U, S, V = torch.svd(classification_weight.to(self.dtype))  # U: [feature_dim, num_classes]

        # 移除主成分e1（U[:,0]），保留e2到e_cut_dim
        if self.cut_dim > U.size(1):
            self.cut_dim = U.size(1)  # 避免维度溢出

        U_reduce = U[:, 1:self.cut_dim]  # 移除第一主成分

        # 计算投影矩阵 P = U'U'^T
        self.projection_matrix.data = (U_reduce @ U_reduce.t()).to(self.dtype)
        return self.projection_matrix

    def get_projection_matrix(self):
        """返回当前投影矩阵"""
        return self.projection_matrix

    def project_features(self, features):
        """
        将特征投影到对齐子空间
        Args:
            features: [batch_size, feature_dim] 的视觉/文本特征
        Returns:
            projected_features: 投影后的特征
        """
        return features @ self.projection_matrix.to(features.device)


if __name__ == '__main__':
    feature_dim = 768
    num_classes = 10  # 假设有10个类别
    batch_size = 4

    projector = UpdateProjection(feature_dim=feature_dim, cut_dim=250).to('cuda')

    # 生成模拟输入 (分类权重矩阵)
    classification_weight = torch.randn(feature_dim, num_classes).to('cuda')  # [768, 10]
    print("输入分类权重矩阵形状:", classification_weight.shape)

    projection_matrix = projector(classification_weight)
    print("\n投影矩阵形状:", projection_matrix.shape)

    # # 测试特征投影
    # visual_features = torch.randn(batch_size, feature_dim).to('cuda')
    # projected_features = projector.project_features(visual_features)
    # print("\n投影前特征形状:", visual_features.shape)
    # print("投影后特征形状:", projected_features.shape)
    #
    # # 验证投影矩阵性质
    # print("\n投影矩阵范数:", torch.norm(projection_matrix).item())
    # print("投影矩阵是否对称:", torch.allclose(projection_matrix, projection_matrix.t(), atol=1e-6))
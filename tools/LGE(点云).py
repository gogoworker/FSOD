import torch
import torch.nn as nn


"""《Self-Attentive Local Aggregation Learning With Prototype Guided Regularization for Point Cloud Semantic Segmentation of High-Speed Railways》IEEE Transactions on Intelligent Transportation Systems, 2023
铁路基础设施的点云语义分割是建立铁路数字孪生的重要一步。与依赖手工制作特征的传统方法相比，基于深度学习的方法在这一领域显示出巨大的潜力。然而，基于深度学习的铁路点云方法仍然面临着需要解决的典型挑战。
在这方面，我们提出了一种名为 SALAProNet 的新型学习框架，它由一组有效而简洁的模块化解决方案组成。解决的第一个挑战是铁路点云的数据规模庞大，由于内存限制，直接处理大规模点云变得困难。
为了解决这个问题，我们在网络中采用了高效的随机采样，并提出了基于注意力机制的自注意力聚合 (SAA) 模块，以大大扩展接受场，覆盖未采样的点并成功地在高维特征空间中保留信息。
第二个挑战是细粒度分割，我们提出了局部几何嵌入 (LGE) 模块来嵌入局部几何。借助 SAA 提供的上下文信息，网络可以对铁路基础设施进行细粒度分割。
第三个挑战是网络的泛化能力不足，我们提出了一种原型引导正则化 (PGR) 方法来指导网络在不同建设标准的铁路之间分割点云。这种方法增强了网络的可解释性并提高了其泛化能力。
我们通过对不同数据集的实验验证了我们提出的框架，并且它的表现优于最先进的方法。
"""


class LocalGeometryEmbedding(nn.Module):
    def __init__(self):
        super(LocalGeometryEmbedding, self).__init__()

    def forward(self, input_cloud):
        # 点云数据形状为 [B, N, 3]，其中 B 是 batch size，N 是点的数量，3 是坐标维度
        B, N, _ = input_cloud.shape

        # 假设取每个点的最近 10 个邻居点，可根据需要调整
        K = 10

        distances = torch.cdist(input_cloud, input_cloud)
        _, knn_indices = torch.topk(distances, k=K + 1, dim=-1, largest=False)
        # 去掉自身，取最近的 K 个邻居点索引
        knn_indices = knn_indices[:, :, 1:]

        # 初始化一个列表来存储每个点的局部几何特征
        local_geometry_features = []
        for i in range(B):
            point_local_features = []
            for j in range(N):
                # 获取当前点的邻居点坐标
                neighbors = input_cloud[i, knn_indices[i, j]]
                center_point = input_cloud[i, j]

                # 计算相对位置关系和欧几里得距离
                relative_positions = neighbors - center_point
                distances = torch.norm(relative_positions, dim=1)

                # 计算线性、平面性和散射性特征
                covariance_matrix = torch.cov(neighbors.T)
                eigenvalues = torch.linalg.eigvalsh(covariance_matrix)
                eigenvalues = eigenvalues[torch.argsort(eigenvalues, descending=True)]
                lambda_1, lambda_2, lambda_3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
                linearity = (lambda_1 - lambda_2) / lambda_1
                planarity = (lambda_2 - lambda_3) / lambda_1
                scattering = lambda_3 / lambda_1

                # 将当前点和邻居点的信息拼接
                features = torch.cat([
                    center_point,
                    neighbors.view(-1),
                    relative_positions.view(-1),
                    distances,
                    torch.tensor([linearity, planarity, scattering], device=input_cloud.device)
                ])

                mlp = nn.Sequential(
                    nn.Linear(features.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 3),
                    nn.ReLU()
                )
                point_local_features.append(mlp(features))

            # 将当前 batch 中所有点的局部特征拼接起来
            local_geometry_features.append(torch.stack(point_local_features))

        # 将所有 batch 的局部几何特征拼接起来，最终形状为 [B, N, D]
        return torch.stack(local_geometry_features)


if __name__ == '__main__':
    batch_size = 8
    num_points = 1024
    input_data = torch.rand(batch_size, num_points, 3)

    block = LocalGeometryEmbedding()

    output = block(input_data)

    # 打印输入和输出的尺寸
    print("Input size:", input_data.size())
    print("Output size:", output.size())
import torch
import torch.nn as nn


"""《PointeNet: A lightweight framework for effective and efficient point cloud analysis》
点云分析的传统观点主要探索 3D 几何。它通常通过在编码器中引入复杂的可学习几何提取器或通过加深具有重复块的网络来实现。然而，这些方法包含大量可学习参数，导致大量计算成本并给 CPU/GPU 带来内存负担。
此外，它们主要针对对象级点云分类和分割任务，对关键场景级应用（如自动驾驶）的扩展有限。为此，我们介绍了PointeNet，一种为点云分析设计的高效网络。
PointeNet 以其轻量级架构、低训练成本和即插即用功能而著称，同时还能有效捕捉代表性特征。该网络由多元几何编码 (MGE) 模块和可选的距离感知语义增强 (DSE) 模块组成。 
MGE 采用采样、分组、池化和多元几何聚合操作，轻量级捕获和自适应聚合多元几何特征，提供 3D 几何的全面描述。DSE 专为现实世界的自动驾驶场景而设计，可增强点云的语义感知，尤其是对于远距离点。
我们的方法通过与分类/分割头无缝集成或嵌入到现成的3D 物体检测网络中展示了灵活性，以最小的成本实现了显着的性能提升。
在包括 ModelNet40、ScanObjectNN、ShapeNetPart 和场景级数据集 KITTI 在内的对象级数据集上进行的大量实验证明了 PointeNet 在点云分析方面优于最新方法的性能。
代码可在https://github.com/lipeng-gu/PointeNet上公开获取。
"""


def furthest_point_sample(x, num_samples):
    """
    通过最远点采样从输入点云中选择指定数量的点

    参数:
    - x: 输入的点云数据，形状为 (batch_size, num_points, 3)
    - num_samples: 需要采样的点数

    返回:
    - idx: 选中的点的索引，形状为 (batch_size, num_samples)
    """
    batch_size, num_points, _ = x.size()

    # 初始化距离为非常大的值
    dist = torch.ones(batch_size, num_points).to(x.device) * 1e10
    # 随机选择一个起始点，选择批次中的第一个点作为起点
    farthest_pts = torch.randint(0, num_points, (batch_size, 1)).to(x.device)
    # 记录返回的采样点的索引
    idx = farthest_pts

    # 通过迭代选择最远点
    for _ in range(num_samples - 1):
        # 获取当前采样点的坐标
        current_pts = x.gather(1, farthest_pts.unsqueeze(-1).repeat(1, 1, 3))  # (batch_size, 1, 3)
        # 计算当前点到所有其他点的距离
        dist_batch = torch.norm(x - current_pts, dim=-1)  # (batch_size, num_points)

        # 更新距离
        dist = torch.min(dist, dist_batch)  # 每个点到已有点集中最近的距离
        # 找到当前距离最远的点
        farthest_pts = torch.argmax(dist, dim=-1, keepdim=True)  # (batch_size, 1)

        # 将最远点加入已选的点集
        idx = torch.cat([idx, farthest_pts], dim=1)  # (batch_size, num_samples)

    return idx

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.group_num).long() # source
        fps_idx = furthest_point_sample(xyz.contiguous(), self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


class MAA(nn.Module):
    def __init__(self, in_channels, group_num, features_num):
        super(MAA, self).__init__()
        self.in_channels = in_channels
        self.group_num = group_num
        self.features_num = features_num
        self.alpha_list = []
        self.beta_list = []
        for i in range(self.features_num):
            self.alpha_list.append(nn.Parameter(torch.ones([1, self.in_channels, 1])))
            self.beta_list.append(nn.Parameter(torch.zeros([1, self.in_channels, 1])))
        self.alpha_list = nn.ParameterList(self.alpha_list)
        self.beta_list = nn.ParameterList(self.beta_list)

        self.linear = Linear1Layer(self.in_channels, self.in_channels, bias=False)

    def forward(self, features_list):
        assert len(features_list) == self.features_num
        for i in range(self.features_num):
            features_list[i] = self.alpha_list[i] * features_list[i] + self.beta_list[i]

        features_list = torch.stack(features_list).sum(dim=0)
        features_list = self.linear(features_list)
        return features_list


# Local Geometry Aggregation
# Local Geometry Aggregation
class MLGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, surface_points, group_num):
        super().__init__()
        self.surface_points = surface_points
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

        # 修改特征扩展的计算
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        else:
            expand = 1

        # 修改 linear1 的输入通道数，确保与拼接后的特征维度匹配
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)

        # 残差块
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)

        self.Pooling = Pooling()

        # 几何特征嵌入
        self.norm_embedding = Linear1Layer(3, out_dim, bias=False)
        self.curv_embedding = Linear1Layer(3, out_dim, bias=False)
        self.MAA = MAA(out_dim, group_num, 3)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # Surface Normal and Curvature
        if self.surface_points is not None:
            est_normal, est_curvature = get_local_geo(knn_xyz[..., :self.surface_points, :])
            est_normal = self.norm_embedding(est_normal.permute(0, 2, 1))
            est_curvature = self.curv_embedding(est_curvature.permute(0, 2, 1))

        # Normalization
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)  # [B, 3, G, K]
        knn_x = knn_x.permute(0, 3, 1, 2)  # [B, C*2, G, K]

        # 重要：调整维度以匹配 linear1 的输入要求
        knn_x = self.linear1(knn_x.reshape(B, -1, G * K))  # [B, out_dim, G*K]
        knn_x = knn_x.reshape(B, -1, G, K)  # [B, out_dim, G, K]

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        # Pooling
        knn_x_w = self.Pooling(knn_x_w)

        if self.surface_points is not None:
            knn_x_w = self.MAA([knn_x_w, est_normal, est_curvature])
        return knn_x_w
# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)



# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels / 2),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels / 2)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels / 2), out_channels=in_channels,
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


def get_local_geo(knn_xyz):
    # Surface Normal and Curvature
    centroid = knn_xyz.mean(dim=2, keepdim=True)
    matrix1 = torch.matmul(centroid.permute(0, 1, 3, 2), centroid)
    matrix2 = torch.matmul(knn_xyz.permute(0, 1, 3, 2), knn_xyz) / knn_xyz.shape[2]
    matrix = matrix1 - matrix2
    u, s, v = torch.svd(matrix)
    est_normal = v[:, :, :, 2]
    est_normal = est_normal / torch.norm(est_normal, p=2, dim=-1, keepdim=True)
    est_curvature = s + 1e-9
    est_curvature = est_curvature / est_curvature.sum(dim=-1, keepdim=True)

    return est_normal, est_curvature


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

        # 添加一个线性层来调整维度
        self.adjust_dim = nn.Conv2d(3 * 2 * (out_dim // (3 * 2)), out_dim, 1)

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        C = knn_x.size(1)  # 获取输入特征的通道数

        # 计算特征维度
        feat_dim = max(1, self.out_dim // (self.in_dim * 2))

        # 生成位置编码
        feat_range = torch.arange(feat_dim, device=knn_xyz.device).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)

        # 扩展维度
        dim_embed = dim_embed.view(1, 1, 1, 1, -1)
        knn_xyz = knn_xyz.unsqueeze(-1)

        # 计算位置编码
        div_embed = torch.div(self.beta * knn_xyz, dim_embed)
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)

        # 合并编码
        position_embed = torch.cat([sin_embed, cos_embed], dim=-1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.reshape(B, -1, G, K)

        # 使用卷积层调整维度
        position_embed = self.adjust_dim(position_embed)

        # 确保维度匹配
        assert position_embed.size(
            1) == C, f"Position embedding channels ({position_embed.size(1)}) don't match input channels ({C})"

        # 应用位置编码
        knn_x_w = knn_x + position_embed
        knn_x_w = knn_x_w * position_embed

        return knn_x_w

# Parametric Encoder
class MGE(nn.Module):
    def __init__(self, in_channels, input_points, num_stages, embed_dim, reducers, k_neighbors, k_neighbors_list,
                 alpha, beta, MLGA_block, dim_expansion):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.MLGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // reducers[i]
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.MLGA_list.append(MLGA(out_dim, self.alpha, self.beta, MLGA_block[i], dim_expansion[i],
                                     surface_points=k_neighbors_list[i], group_num=group_num))

    def forward(self, xyz, x):

        # Raw-point Embedding
        # pdb.set_trace()
        x = self.raw_point_embed(x)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            x = self.MLGA_list[i](xyz, lc_x, knn_xyz, knn_x)

            x_list.append(x)
            xyz_list.append(xyz)

        return xyz_list, x_list



if __name__ == '__main__':
    # 设置输入维度
    B = 8  # 批量大小
    N = 1024  # 点云数据中的点数
    C = 3  # 每个点的特征维度（如xyz坐标）
    D = 3  # 输入特征的维度

    # 创建模拟输入数据
    xyz = torch.randn(B, N, C).cuda()  # 点云坐标数据
    x = torch.randn(B, D, N).cuda()    # 点云特征数据

    # 创建 MGE 模型参数
    input_points = 1024
    num_stages = 3
    embed_dim = 64
    reducers = [2, 2, 2]
    k_neighbors = 16
    k_neighbors_list = [16, 16, 16]
    alpha, beta = 1.0, 1.0
    MLGA_block = [2, 2, 2]
    dim_expansion = [1, 1, 1]  # 保持维度不变

    # 初始化模型
    block = MGE(
        in_channels=C,
        input_points=input_points,
        num_stages=num_stages,
        embed_dim=embed_dim,
        reducers=reducers,
        k_neighbors=k_neighbors,
        k_neighbors_list=k_neighbors_list,
        alpha=alpha,
        beta=beta,
        MLGA_block=MLGA_block,
        dim_expansion=dim_expansion
    ).to('cuda')

    # 前向传播
    xyz_list, x_list = block(xyz, x)

    # 打印维度信息
    print("Input xyz size:", xyz.size())
    print("Input x size:", x.size())
    print("\nOutput sizes:")
    print("xyz_list sizes:", [xyz_.size() for xyz_ in xyz_list])
    print("x_list sizes:", [x_.size() for x_ in x_list])
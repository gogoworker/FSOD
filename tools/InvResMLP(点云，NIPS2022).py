import torch
import torch.nn as nn


"""《Pointnext: Revisiting pointnet++ with improved training and scaling strategies》NIPS 2022
PointNet++ 是用于点云理解的最具影响力的神经架构之一。尽管 PointNet++ 的准确率已被 PointMLP 和 Point Transformer 等近期网络大幅超越，但我们发现性能提升的很大一部分归因于改进的训练策略，即数据增强和优化技术，以及增加的模型大小，而不是架构创新。
因此，PointNet++ 的全部潜力尚未得到发掘。在这项工作中，我们通过系统研究模型训练和扩展策略重新审视经典的 PointNet++，并提出两大贡献。首先，我们提出了一套改进的训练策略，可显着提高 PointNet++ 的性能。
例如，我们表明，在架构没有任何变化的情况下，PointNet++ 在 ScanObjectNN 对象分类上的整体准确率 (OA) 可以从 77.9% 提高到 86.1%，甚至优于最先进的 PointMLP。
其次，我们在 PointNet++ 中引入了倒置残差瓶颈设计和可分离 MLP，以实现高效且有效的模型扩展，并提出了 PointNets 的下一个版本 PointNeXt。PointNeXt 可以灵活扩展，并且在 3D 分类和分割任务上均优于最先进的方法。
对于分类，PointNeXt 在 ScanObjectNN 上的整体准确率达到 87.7，比 PointMLP 高出 2.3%，同时推理速度提高了 10 倍。对于语义分割，PointNeXt 在 S3DIS（6 倍交叉验证）上以 74.9% 的平均 IoU 建立了新的最先进的性能，优于最近的 Point Transformer。
代码和模型可在 https://github.com/guochengqian/pointnext 上找到。
"""

def build_mlp(in_channel, channel_list, dim=2, bias=False, drop_last_act=False,
              drop_last_norm_act=False, dropout=False):
    """
    构造基于n dim 1x1卷积的mlp
    :param in_channel: <int> 特征维度的输入值
    :param channel_list: <list[int]> mlp各层的输出通道维度数
    :param dim: <int> 维度，1或2
    :param bias: <bool> 卷积层是否添加bias，一般BN前的卷积层不使用bias
    :param drop_last_act: <bool> 是否去除最后一层激活函数
    :param drop_last_norm_act: <bool> 是否去除最后一层标准化层和激活函数
    :param dropout: <bool> 是否添加dropout层
    :return: <torch.nn.ModuleList[torch.nn.Sequential]>
    """
    # 解析参数获取相应卷积层、归一化层、激活函数
    if dim == 1:
        Conv = nn.Conv1d
        NORM = nn.BatchNorm1d
    else:
        Conv = nn.Conv2d
        NORM = nn.BatchNorm2d
    ACT = nn.ReLU

    # 根据通道数构建mlp
    mlp = []
    for i, channel in enumerate(channel_list):
        if dropout and i > 0:
            mlp.append(nn.Dropout(0.5, inplace=False))
        # 每层为conv-bn-relu
        mlp.append(Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))
        if i < len(channel_list) - 1:
            in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]
    elif drop_last_norm_act:
        mlp = mlp[:-2]
        mlp[-1] = Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=True)

    return nn.Sequential(*mlp)


def coordinate_distance(src, dst):
    """
    计算两个点集的各点间距
    !!!使用半精度运算或自动混合精度时[不要]使用化简的方法，否则会出现严重的浮点误差
    :param src: <torch.Tensor> (B, M, C) C为坐标
    :param dst: <torch.Tensor> (B, N, C) C为坐标
    :return: <torch.Tensor> (B, M, N)
    """
    B, M, _ = src.shape
    _, N, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, M, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)

    # dist = torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)).pow(2), dim=-1)
    return dist


def index_points(points, idx):
    """
    跟据采样点索引获取其原始点云xyz坐标等信息
    :param points: <torch.Tensor> (B, N, 3+) 原始点云
    :param idx: <torch.Tensor> (B, S)/(B, S, G) 采样点索引，S为采样点数量，G为每个采样点grouping的点数
    :return: <torch.Tensor> (B, S, 3+)/(B, S, G, 3+) 获取了原始点云信息的采样点
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_hybrid(radius, nsample, xyz, new_xyz):
    """
    基于采样点进行KNN与ball query混合的grouping
    :param radius: <float> grouping半径
    :param nsample: <int> group内点云数量
    :param xyz: <torch.Tensor> (B, N, 3) 原始点云
    :param new_xyz: <torch.Tensor> (B, S, 3) 采样点
    :return: <torch.Tensor> (B, S, nsample) 每个采样点grouping的点云索引
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    dist = coordinate_distance(new_xyz, xyz)  # 每个采样点与其他点的距离的平方
    dist, group_idx = torch.topk(dist, k=nsample, dim=-1, largest=False)  # 基于距离选择最近的作为采样点
    radius = radius ** 2
    mask = dist > radius  # 距离较远的点替换为距离最近的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    group_idx[mask] = group_first[mask]

    return group_idx




class LocalAggregation(nn.Module):
    """
    局部特征提取
    包含一个单尺度G-P过程，每一个点都作为采样点进行group以聚合局部特征，无下采样过程
    """

    def __init__(self,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        """
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp = build_mlp(in_channel=in_channel + coor_dim, channel_list=[in_channel], dim=2)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_fea: <torch.Tensor> (B, D, N) 局部特征聚合后的特征
        """
        # (B, C, N) -> (B, N, C)
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)
        bs, npoint, _ = points_coor.shape

        '''G 分组'''
        # 每个group的点云索引 (B, N, K)
        group_idx = query_hybrid(self.radius, self.nsample, points_coor[..., :3], points_coor[..., :3])

        # 基于分组获取各组内点云坐标和特征，并进行拼接
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 每个group内所有点云的坐标 (B, N, K, 3)
        grouped_points_coor = grouped_points_coor - points_coor[..., :3].view(bs, npoint, 1, 3)  # 坐标转化为与采样点的偏移量
        grouped_points_coor = grouped_points_coor / self.radius  # 相对坐标归一化
        grouped_points_fea = index_points(points_fea, group_idx)  # 每个group内所有点云的特征 (B, N, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接坐标偏移量 (B, N, K, C+3)

        '''P 特征提取'''
        # (B, N, K, C+3) -> (B, C+3, K, N) -mlp-> (B, D, K, N) -pooling-> (B, D, N)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 2d卷积作用于维度1
        grouped_points_fea = self.mlp(grouped_points_fea)
        new_fea = torch.max(grouped_points_fea, dim=2)[0]

        return new_fea


class InvResMLP(nn.Module):
    """
    逆瓶颈残差块
    """

    def __init__(self,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3,
                 expansion: int = 4):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        :param expansion: 中间层通道数扩张倍数
        """
        super().__init__()
        self.la = LocalAggregation(radius=radius, nsample=nsample, in_channel=in_channel, coor_dim=coor_dim)
        channel_list = [in_channel * expansion, in_channel]
        self.pw_conv = build_mlp(in_channel=in_channel, channel_list=channel_list, dim=1, drop_last_act=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points):
        """
        :param points:
            <torch.Tensor> (B, 3, N) 点云原始坐标
            <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_fea: <torch.Tensor> (B, D, N)
        """
        points_coor, points_fea = points
        identity = points_fea
        points_fea = self.la(points_coor, points_fea)
        points_fea = self.pw_conv(points_fea)
        points_fea = points_fea + identity
        points_fea = self.act(points_fea)
        return [points_coor, points_fea]


if __name__ == '__main__':

    batch_size = 2

    num_points = 1024

    in_channel = 64

    radius = 0.1  # 邻域查询的半径（0.1米）

    nsample = 32 # 每个邻域采样的点数（最多32个邻居点）

    block = InvResMLP(
        radius=radius,
        nsample=nsample,
        in_channel=in_channel,
        coor_dim=3,
        expansion=4             # 逆瓶颈扩展倍数（通道扩展4倍）
    ).to('cuda')

    points_coor = torch.rand(batch_size, 3, num_points).to('cuda')  # (B, 3, N)

    points_fea = torch.rand(batch_size, in_channel, num_points).to('cuda')  # (B, C, N)

    output_coor, output_fea = block([points_coor, points_fea])

    print("Input coordinates size:", points_coor.size())  # 输入坐标尺寸
    print("Input features size:", points_fea.size())      # 输入特征尺寸
    print("Output coordinates size:", output_coor.size()) # 输出坐标尺寸
    print("Output features size:", output_fea.size())     # 输出特征尺寸

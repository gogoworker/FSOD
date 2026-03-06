import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter
import math

from torch.nn import Sequential as Seq, Dropout, Linear as Lin,BatchNorm1d as BN



"""《Hypergraph convolutional network based weakly supervised point cloud semantic segmentation with scene-level annotations》 Neurocomputing 2025 （中科院2区TOP）
利用场景级标注进行点云分割是一项极具前景但又极具挑战性的任务。目前，最流行的方法是利用类激活图 (CAM) 定位判别性区域，然后根据场景级标注生成点级伪标签。然而，这些方法通常存在类别间点不平衡以及 CAM 稀疏且不完全监督的问题。
本文提出了一种基于加权超图卷积网络的新型方法 WHCN，以应对从场景级标注中学习点级标签的挑战。首先，为了同时克服不同类别间的点不平衡并降低模型复杂度，利用几何同质划分生成训练点云的超点。
然后，基于从场景级标注转换而来的高置信度超点级种子点构建超图。其次，WHCN 将超图作为输入，通过标签传播学习预测高精度点级伪标签。除了由谱超图卷积块组成的骨干网络外，还学习了一个超边注意模块来调整 WHCN 中超边的权重。
最后，利用这些伪点云标签训练分割网络。在 scanNet、S3DIS、Semantic3D 和 ShapeNet Part 基准测试上的实验结果表明，所提出的 WHCN 能够有效预测带有场景标注的点标签，其 mIou 性能比当前最佳方法高出 3.5% 至 36.1%。
"""


def MLP(channels, bias=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias),BN(channels[i]),torch.nn.LeakyReLU(negative_slope=0.2),Lin(channels[i], channels[i], bias=bias))
        for i in range(1, len(channels))
    ])


class HyperedgeAttention(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperedgeAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  
        N2 = adj.shape[2]  

        pair = adj.nonzero().t()

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)

        pair_h = torch.cat((q1, x1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)

        edge = torch.matmul(attention_edge, x)

        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)

        node = torch.matmul(attention_node, edge)

        if self.concat:
            node = F.elu(node)

        return node


if __name__ == '__main__':
    batch_size = 4  # 批大小
    num_nodes = 10  # 节点数量
    num_edges = 6  # 超边数量
    in_features = 32  # 输入特征维度
    out_features = 32  # 输出特征维度
    dropout = 0.1  # dropout率
    alpha = 0.2  # LeakyReLU参数

    # 生成随机输入数据 (batch_size, num_nodes, in_features)
    x = torch.randn(batch_size, num_nodes, in_features).to('cuda')

    # 创建随机邻接矩阵 (batch_size, num_edges, num_nodes)
    adj = torch.zeros(batch_size, num_edges, num_nodes).to('cuda')
    for b in range(batch_size):
        for e in range(num_edges):
            # 每个超边随机连接2-4个节点
            connected_nodes = torch.randperm(num_nodes)[:torch.randint(2, 5, (1,))]
            adj[b, e, connected_nodes] = 1

    # 初始化超边注意力模块
    block = HyperedgeAttention(
        in_features=in_features,
        out_features=out_features,
        dropout=dropout,
        alpha=alpha,
        transfer=True,  # 是否使用线性变换
        concat=True,  # 是否使用ELU激活
        bias=False  # 是否使用偏置
    ).to('cuda')

    # 前向传播
    output = block(x, adj)

    # 打印输入输出尺寸
    print("输入特征尺寸:", x.size())
    print("邻接矩阵尺寸:", adj.size())
    print("输出特征尺寸:", output.size())

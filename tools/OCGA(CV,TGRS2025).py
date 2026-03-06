import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


def build_upsample_layer(cfg):
    if cfg['type'] == 'deconv':
        in_channels = cfg['in_channels']
        out_channels = cfg['out_channels']
        kernel_size = cfg['kernel_size']
        stride = cfg['stride']
        padding = (kernel_size - 1) // 2

        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)


def adj_index(h, k, node_num):
    dist = torch.cdist(h, h, p=2)
    each_adj_index = torch.topk(dist, k, dim=2).indices
    adj = torch.zeros(
        h.size(0), node_num, node_num,
        dtype=torch.int, device=h.device, requires_grad=False
    ).scatter_(dim=2, index=each_adj_index, value=1)
    return adj


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵W和注意力机制参数a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.activation = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self._init_weights()

    # 帮助模型稳定训练
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # 计算h和W的矩阵乘积
        e = self._prepare_attentional_mechanism_input(Wh)
        # 设置未连接节点分数设置非常低的-9e15，以便在接下来的softmax中趋于0
        attention = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # 计算注意力参数a对应的两部分与Wh的矩阵乘积
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)  # 将Wh1和Wh2转置后相加，形成注意力的未标准化分数
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class OCGA(nn.Module):

    def __init__(self, in_feature, out_feature, top_k=11, token=3, alpha=0.2, num_heads=1):
        super(OCGA, self).__init__()
        self.top_k = top_k
        hidden_feature = in_feature
        self.conv = nn.Sequential(
            nn.Conv2d(in_feature, hidden_feature, token, stride=token),
            nn.BatchNorm2d(hidden_feature),
            nn.ReLU(inplace=True)
        )
        self.attentions = [
            GraphAttentionLayer(
                hidden_feature, hidden_feature, alpha=alpha, concat=True
            ) for _ in range(num_heads)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 定义一个输出的注意力层
        self.out_att = GraphAttentionLayer(
            hidden_feature * num_heads, out_feature, alpha=alpha, concat=False)

        self.deconv = build_upsample_layer(
            cfg=dict(type='deconv',
                     in_channels=out_feature, out_channels=out_feature,
                     kernel_size=token, stride=token)
        )
        self.activation = nn.ELU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.deconv]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv(x)
        batch_size, in_feature, column, row = h.shape
        # 计算节点数量
        node_num = column * row
        h = h.view(
            batch_size, in_feature, node_num).permute(0, 2, 1)
        adj = adj_index(h, self.top_k, node_num)

        # 如有多头的设置，需要拼接
        h = torch.cat([att(h, adj) for att in self.attentions], dim=2)
        h = self.activation(self.out_att(h, adj))

        h = h.view(batch_size, column, row, -1).permute(0, 3, 1, 2)
        h = F.interpolate(
            self.deconv(h), x.shape[-2:], mode='bilinear', align_corners=True)
        return F.relu(h + x)

if __name__ == '__main__':
    batch_size = 2
    in_channels = 8
    out_channels = 8
    H, W = 32, 32
    token = 3
    top_k = 8            # 一定要 < node_num
    num_heads = 2

    block = OCGA(
        in_feature=in_channels,
        out_feature=out_channels,
        top_k=top_k,
        token=token,
        num_heads=num_heads
    ).to('cuda')

    x = torch.randn(batch_size, in_channels, H, W).to('cuda')

    y = block(x)


    print('Input shape :', x.shape)
    print('Output shape:', y.shape)

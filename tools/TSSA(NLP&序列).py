import torch
import torch.nn as nn
from torch.nn import functional as F


"""《Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction》ICLR 2025
注意力操作符可以说是 Transformer 架构的关键区别因素，它在各种任务上都表现出了最先进的性能。然而，Transformer 注意力操作符通常会带来很大的计算负担，计算复杂度与 token 数量成二次方关系。
在这项工作中，我们提出了一种新型的 Transformer 注意力操作符，其计算复杂度与 token 数量成线性关系。我们通过扩展先前的工作来推导出我们的网络架构，该工作表明 Transformer 风格的架构自然地由“白盒”架构设计产生，
其中网络的每一层都设计为实现最大编码率降低目标 (MCR ) 的增量优化步骤。具体来说，我们推导出 MCR 目标的一种新型变分形式，并表明由该变分目标的展开梯度下降产生的架构导致了一个新的注意力模块，
称为 Token Statistics Self-Attention (TSSA)。TSSA 具有线性计算和内存复杂度，并且与计算 token 之间成对相似性的典型注意力架构截然不同。
在视觉、语言和长序列任务上的实验表明，只需将 TSSA 换成标准自注意力机制（我们称之为 Token Statistics Transformer (ToST)），即可实现与传统 Transformer 相媲美的性能，同时计算效率和可解释性也显著提高。
我们的结果也在一定程度上质疑了传统观点，即成对相似性风格注意力机制对于 Transformer 架构的成功至关重要。
"""


class CausalSelfAttention_TSSA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.temp = nn.Parameter(torch.ones(config.n_head, 1))
        self.denom_bias = nn.Parameter(torch.zeros(config.n_head, config.block_size, 1))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        w = self.c_attn(x)
        w = w.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        w_sq = w ** 2
        denom = (torch.cumsum(w_sq, dim=-2)).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias[:, :T, :]
        tmp = torch.sum(w_normed, dim=-1) * self.temp

        Pi = F.softmax(tmp, dim=1)  # B, nh, T
        dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=-2) / (Pi.cumsum(dim=-1) + 1e-8).unsqueeze(-1)
        attn = 1. / (1 + dots)
        attn = self.attn_dropout(attn)
        y = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y

class Config:
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        self.n_embd = n_embd
        self.n_head = n_head
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size

if __name__ == '__main__':
    n_embd = 64  # 嵌入维度
    n_head = 8
    bias = True
    dropout = 0.1
    block_size = 32  # 序列长度

    config = Config(n_embd, n_head, bias, dropout, block_size)

    block = CausalSelfAttention_TSSA(config).to('cuda')

    #  (batch_size, sequence_length, embedding_dim)
    input = torch.rand(2, block_size, n_embd).to('cuda')

    output = block(input)

    print("Input size:", input.size())
    print("Output size:", output.size())

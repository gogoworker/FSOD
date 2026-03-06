from typing import List, Type, Tuple
import torch
import torch.nn as nn


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                    .transpose(1, 2)
                    .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


def create_linear_block1d(in_channels: int,
                          out_channels: int,
                          bn: bool = True,
                          act: bool = True
                          ) -> nn.Sequential:
    """
    Linear -> [BatchNorm] -> [ReLU]
    """
    layers: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if bn:
        layers.append(PointBatchNorm(out_channels))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DSA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expanse=1,
                 ):
        super().__init__()
        self.pre = create_linear_block1d(in_channels, out_channels)
        self.pool = lambda x: torch.max(x, dim=-2, keepdim=False)[0]
        self.bn = nn.BatchNorm1d(out_channels)

        # feed-forward network
        c = [out_channels, out_channels * expanse, out_channels]
        layers = [
            create_linear_block1d(c[i], c[i+1], act=(i < len(c)-2))
            for i in range(len(c)-1)
        ]
        self.ffn = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        p, f, pe, knn_index = inputs
        identity = f
        f = self.pre(f)
        f = self.bn(self.pool(pe + f[knn_index]))

        f = self.ffn(f)

        f = identity + f
        f = self.act(f)
        return [p, f, pe, knn_index]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- test shapes ----
    B = 2
    N = 1024
    BN = B * N
    C = 64          # set in_channels == out_channels to make residual add valid
    K = 16
    expanse = 2

    block = DSA(in_channels=C, out_channels=C, expanse=expanse).to(device)
    block.train()

    # p: 点云坐标
    p = torch.randn(BN, 3, device=device)

    # f: features [BN, C]
    f = torch.randn(BN, C, device=device)

    # knn_index: [BN, K], each row contains neighbor indices in [0, BN-1]
    knn_index = torch.randint(low=0, high=BN, size=(BN, K), device=device, dtype=torch.long)

    # pe: positional encoding / edge feature, broadcastable with f[knn_index] = [BN, K, C]
    pe = torch.randn(BN, K, C, device=device)

    out = block([p, f, pe, knn_index])
    p_out, f_out, pe_out, knn_out = out

    print("Input shapes:")
    print("  p        :", p.shape)
    print("  f        :", f.shape)
    print("  pe       :", pe.shape)
    print("  knn_index:", knn_index.shape, knn_index.dtype)

    print("Output shapes:")
    print("  p_out    :", p_out.shape)  # 点云坐标
    print("  f_out    :", f_out.shape)  # 更新后的点特征 f
    print("  pe_out   :", pe_out.shape)  # 位置编码
    print("  knn_out  :", knn_out.shape, knn_out.dtype)  # KNN邻接索引
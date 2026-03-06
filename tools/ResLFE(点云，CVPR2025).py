import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        batch_size = x.size(0)
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, device=x.device)
        random_tensor = random_tensor.floor()  # binary mask
        output = x / keep_prob * random_tensor
        return output


def vector_feature(x, knn, training=True):
    # Assuming knn is a tensor of shape (B, N, k) where k is the number of nearest neighbors
    # and x is of shape (B, N, C), where B is the batch size, N is the number of points, and C is the feature dimension.
    B, N, C = x.shape
    if not training:
        return x  # For inference, just return the input features

    # Example: simple knn-based feature aggregation
    knn_features = []
    for i in range(B):
        for j in range(N):
            # Assuming knn[i, j, :] are indices of k nearest neighbors for point (i, j)
            neighbors = x[i, knn[i, j, :], :]  # Shape: (k, C)
            aggregated_feature = torch.mean(neighbors, dim=0)  # Aggregate by mean
            knn_features.append(aggregated_feature)

    # Convert list to tensor and reshape
    knn_features = torch.stack(knn_features, dim=0).view(B, N, C)
    return knn_features


def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

class VFR(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.linear(x)
        x = vector_feature(x, knn, self.training)
        x = self.bn(x.view(B*N, -1)).view(B, N, -1)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.ffn[-1].weight, init)

    def forward(self, x):
        B, N, C = x.shape
        x = self.ffn(x.view(B*N, -1)).view(B, N, -1)
        return x

class ResLFE_Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()

        self.depth = depth
        self.VFRs = nn.ModuleList([VFR(dim, dim, bn_momentum) for _ in range(depth)])
        self.mlp = FFN(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.FFNs = nn.ModuleList([FFN(dim, mlp_ratio, bn_momentum, act) for _ in range(depth)])

        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        self.drop_paths = nn.ModuleList([DropPath(dpr) for dpr in drop_rates])

    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        if pts is None:
            # 如果pts为None，直接对整个张量应用drop path
            return self.drop_paths[i](x)
        else:
            # 如果pts不为None，按照pts分割并分别应用drop path
            return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, pe, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            x = x + pe
            x = x + self.drop_path(self.VFRs[i](x, knn), i, pts)
            x = x + self.drop_path(self.FFNs[i](x), i, pts)
        return x


if __name__ == '__main__':
    # Test parameters
    batch_size = 2
    num_points = 1024
    feature_dim = 64
    k_neighbors = 16

    # Create input tensor
    x = torch.randn(batch_size, num_points, feature_dim).to('cuda')

    # Create positional encoding (dummy)
    pe = torch.randn(batch_size, num_points, feature_dim).to('cuda')

    # Create dummy knn indices (for testing)
    knn = torch.randint(0, num_points, (batch_size, num_points, k_neighbors)).to('cuda')

    # Create block
    block = ResLFE_Block(
        dim=feature_dim,
        depth=2,
        drop_path=0.1,
        mlp_ratio=4.0,
        bn_momentum=0.1,
        act=nn.GELU
    ).to('cuda')

    # Forward pass
    output = block(x, pe, knn)

    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")


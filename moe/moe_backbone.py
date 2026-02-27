import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import ResNet
from mmdet.models.builder import BACKBONES
from mmcv.cnn.bricks import build_activation_layer
from typing import Tuple


class FFN(nn.Module):
    def __init__(self, in_channels, mid_channels, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CosineTopKGate(nn.Module):
    def __init__(self, model_dim, num_experts, init_t=0.5):
        super().__init__()
        proj_dim = min(model_dim // 2, 256)
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)))
        self.cos_proj = nn.Linear(model_dim, proj_dim)
        self.sim_matrix = nn.Parameter(torch.randn(proj_dim, num_experts))

    def forward(self, x):
        sim = F.normalize(self.cos_proj(x), dim=-1) @ F.normalize(self.sim_matrix, dim=0)
        scale = self.temperature.exp()
        return sim * scale


class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self.gates = gates
        self.num_experts = num_experts
        nonzero = gates.nonzero(as_tuple=False)
        self.batch_idx = nonzero[:, 0]
        self.expert_idx = nonzero[:, 1]
        self.part_sizes = (gates > 0).sum(0).tolist()
        self.nonzero_gates = gates[self.batch_idx, self.expert_idx]

    def dispatch(self, x):
        return torch.split(x[self.batch_idx], self.part_sizes, dim=0)

    def combine(self, expert_outs):
        stitched = torch.cat(expert_outs, dim=0) * self.nonzero_gates.unsqueeze(-1)
        output = torch.zeros_like(self.gates[:, 0].unsqueeze(-1).expand(-1, stitched.size(1)))
        return output.index_add(0, self.batch_idx, stitched)


class MoE(nn.Module):
    def __init__(self, in_channels, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([FFN(in_channels, in_channels * 4) for _ in range(num_experts)])
        self.gate = CosineTopKGate(in_channels, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.k = top_k

    def forward(self, x):
        gates_logits = self.gate(x)
        topk_vals, topk_idx = gates_logits.topk(self.k, dim=-1)
        topk_gates = self.softmax(topk_vals)
        gates = torch.zeros_like(gates_logits)
        gates.scatter_(1, topk_idx, topk_gates)

        dispatcher = SparseDispatcher(len(self.experts), gates)
        inputs = dispatcher.dispatch(x)
        outputs = [self.experts[i](inputs[i]) for i in range(len(inputs))]
        return dispatcher.combine(outputs)


class MoEResBlock(nn.Module):
    def __init__(self, in_channels, use_moe=False, **moe_cfg):
        super().__init__()
        self.use_moe = use_moe
        self.norm = nn.LayerNorm(in_channels)
        if use_moe:
            self.ffn = MoE(in_channels, **moe_cfg)
        else:
            self.ffn = FFN(in_channels, in_channels * 4)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out = self.ffn(self.norm(x_flat))
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


@BACKBONES.register_module()
class ResNetWithMoEMetaConv(ResNet):
    def __init__(self, moe_stage_inds=[2, 3], moe_block_inds=[[0], [1]], num_experts=4, top_k=2, **kwargs):
        super().__init__(**kwargs)
        self.meta_conv = build_conv_layer(
            self.conv_cfg,
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)

        self.moe_stage_inds = moe_stage_inds
        self.moe_block_inds = moe_block_inds
        self.num_experts = num_experts
        self.top_k = top_k

        # patch res_layers to add MoE blocks
        for stage_idx, layer_name in enumerate(self.res_layers):
            stage = getattr(self, layer_name)
            for block_idx in range(len(stage)):
                if stage_idx in self.moe_stage_inds and block_idx in self.moe_block_inds[self.moe_stage_inds.index(stage_idx)]:
                    stage[block_idx].ffn = MoE(
                        in_channels=stage[block_idx].conv1.out_channels,
                        num_experts=self.num_experts,
                        top_k=self.top_k
                    )

    def forward(self, x: torch.Tensor, use_meta_conv: bool = False) -> Tuple[torch.Tensor]:
        if use_meta_conv:
            x = self.meta_conv(x)
        else:
            x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

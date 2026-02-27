# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES

# ========== MoE 相关组件 ==========
class CosineTopKGate(nn.Module):
    def __init__(self, model_dim, num_global_experts, init_t=0.5):
        super().__init__()
        proj_dim = min(model_dim//2, 256)
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = nn.Linear(model_dim, proj_dim)
        self.sim_matrix = nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        logits = torch.matmul(F.normalize(self.cosine_projector(x), dim=1),
                              F.normalize(self.sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates, as_tuple=True)
        _, self._expert_index = sorted_experts.sort(0)
        self._batch_index = torch.nonzero(gates, as_tuple=True)[0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.long(), index_sorted_experts.long()].view(-1)
        self._nonzero_gates = gates_exp.type_as(gates)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index.long()]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.unsqueeze(-1)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(-1), device=stitched.device)
        combined = zeros.index_add(0, self._batch_index.long(), stitched)
        return combined

class MoE_layer(nn.Module):
    def __init__(self, moe_cfg):
        super().__init__()
        self.noisy_gating = moe_cfg['noisy_gating']
        self.num_experts = moe_cfg['num_experts']
        self.input_size = moe_cfg['in_channels']
        self.k = moe_cfg['top_k']
        self.gating = moe_cfg['gating']
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.input_size),
                nn.GELU(),
                nn.Linear(self.input_size, self.input_size)
            ) for _ in range(self.num_experts)
        ])
        if moe_cfg['gating'] == 'linear':
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        elif moe_cfg['gating'] == 'cosine':
            self.w_gate = CosineTopKGate(self.input_size, self.num_experts)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def forward(self, x, loss_coef=1e-2):
        train = self.training
        x_flat = x.reshape(-1, x.shape[-1])
        if self.gating == 'linear':
            clean_logits = x_flat @ self.w_gate
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x_flat)
        logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)
        importance = gates.sum(dim=0)
        load = self._gates_to_load(gates)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts) if i < len(expert_inputs)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(x.shape)
        return y, loss

# ========== ConvNeXt Block ==========
class ConvNeXtBlock(BaseModule):
    def __init__(self, in_channels, mlp_ratio=4., drop_path=0., layer_scale_init_value=1e-6, use_moe=False, moe_cfg=None, norm_cfg=dict(type='LN', eps=1e-6)):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=norm_cfg.get('eps', 1e-6))
        self.pwconv1 = nn.Linear(in_channels, int(mlp_ratio * in_channels))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * in_channels), in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # 可选DropPath
        self.use_moe = use_moe
        if use_moe and moe_cfg is not None:
            moe_cfg = moe_cfg.copy()
            moe_cfg['in_channels'] = in_channels
            self.moe = MoE_layer(moe_cfg)
        else:
            self.moe = None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW->NHWC
        x = self.norm(x)
        if self.use_moe and self.moe is not None:
            x, moe_loss = self.moe(x)
        else:
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            moe_loss = 0.0
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC->NCHW
        x = input + self.drop_path(x)
        return x, moe_loss

# ========== ConvNeXt-MoE Backbone ==========
@BACKBONES.register_module()
class ConvNeXtMoE(BaseModule):
    arch_settings = {
        'tiny':   {'depths': [3, 3, 9, 3],  'dims': [96, 192, 384, 768]},
        'small':  {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
        'base':   {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'large':  {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
    }
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 depths=None,
                 dims=None,
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 moe_block_inds=[[], [], [], []],
                 num_experts=4,
                 top_k=2,
                 gate='cosine',
                 noisy_gating=True,
                 init_cfg=None,
                 pretrained=None):
        if depths is None or dims is None:
            assert arch in self.arch_settings, f"Unknown arch: {arch}"
            depths = self.arch_settings[arch]['depths']
            dims = self.arch_settings[arch]['dims']
        super().__init__(init_cfg)
        self.downsample_layers = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(self.stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.moe_cfg = {
            'num_experts': num_experts,
            'top_k': top_k,
            'gating': gate,
            'noisy_gating': noisy_gating
        }
        self.moe_block_inds = moe_block_inds
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                use_moe = j in moe_block_inds[i]
                blocks.append(
                    ConvNeXtBlock(
                        in_channels=dims[i],
                        mlp_ratio=4.,
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        use_moe=use_moe,
                        moe_cfg=self.moe_cfg if use_moe else None
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]
        self.out_indices = out_indices
        self.pretrained = pretrained
        self._init_weights()

    def _init_weights(self):
        if self.pretrained is not None:
            state_dict = torch.hub.load_state_dict_from_url(self.pretrained, map_location='cpu', check_hash=True)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        moe_losses = []
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x, moe_loss = block(x)
                if moe_loss != 0.0:
                    moe_losses.append(moe_loss)
            if i in self.out_indices:
                outs.append(x)
        self.avg_moe_loss = sum(moe_losses) / len(moe_losses) if moe_losses else 0.0
        return tuple(outs)

    def get_moe_loss(self):
        return getattr(self, 'avg_moe_loss', 0.0) 
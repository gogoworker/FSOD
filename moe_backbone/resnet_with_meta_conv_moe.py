# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, List, Dict, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from mmcv.cnn import build_conv_layer
from mmdet.models import ResNet
from mmdet.models.builder import BACKBONES
from torch import Tensor
from mmcv.runner import BaseModule
from mmfewshot.detection.models.backbones.resnet_with_meta_conv import ResNetWithMetaConv


class CosineTopKGate(nn.Module):
    """Cosine similarity based top-k gating mechanism for MoE."""
    def __init__(self, model_dim, num_experts, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        proj_dim = min(model_dim//2, 256)
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = nn.Linear(model_dim, proj_dim)
        self.sim_matrix = nn.Parameter(torch.randn(size=(proj_dim, num_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        logits = torch.matmul(F.normalize(self.cosine_projector(x), dim=1),
                            F.normalize(self.sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts."""
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher.
        
        Args:
            num_experts (int): Number of experts.
            gates (Tensor): Tensor of shape (batch_size, num_experts).
        """
        self._gates = gates
        self._num_experts = num_experts
        # Sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates, as_tuple=True)
        # Get the nonzero indices for each sample
        _, self._expert_index = sorted_experts.sort(0)
        # Gather the gating weights for each expert
        self._batch_index = torch.nonzero(gates, as_tuple=True)[0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.long(), index_sorted_experts.long()].view(-1)
        self._nonzero_gates = gates_exp.type_as(gates)

    def dispatch(self, inp):
        """Dispatch inputs to experts.
        
        Args:
            inp (Tensor): Input tensor.
        
        Returns:
            List[Tensor]: List of tensors, one for each expert.
        """
        inp_exp = inp[self._batch_index.long()]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Combine outputs from different experts.
        
        Args:
            expert_out (List[Tensor]): List of expert outputs.
            multiply_by_gates (bool): Whether to multiply by gates.
        
        Returns:
            Tensor: Combined output.
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.unsqueeze(-1)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(-1),
                          device=stitched.device)
        combined = zeros.index_add(0, self._batch_index.long(), stitched)
        return combined

    def expert_to_gates(self):
        """Return the gates per expert as a list of tensors."""
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_layer(nn.Module):
    """Mixture of Experts layer."""
    def __init__(self, 
                moe_cfg
                ):
        super(MoE_layer, self).__init__() 
        self.noisy_gating = moe_cfg['noisy_gating']
        self.num_experts = moe_cfg['num_experts']
        self.input_size = moe_cfg['in_channels']
        self.k = moe_cfg['top_k']
        self.gating = moe_cfg['gating']
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.input_size, self.input_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.input_size),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_experts)
        ])
        
        # 门控网络
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
        """计算变异系数的平方。"""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """计算每个专家的负载。"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """计算一个噪声值在top k中的概率。"""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """使用噪声top-k门控机制。"""
        # 将空间维度展平为特征维度
        batch_size, channels, height, width = x.shape
        x_flat = x.reshape(batch_size, channels, -1).mean(dim=2)
        
        if self.gating == 'linear':
            clean_logits = x_flat @ self.w_gate
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x_flat)

        if self.noisy_gating and train:
            raw_noise_stddev = x_flat @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """前向传播。"""
        train = self.training
        
        original_shape = x.shape
        batch_size, channels, height, width = original_shape
        
        # 为门控网络准备输入
        x_flat = x.reshape(batch_size, channels, -1).mean(dim=2)
        
        # 获取门控信号和负载
        gates, load = self.noisy_top_k_gating(x, train)
        
        # 计算重要性和损失
        importance = gates.sum(dim=0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # 分发输入到专家
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        
        # 专家处理
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts) if i < len(expert_inputs)]
        
        # 组合专家输出
        y = dispatcher.combine(expert_outputs)
        
        return y, loss


class MoEBasicBlock(BaseModule):
    """ResNet基本块的MoE版本。"""
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None,
                 moe_cfg=None):
        super(MoEBasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        
        # 使用MoE层替代常规卷积
        if moe_cfg is not None:
            moe_cfg['in_channels'] = planes
            self.moe = MoE_layer(moe_cfg)
            self.use_moe = True
        else:
            self.conv2 = build_conv_layer(
                conv_cfg, planes, planes, 3, padding=1, bias=False)
            self.use_moe = False
            
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.use_moe:
                out, loss = self.moe(out)
            else:
                out = self.conv2(out)
            
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class MoEBottleneck(BaseModule):
    """ResNet瓶颈块的MoE版本。"""
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None,
                 moe_cfg=None):
        super(MoEBottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        
        # 使用MoE层替代常规卷积
        if moe_cfg is not None:
            moe_cfg['in_channels'] = planes
            self.moe = MoE_layer(moe_cfg)
            self.use_moe = True
        else:
            self.use_moe = False
        
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """制作块插件。"""
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """前向传播插件。"""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            if self.use_moe:
                out, loss = self.moe(out)
            
            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def build_norm_layer(cfg, num_features, postfix=''):
    """构建标准化层。"""
    if cfg is None:
        return 'identity', nn.Identity()
    
    if cfg['type'] == 'BN':
        norm_layer = nn.BatchNorm2d(num_features)
        norm_name = 'bn' + postfix
    elif cfg['type'] == 'LN':
        norm_layer = nn.LayerNorm(num_features)
        norm_name = 'ln' + postfix
    else:
        raise NotImplementedError
        
    return norm_name, norm_layer


def build_plugin_layer(cfg, **kwargs):
    """构建插件层。"""
    # 这里简化实现，实际应该根据cfg构建相应的插件层
    return 'plugin', nn.Identity()


@BACKBONES.register_module()
class ResNetWithMetaConvMoE(ResNetWithMetaConv):
    """带有MoE层的ResNetWithMetaConv。
    
    Args:
        MoE_Block_inds (List[List[int]]): 每个阶段中应用MoE的块索引列表。
        num_experts (int): 专家数量。
        top_k (int): 每次选择的专家数量。
        gate (str): 门控机制类型，'linear'或'cosine'。
        noisy_gating (bool): 是否使用噪声门控。
    """
    
    arch_settings = {
        18: (MoEBasicBlock, (2, 2, 2, 2)),
        34: (MoEBasicBlock, (3, 4, 6, 3)),
        50: (MoEBottleneck, (3, 4, 6, 3)),
        101: (MoEBottleneck, (3, 4, 23, 3)),
        152: (MoEBottleneck, (3, 8, 36, 3))
    }
    
    def __init__(self,
                 MoE_Block_inds=[[],[],[],[]],
                 num_experts=8,
                 top_k=2,
                 gate='cosine',
                 noisy_gating=True,
                 **kwargs):
        self.MoE_Block_inds = MoE_Block_inds
        self.moe_cfg = {
            'num_experts': num_experts,
            'top_k': top_k,
            'gating': gate,
            'noisy_gating': noisy_gating
        }
        super(ResNetWithMetaConvMoE, self).__init__(**kwargs)
    
    def make_res_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, 
                      style='pytorch', with_cp=False, conv_cfg=None, 
                      norm_cfg=dict(type='BN'), dcn=None, plugins=None,
                      init_cfg=None, stage_idx=None):
        """构建ResNet层。"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        blocks_moe_cfg = self.MoE_Block_inds[stage_idx] if stage_idx is not None else []
        
        for i in range(blocks):
            # 只有在MoE_Block_inds中指定的块才使用MoE
            use_moe = i in blocks_moe_cfg if blocks_moe_cfg else False
            moe_cfg_block = self.moe_cfg.copy() if use_moe else None
            
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride if i == 0 else 1,
                    dilation=dilation,
                    downsample=downsample if i == 0 else None,
                    style=style,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn if self.stage_with_dcn[stage_idx] else None,
                    plugins=plugins,
                    init_cfg=init_cfg,
                    moe_cfg=moe_cfg_block))
            inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, stage_idx=None):
        """构建ResNet层的包装函数。"""
        return self.make_res_layer(
            block=block,
            inplanes=inplanes,
            planes=planes,
            blocks=blocks,
            stride=stride,
            dilation=dilation,
            style=self.style,
            with_cp=self.with_cp,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=self.dcn,
            plugins=self.plugins,
            stage_idx=stage_idx)
    
    def _make_stem_layer(self, in_channels, stem_channels):
        """构建ResNet的stem层。"""
        # 保持与原始ResNetWithMetaConv相同的实现
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 添加meta_conv，与原始ResNetWithMetaConv相同
        self.meta_conv = build_conv_layer(
            self.conv_cfg,
            4,  # 4通道输入（图像+掩码）
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
    
    def init_weights(self):
        """初始化权重。"""
        super(ResNetWithMetaConvMoE, self).init_weights()
        
        # 初始化MoE相关参数
        for m in self.modules():
            if isinstance(m, MoE_layer):
                if hasattr(m, 'w_gate') and isinstance(m.w_gate, nn.Parameter):
                    nn.init.zeros_(m.w_gate)
                if hasattr(m, 'w_noise'):
                    nn.init.zeros_(m.w_noise)
    
    def forward(self, x, use_meta_conv=False):
        """前向传播函数。
        
        Args:
            x (Tensor): 输入张量，形状为(N, 3, H, W)或(N, 4, H, W)
            use_meta_conv (bool): 是否使用meta_conv。默认为False。
        
        Returns:
            tuple[Tensor]: 特征图元组。
        """
        # 保持与原始ResNetWithMetaConv相同的接口
        return super(ResNetWithMetaConvMoE, self).forward(x, use_meta_conv) 
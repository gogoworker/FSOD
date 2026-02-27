# 文件位置建议：mmfewshot/detection/models/roi_heads/bbox_heads/proto_moe_bbox_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmfewshot.detection.models.roi_heads.bbox_heads.meta_bbox_head import MetaBBoxHead
from mmcv.runner import force_fp32

class ProtoMoEBBoxHead(MetaBBoxHead):
    def __init__(self,
                 num_classes,
                 num_prototypes=3,
                 with_cls=True,
                 with_reg=True,
                 loss_proto_diversity_weight=0.1,
                 loss_proto_inter_weight=0.1,
                 **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_prototypes = num_prototypes
        self.with_cls = with_cls
        self.with_reg = with_reg

        # 用于存放每类多个原型，参数大小：[num_classes, num_prototypes, feat_dim]
        self.proto_embeddings = nn.Parameter(
            torch.randn(num_classes, num_prototypes, self.feat_dim))
        nn.init.xavier_uniform_(self.proto_embeddings)

        # loss 权重参数
        self.loss_proto_diversity_weight = loss_proto_diversity_weight
        self.loss_proto_inter_weight = loss_proto_inter_weight

    @force_fp32(apply_to=('support_feats', 'query_feats'))
    def compute_prototype_loss(self):
        """计算原型多样性 & 类间距离正则."""
        # [C, P, D]
        protos = F.normalize(self.proto_embeddings, dim=-1)

        # 类内多样性：鼓励同类原型方向不同
        intra_loss = 0.
        for c in range(self.num_classes):
            pc = protos[c]  # [P, D]
            cos_sim = torch.mm(pc, pc.T)  # [P, P]
            mask = torch.ones_like(cos_sim) - torch.eye(self.num_prototypes, device=cos_sim.device)
            intra_loss += (cos_sim * mask).sum() / (self.num_prototypes * (self.num_prototypes - 1))

        # 类间分离性：鼓励不同类原型方向远离
        inter_loss = 0.
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                pi = protos[i].reshape(-1, self.feat_dim)
                pj = protos[j].reshape(-1, self.feat_dim)
                inter_cos = torch.mm(pi, pj.T)
                inter_loss += inter_cos.mean()

        loss_div = self.loss_proto_diversity_weight * intra_loss
        loss_inter = self.loss_proto_inter_weight * (-inter_loss)  # 拉远，越小越好
        return {'loss_proto_diversity': loss_div, 'loss_proto_inter': loss_inter}

    @force_fp32(apply_to=('query_feats', ))
    def forward_proto_classification(self, query_feats):
        """
        query_feats: [N, D]
        return: cosine logits [N, num_classes]
        """
        N, D = query_feats.size()
        query_feats = F.normalize(query_feats, dim=-1)  # 单位球面归一化

        # 获取原型并归一化：[C, P, D]
        protos = F.normalize(self.proto_embeddings, dim=-1)

        # 与每个原型计算相似度：[N, C, P]
        sim = torch.einsum('nd,cpd->ncp', query_feats, protos)

        # gate 权重：softmax over P 原型
        gate = F.softmax(sim, dim=-1)  # [N, C, P]

        # 聚合后的 class prototype（注意权重来自每个 query）
        # [N, C, D]
        weighted_proto = torch.einsum('ncp,cpd->ncd', gate, protos)

        # 再次归一化（防止 gate 平滑后模长不为 1）
        weighted_proto = F.normalize(weighted_proto, dim=-1)

        # cosine 相似度作为 logits：[N, C]
        logits = torch.einsum('nd,ncd->nc', query_feats, weighted_proto)

        return logits

    def forward_train(self, query_feats, query_labels, **kwargs):
        losses = dict()

        # 分类 logits（[N, C]）
        logits = self.forward_proto_classification(query_feats)

        # classification loss
        if self.with_cls:
            cls_loss = F.cross_entropy(logits, query_labels)
            losses['loss_cls'] = cls_loss

        # 原型正则项
        losses.update(self.compute_prototype_loss())

        return losses

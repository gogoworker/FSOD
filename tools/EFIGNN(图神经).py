import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data


"""《Explicit Feature Interaction-Aware Graph Neural Network》IEEE Access2024
图神经网络 (GNN) 是处理图结构数据的强大工具。然而，它们的设计通常限制它们只能学习高阶特征交互，而忽略了低阶特征交互。
为了解决这个问题，我们引入了一种新的 GNN 方法，称为显式特征交互感知图神经网络 (explicit feature interaction-aware graph neural network，EFI-GNN)。
与传统 GNN 不同，EFI-GNN 是一个多层线性网络，旨在显式地对图中的任意阶特征交互进行建模。为了验证 EFI-GNN 的有效性，我们使用各种数据集进行了实验。
实验结果表明，EFI-GNN 的性能与现有的 GNN 相比具有竞争力，并且当 GNN 与 EFI-GNN 联合训练时，预测性能会有所改善。此外，由于 EFI-GNN 的线性结构，它的预测是可解释的。
"""

class _BaseLightningModel(pl.LightningModule):

    def __init__(
            self,
            lr: float,
            weight_decay: float
    ) -> None:
        super().__init__()
        self._lr = lr
        self._weight_decay = weight_decay
        self._y_hat = []
        self._y_true = []

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_index) -> torch.Tensor:
        self.train()
        y_hat = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index) -> torch.Tensor:
        self.eval()
        y_hat = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(y_hat[batch.val_mask], batch.y[batch.val_mask])
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_index) -> None:
        self.eval()
        y_hat = self(batch.x, batch.edge_index)[batch.test_mask].argmax(dim=1)
        y_true = batch.y[batch.test_mask]
        self._y_hat.append(y_hat)
        self._y_true.append(y_true)

    def on_test_epoch_end(self) -> None:
        y_hat = torch.concat(self._y_hat, dim=0)
        y_true = torch.concat(self._y_true, dim=0)
        acc = (y_hat == y_true).to(torch.float32).mean()
        print(acc.item())
        self._y_hat.clear()
        self._y_true.clear()

__all__ = ['EFIGNNConv', 'EFIGNN']


class EFIGNNConv(MessagePassing):

    def __init__(
            self,
            channels: int
    ):
        super().__init__(aggr='add')
        self._linear = nn.Linear(channels, channels, bias=False)
        self._bias = nn.Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self):
        self._linear.reset_parameters()
        self._bias.data.zero_()

    def forward(
            self,
            x: torch.Tensor,
            x0: torch.Tensor,
            edge_index: torch.Tensor
    ):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self._linear(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        out = out * x0
        out = out + self._bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class EFIGNN(_BaseLightningModel):

    def __init__(
            self,
            architecture: list,
            dropout: float = 0,
            batchnorm: bool = False,
            lr: float = 1e-3,
            weight_decay: float = 0,
            skip_conn: bool = False,
            apply_output_layer: bool = True
    ) -> None:

        super().__init__(lr, weight_decay)

        # self.save_hyperparameters()

        n_layers = len(architecture) - 2

        self._layers = nn.ModuleList()

        # self._layers.append(nn.Linear(architecture[0], architecture[1]))
        self._layers.append(GCNConv(architecture[0], architecture[1]))
        if batchnorm:
            self._layers.append(nn.BatchNorm1d(architecture[1]))
        self._layers.append(nn.Dropout(p=dropout))
        for i in range(n_layers - 1):
            self._layers.append(EFIGNNConv(architecture[i + 1]))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])

        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ):
        xs = []
        for layer in self._layers:
            if isinstance(layer, MessagePassing):
                xs.append(x)
                if len(xs) == 1:
                    x = layer(x, edge_index)
                else:
                    if self._skip_conn:
                        # x = layer(x, xs[0], edge_index) + sum(xs[1:])
                        x = layer(x, xs[1], edge_index) + sum(xs[1:])
                    else:
                        # x = layer(x, xs[0], edge_index)
                        x = layer(x, xs[1], edge_index)
            else:
                x = layer(x)
        xs.append(x)
        xs = torch.concat(xs[1:], dim=1)

        if self._apply_output_layer:
            y_hat = self._output_layer(xs)
        else:
            y_hat = xs

        return y_hat

if __name__ == '__main__':
    # 创建一个简单的图数据
    num_nodes = 6   # 节点数量
    num_features = 3  # 每个节点的特征数量
    num_classes = 3  # 类别数量

    # 随机生成节点特征
    x = torch.rand((num_nodes, num_features), dtype=torch.float32)

    # 定义边连接（无向图）
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)

    # 随机生成标签
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, y=y)

    # 创建模型实例
    block = EFIGNN(architecture=[num_features, 4, 4, num_classes]).to('cuda')

    # 将数据移动到 GPU
    data = data.to('cuda')

    # 进行前向传播
    output = block(data.x, data.edge_index)

    # 打印输入和输出的尺寸
    print("Input size:", data.x.size())
    print("Output size:", output.size())
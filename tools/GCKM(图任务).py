import torch
import torch.nn as nn

"""《Graph Convolutional Kernel Machine versus Graph Convolutional Networks》NeurIPS 2023 
具有一个或两个隐藏层的图卷积网络 (GCN) 已广泛用于处理各个学科中普遍存在的图数据。许多研究表明，使 GCN 更深的收益很小甚至是负的。这意味着图数据的复杂性通常有限，浅层模型通常足以提取节点分类等各种任务的表达特征。
因此，在本文中，我们提出了一个称为graph convolutional kernel machine (GCKM) 的框架，用于基于图的机器学习。GCKM 建立在与图卷积集成的核函数之上。
一个例子是用于节点分类的图卷积核支持向量机 (GCKSVM)，我们分析了它的泛化误差界限并讨论了图结构的影响。与 GCN 相比，GCKM 在架构设计、超参数调整和优化方面需要的努力要少得多。
更重要的是，GCKM 保证获得全局最优解，并具有强大的泛化能力和高可解释性。 GCKM 是可组合的，可以扩展到大规模数据，并适用于各种任务（例如，节点或图分类、聚类、特征提取、降维）。
基准数据集上的数值结果表明，除了上述优势外，GCKM 至少具有与 GCN 相比具有竞争力的准确率。
"""

class GCKLayer(nn.Module):
    def __init__(self, gamma, layer):
        super(GCKLayer, self).__init__()
        self.gamma = gamma
        self.layer = layer

    def rbf_kernel_X(self, X, gamma):
        n = X.shape[0]
        Sij = torch.matmul(X, X.T)
        Si = torch.unsqueeze(torch.diag(Sij), 0).T @ torch.ones(1, n).to(X.device)
        Sj = torch.ones(n, 1).to(X.device) @ torch.unsqueeze(torch.diag(Sij), 0)
        D2 = Si + Sj - 2 * Sij
        K = torch.exp(-D2 * gamma)
        # K[torch.isinf(K)1234] = 1.
        return K

    def rbf_kernel_K(self, K_t, gamma):
        n = K_t.shape[0]
        s = torch.unsqueeze(torch.diag(K_t), 0)
        D2 = torch.ones(n, 1).to(K_t.device) @ s + s.T @ torch.ones(1, n).to(K_t.device) - 2 * K_t
        K = torch.exp(-D2 * gamma)
        # K[torch.isinf(K)] = 1.
        return K

    def forward(self, adj, inputs):
        if self.layer == 0:
            X_t = adj @ inputs
            K = self.rbf_kernel_X(X_t, self.gamma)
            return K
        else:
            K_t = adj @ inputs @ adj.t()
            K = self.rbf_kernel_K(K_t, self.gamma)
            return K


class GCKM(nn.Module):
    def __init__(self, gamma_list):
        super(GCKM, self).__init__()
        self.model = nn.ModuleList()
        for i, gamma in enumerate(gamma_list):
            self.model.append(GCKLayer(gamma, i))

    def forward(self, adj, X):
        K = X
        for layer in self.model:
            K = layer(adj, K)
        return K


if __name__ == '__main__':
    # 配置参数
    gamma_list = [0.1, 0.05]  # 每一层的 gamma 值
    num_nodes = 10            # 图中节点数
    feature_dim = 10           # 节点特征维度

    block = GCKM(gamma_list).to('cuda')

    adj = torch.rand(num_nodes, num_nodes).to('cuda')  # 随机生成邻接矩阵（根据实际任务替换）
    X = torch.rand(num_nodes, feature_dim).to('cuda')  # 随机生成节点特征

    output = block(adj, X)

    print("Input adjacency matrix size:", adj.size())
    print("Input feature matrix size:", X.size())
    print("Output kernel matrix size:", output.size())

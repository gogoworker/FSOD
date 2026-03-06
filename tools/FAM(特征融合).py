import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""《FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation》AAAI2025
现有的小样本医学图像分割 (FSMIS) 模型无法解决医学成像中的一个实际问题：不同成像技术导致的领域偏移，这限制了其在当前 FSMIS 任务中的适用性。
为了克服这一限制，我们专注于跨域小样本医学图像分割 (CD-FSMIS) 任务，旨在开发一个通用模型，该模型能够适应新目标域中标记数据有限的更广泛医学图像分割场景。
受不同域间频域相似性特点的启发，我们提出了一种频率感知匹配网络 (FAMNet)，它包括两个关键组件：频率感知匹配 (FAM) 模块和多光谱融合 (MSF) 模块。
FAM 模块在元学习阶段解决了两个问题：1) 由于器官和病变外观不同而导致的固有支持查询偏差引起的域内差异，以及 2) 由不同医学成像技术引起的域间差异。
此外，我们设计了一个 MSF 模块来整合由 FAM 模块解耦的不同频率特征，并进一步减轻域间差异对模型分割性能的影响。
结合这两个模块，我们的 FAMNet 在三个跨域数据集上超越了现有的 FSMIS 模型和跨域小样本语义分割模型，在 CD-FSMIS 任务中实现了最佳性能。
"""

class AttentionMacthcing(nn.Module):
    def __init__(self, feature_dim=512, seq_len=5000):
        super(AttentionMacthcing, self).__init__()
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),

            nn.ReLU(),
            nn.Linear(seq_len // 5, 2 * seq_len),
        )
        self.sigmoid = nn.Sigmoid()

    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        """
        Calculates the correlation matrix between the spatial foreground features and query foreground features.

        Args:
            spt_fg_fts (torch.Tensor): The spatial foreground features.
            qry_fg_fts (torch.Tensor): The query foreground features.

        Returns:
            torch.Tensor: The cosine similarity matrix. Shape: [1, 1, N].
        """

        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)  # shape [1, 512, 900]
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)  # shape [1, 512, 900]

        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)  # shape: [1, 1, N]

        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band):
        """
        Args:
            spt_fg_fts (torch.Tensor): Spatial foreground features.
            qry_fg_fts (torch.Tensor): Query foreground features.
            band (str): Band type, either 'low', 'high', or other.

        Returns:
            torch.Tensor: Fused tensor. Shape: [1, 512, 5000].
        """

        spt_proj = F.relu(self.fc_spt(spt_fg_fts))  # shape: [1, 512, 900]
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))  # shape: [1, 512, 900]

        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))

        if band == 'low' or band == 'high':
            weighted_spt = (1 - similarity_matrix) * spt_proj  # shape: [1, 512, 900]
            weighted_qry = (1 - similarity_matrix) * qry_proj  # shape: [1, 512, 900]
        else:
            weighted_spt = similarity_matrix * spt_proj  # shape: [1, 512, 900]
            weighted_qry = similarity_matrix * qry_proj  # shape: [1, 512, 900]

        combined = torch.cat((weighted_spt, weighted_qry), dim=2)  # shape: [1, 1024, 900]
        fused_tensor = F.relu(self.fc_fusion(combined))  # shape: [1, 512, 900]

        return fused_tensor


class FAM(nn.Module):
    def __init__(self, feature_dim=784, N=64):
        super(FAM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.attention_matching = AttentionMacthcing(feature_dim, N)
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)

    def forward(self, spt_fg_fts, qry_fg_fts):
        """
        Forward pass of the FAM module.

        Args:
            spt_fg_fts (torch.Tensor): Shape [B, C, N]
            qry_fg_fts (torch.Tensor): Shape [B, C, N]

        Returns:
            torch.Tensor: Fused features
        """
        # 将输入张量转换为模块期望的格式
        spt_fg_fts = [[spt_fg_fts]]
        qry_fg_fts = [qry_fg_fts]

        # 应用自适应池化
        spt_fg_fts = [[self.adapt_pooling(fts) for fts in way] for way in spt_fg_fts]
        qry_fg_fts = [self.adapt_pooling(fts) for fts in qry_fg_fts]

        # 获取频率带
        spt_fg_fts_low, spt_fg_fts_mid, spt_fg_fts_high = self.filter_frequency_bands(spt_fg_fts[0][0], cutoff=0.30)
        qry_fg_fts_low, qry_fg_fts_mid, qry_fg_fts_high = self.filter_frequency_bands(qry_fg_fts[0], cutoff=0.30)

        # 注意力匹配
        fused_fts_low = self.attention_matching(spt_fg_fts_low, qry_fg_fts_low, 'low')
        fused_fts_mid = self.attention_matching(spt_fg_fts_mid, qry_fg_fts_mid, 'mid')
        fused_fts_high = self.attention_matching(spt_fg_fts_high, qry_fg_fts_high, 'high')

        # 返回融合特征
        return fused_fts_low + fused_fts_mid + fused_fts_high

    def reshape_to_square(self, tensor):
        """
        Reshapes a tensor to a square shape.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, N), where B is the batch size,
                C is the number of channels, and N is the number of elements.

        Returns:
            tuple: A tuple containing:
                - square_tensor (torch.Tensor): The reshaped tensor of shape (B, C, side_length, side_length),
                  where side_length is the length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - N (int): The original number of elements in the input tensor.
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2

        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor

        square_tensor = padded_tensor.view(B, C, side_length, side_length)

        return square_tensor, side_length, side_length, N

    def filter_frequency_bands(self, tensor, cutoff=0.2):
        """
        Filters the input tensor into low, mid, and high frequency bands.
        """
        # 获取输入张量的设备
        device = tensor.device

        tensor = tensor.float()
        tensor, H, W, N = self.reshape_to_square(tensor)
        B, C, _, _ = tensor.shape

        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        low_cutoff = max_radius * cutoff
        high_cutoff = max_radius * (1 - cutoff)

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=device):
            rows, cols = shape
            center_row, center_col = rows // 2, cols // 2

            # 确保在正确的设备上创建张量
            y, x = torch.meshgrid(
                torch.arange(rows, device=device),
                torch.arange(cols, device=device)
            )
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)

            if mode == 'low':
                mask[distance <= low_cutoff] = 1
            elif mode == 'high':
                mask[distance >= high_cutoff] = 1
            elif mode == 'band':
                mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1

            return mask

        low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
        high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
        mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter
        mid_freq_fft = fft_tensor * mid_pass_filter

        low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

        low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
        high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]
        mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[:, :, :N]

        return low_freq_tensor, mid_freq_tensor, high_freq_tensor


if __name__ == '__main__':
    batch_size = 32
    feature_dim = 784
    num_elements = 128

    input_tensor = torch.rand(batch_size, feature_dim, num_elements).to('cuda')

    block = FAM().to('cuda')

    spt_fg_fts = torch.rand(batch_size, feature_dim, num_elements).to('cuda')
    qry_fg_fts = torch.rand(batch_size, feature_dim, num_elements).to('cuda')

    fused_fts_low, fused_fts_mid, fused_fts_high = block(spt_fg_fts, qry_fg_fts)

    print(f"Fused Low Frequency Features: {fused_fts_low.size()}")
    print(f"Fused Mid Frequency Features: {fused_fts_mid.size()}")
    print(f"Fused High Frequency Features: {fused_fts_high.size()}")







    # self.fam = FAM()
    #
    #
    #
    # def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    #     src_fam = src
    #     src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
    #     src = self.norm1(src)
    #     src2 = self.linear2(
    #         self.dropout1(self.activation(self.linear1(src))))
    #
    #     src = src + self.drop_path(self.dropout2(src2))
    #
    #     src5 = self.fam(src_fam,src)
    #
    #     return src5
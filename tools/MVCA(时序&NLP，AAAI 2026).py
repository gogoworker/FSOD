import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class CoherenceAttention(nn.Module):
    """
    Cross-Temporal Spectral Coherence Attention
    """

    def __init__(self, d_model, d_k, hidden_dim):
        super().__init__()
        self.d_k = d_k
        self.hidden_dim = hidden_dim
        self.scale = d_k**-0.5
        self.proj = nn.Linear(d_model, hidden_dim * 3)

        self.var_attn = nn.Parameter(torch.eye(d_model))
        self.var_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, coeffs):
        # coeffs: [B, N, T, D]
        B, N, T, D = (
            coeffs.shape
        )  # B: batch size, N: n_atoms, T: sequence length, D: n_vars

        qkv = self.proj(coeffs)
        qkv = qkv.reshape(B, N, T, self.hidden_dim, 3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # Each [B, N, T, n_heads]
        q, k = [rearrange(embed, "b n l d -> b n d l") for embed in (q, k)]
        v = rearrange(v, "b n l d -> b n d l")

        Q_fft = torch.fft.rfft(q, dim=2)  # [B*T, N, n_heads, K//2+1]
        K_fft = torch.fft.rfft(k, dim=2)

        P_xy = (Q_fft * K_fft.conj()).mean(dim=-2)
        P_xx = (Q_fft * Q_fft.conj()).mean(dim=-2)
        P_yy = (K_fft * K_fft.conj()).mean(dim=-2)

        coherence = P_xy.abs().pow(2) / (P_xx.abs() * P_yy.abs()).clamp(min=1e-6)

        time_attn = F.softmax(coherence / self.scale, dim=-1)
        time_attn = self.dropout(time_attn)

        # Apply attention to values (v)
        out_time = time_attn.unsqueeze(2) * v

        out_time = rearrange(out_time, "b nh hd l -> b l (nh hd)")
        var_attn = F.softmax(self.var_attn, dim=-1)  # [C, C]
        out_var = torch.einsum(
            "b l d, c c -> b l d", out_time, var_attn
        )  # Mix variables
        out_var = rearrange(out_var, "b l (nh hd) -> b l nh hd", nh=self.d_k)

        # Residual connection + MLP
        out = out_var + self.var_mlp(out_var)
        out = self.out_proj(out)
        return out.permute(0, 2, 1, 3)  # [B, T, N, d_model]


if __name__ == "__main__":
    block = CoherenceAttention(d_model=8, d_k=4, hidden_dim=8).to("cuda")

    #  [B, N, T, D]，其中 D=d_model，N=d_k
    input = torch.rand(2, 4, 16, 8, device="cuda")

    output = block(input)

    print(input.size())
    print(output.size())




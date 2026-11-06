from torch import nn
import torch

class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.scale = head_dim ** 0.5

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale_q = q / self.scale
        attn_weights = torch.matmul(scale_q, k.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out
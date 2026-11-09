from torch import nn
import torch

class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        causal: bool = False
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
        self.causal = causal

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale_q = q / self.scale
        attn_weights = torch.matmul(scale_q, k.transpose(-2, -1))
        if self.causal:
            rows, cols = attn_weights.size()[-2:]
            all_positions = torch.ones(rows, cols, device=attn_weights.device)
            mask = torch.triu(all_positions, diagonal=1)
            attn_weights = attn_weights.masked_fill(mask == 1, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = True,
    ):
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise ValueError("d_model must be an integer")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if isinstance(num_heads, bool) or not isinstance(num_heads, int):
            raise ValueError("num_heads must be an integer")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        super().__init__()
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.head_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, D = x.size()
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale_q = q / self.scale
        attn_weights = torch.matmul(scale_q, k.transpose(-2, -1)) # B, num_heads, T, T
        if self.causal:
            rows, cols = attn_weights.size()[-2:]
            all_positions = torch.ones(rows, cols, device=attn_weights.device)
            mask = torch.triu(all_positions, diagonal=1)
            attn_weights = attn_weights.masked_fill(mask == 1, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        return out

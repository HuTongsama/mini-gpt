import torch
from torch import nn
from .attention import MultiHeadAttention

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise ValueError("d_model must be an integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise ValueError("hidden_dim must be an integer")
        if not isinstance(dropout, float):
            raise ValueError("dropout must be a float")
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0)")

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.fc1(x)
        result = self.gelu(result)
        result = self.fc2(result)
        result = self.dropout(result)
        return result

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise ValueError("d_model must be an integer")
        if isinstance(num_heads, bool) or not isinstance(num_heads, int):
            raise ValueError("num_heads must be an integer")
        if isinstance(mlp_hidden_dim, bool) or not isinstance(mlp_hidden_dim, int):
            raise ValueError("mlp_hidden_dim must be an integer")
        if not isinstance(dropout, float):
            raise ValueError("dropout must be a float")
        if not isinstance(causal, bool):
            raise ValueError("causal must be a boolean")
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be a positive integer")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0)")

        self.attn = MultiHeadAttention(d_model, num_heads, causal)
        self.mlp = MLP(d_model, mlp_hidden_dim, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        l1 = self.ln1(x)
        h1 = x + self.attn(l1)
        l2 = self.ln2(h1)
        h2 = h1 + self.mlp(l2)
        return h2
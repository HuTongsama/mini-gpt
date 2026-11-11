import torch
from torch import nn
from .attention import MultiHeadAttention
from .validate import validate_positive_int, validate_range

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):

        validate_positive_int("d_model", d_model)
        validate_positive_int("hidden_dim", hidden_dim)
        validate_range("dropout", dropout, 0.0, 1.0, include_min=True, include_max=False)
        
        super().__init__()
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

        validate_positive_int("d_model", d_model)
        validate_positive_int("num_heads", num_heads)
        validate_positive_int("mlp_hidden_dim", mlp_hidden_dim)
        validate_range("dropout", dropout, 0.0, 1.0, include_min=True, include_max=False)

        super().__init__()
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
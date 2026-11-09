import torch
from torch import nn

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
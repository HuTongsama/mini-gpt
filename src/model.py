import torch
import torch.nn as nn
from .validate import validate_positive_int, validate_range, validate_boolean
from .embedding import InputEmbedding
from .transformer import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        validate_positive_int("vocab_size", vocab_size)
        validate_positive_int("context_size", context_size)
        validate_positive_int("embedding_dim", embedding_dim)
        validate_positive_int("num_layers", num_layers)
        validate_positive_int("num_heads", num_heads)
        validate_positive_int("mlp_hidden_dim", mlp_hidden_dim)
        validate_range("dropout", dropout, 0.0, 1.0, include_min=True, include_max=False)
        validate_boolean("causal", causal)

        super().__init__()
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            context_size=context_size,
            embedding_dim=embedding_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=embedding_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                causal=causal
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)



    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

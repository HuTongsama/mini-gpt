import torch
from torch import nn
from .validate import validate_positive_int

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim):
        
        validate_positive_int("vocab_size", vocab_size)
        validate_positive_int("context_size", context_size)
        validate_positive_int("embedding_dim", embedding_dim)

        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)

    def forward(self, token_ids):

        batch_size, token_size = token_ids.shape
        positions = torch.arange(token_size, device=token_ids.device)

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings # broadcast position embeddings across batch dimension
        return x
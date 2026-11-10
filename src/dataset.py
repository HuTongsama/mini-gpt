from pathlib import Path
from .validate import validate_real_range, validate_non_negative_int, validate_positive_int
from collections.abc import Sequence
import torch
from torch.utils.data import Dataset

def read_text_utf8(path: str | Path) -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()
    return ""

def split_text(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    validate_real_range("train_fraction", train_fraction, 0.0, 1.0, include_min=False, include_max=False)
    split_idx = int(len(text) * train_fraction)
    if split_idx == 0 or split_idx == len(text):
        raise ValueError("text or train_fraction results in an failed split")
    return text[:split_idx], text[split_idx:]


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], context_length: int):

        for i, token in enumerate(token_ids):
            validate_non_negative_int(f"token_ids[{i}]", token)
        validate_positive_int("context_length", context_length)
        if len(token_ids) < context_length + 1:
            raise ValueError("token_ids is too short for the given context_length")
        self.token_ids = token_ids
        self.context_length = context_length
    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        x = self.token_ids[index : index + self.context_length]
        y = self.token_ids[index + 1 : index + 1 + self.context_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
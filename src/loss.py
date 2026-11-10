import torch

def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    if B < 1 or T < 1 or V < 2:
        raise ValueError("Invalid shape for logits tensor")
    B1, T1 = targets.shape
    if B1 != B or T1 != T:
        raise ValueError("Shape of targets does not match shape of logits")
    if targets.dtype != torch.long:
        raise ValueError("Targets must be of dtype torch.long")
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    for i in range(B * T):
        if targets_flat[i] < 0 or targets_flat[i] >= V:
            raise ValueError(f"Invalid target value at index {i}")
    return torch.nn.functional.cross_entropy(logits_flat, targets_flat)

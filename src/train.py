import torch
from torch import nn
from torch.utils.data import DataLoader
from fractions import Fraction
from .loss import next_token_loss
from .validate import validate_range


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:

    if isinstance(learning_rate, Fraction):
        learning_rate = float(learning_rate)
    if isinstance(weight_decay, Fraction):
        weight_decay = float(weight_decay)
    validate_range("learning_rate", learning_rate,
                         min_value=0.0, max_value=None,
                         include_min=False)
    validate_range("weight_decay", weight_decay,
                         min_value=0.0, max_value=None,
                         include_min=True)

    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device = "cpu",
) -> float:

    model.train()
    total_loss = 0.0
    total_k = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = next_token_loss(outputs, y)
        if not torch.isfinite(loss).item():
            raise ValueError("Loss must be finite")
        loss.backward()
        optimizer.step()
        total_k += y.numel()
        total_loss += (loss.item() * y.numel())
    if total_k == 0:
        raise ValueError("Dataloader is empty")
    return total_loss / total_k

def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: str | torch.device = "cpu",
) -> float:

    state = model.training
    try:
        model.eval()
        total_loss = 0.0
        total_k = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = next_token_loss(outputs, y)
                if not torch.isfinite(loss).item():
                    raise ValueError("Loss must be finite")
                total_k += y.numel()
                total_loss += (loss.item() * y.numel())
        if total_k == 0:
            raise ValueError("Dataloader is empty")
        return total_loss / total_k
    
    finally:
        model.train(state)

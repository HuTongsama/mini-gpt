from pathlib import Path
import torch
from torch import nn
from torch.optim import Optimizer

from .validate import validate_non_negative_int, validate_boolean

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    epoch: int,
) -> None:

    validate_non_negative_int("epoch", epoch)
    path = Path(path)
    if not path.parent.exists():
        raise ValueError(f"Directory {path.parent} does not exist")
    if optimizer is None:
        raise ValueError("Optimizer cannot be None")
    checkpoint = {
        "format_version": 1,
        "completed_epochs": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state()
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> int:

    validate_boolean("restore_rng", restore_rng)
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Checkpoint file {path} does not exist")
    checkpoint = torch.load(path, map_location=map_location)
    if "format_version" not in checkpoint or checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format version")
    if "completed_epochs" not in checkpoint:
        raise ValueError("Missing 'completed_epochs' in checkpoint")
    if "model_state" not in checkpoint:
        raise ValueError("Missing 'model_state' in checkpoint")
    if "optimizer_state" not in checkpoint:
        raise ValueError("Missing 'optimizer_state' in checkpoint")
    if "cpu_rng_state" not in checkpoint:
        raise ValueError("Missing 'cpu_rng_state' in checkpoint")
      
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    if restore_rng:
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        
    validate_non_negative_int("completed_epochs", checkpoint["completed_epochs"])
    return checkpoint["completed_epochs"]

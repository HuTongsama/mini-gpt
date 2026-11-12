import torch
from torch import nn
from .validate import validate_non_negative_int, validate_positive_int, validate_range

def generate_greedy(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    
    if not isinstance(prompt_ids, torch.Tensor):
        raise TypeError("prompt_ids must be a torch.Tensor")
    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must have dtype torch.long")
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must be two-dimensional [B, T]")

    B, T = prompt_ids.shape
    validate_range("batch size", B, 1, None)
    validate_range("sequence length", T, 1, None)
    validate_non_negative_int("max_new_tokens", max_new_tokens)
    validate_positive_int("context_size", context_size)

    prompt_ids = prompt_ids.clone()
    if max_new_tokens == 0:
        return prompt_ids
    train_mode = model.training
    try:        
        model.eval()
        iter_count = 0
        while iter_count < max_new_tokens:
            iter_count += 1
            current_ids = prompt_ids[:,-context_size:]
            with torch.no_grad():
                logits = model(current_ids)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("model output must be a torch.Tensor")
                if logits.ndim != 3:
                    raise ValueError("model output must be three-dimensional [B, T, V]")
                if logits.shape[0] != B or \
                    logits.shape[1] != current_ids.shape[1] or \
                    logits.shape[2] <= 0:
                    raise ValueError("model output wrong shape")
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt_ids = torch.cat([prompt_ids, next_tokens], dim=1)
        return prompt_ids
    finally:
        model.train(train_mode)
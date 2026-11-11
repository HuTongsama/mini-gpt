import importlib
import math

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset
from src.model import MiniGPT
from src.train import create_optimizer, evaluate_loss, train_one_epoch


SEED = 43
EPOCHS = 3
LEARNING_RATE = 0.03


def tiny_model():
    return MiniGPT(2, 4, 8, 1, 2, 16, dropout=0.0)


@pytest.fixture
def checkpoint():
    return importlib.import_module("src.checkpoint")


@pytest.fixture
def fitted_model():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            dataset = NextTokenDataset([0, 1] * 6, context_length=4)
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            model = tiny_model()
            optimizer = create_optimizer(model, learning_rate=LEARNING_RATE)
            initial = evaluate_loss(model, loader)
            for _ in range(EPOCHS):
                train_one_epoch(model, loader, optimizer)
            final = evaluate_loss(model, loader)
            yield model, optimizer, loader, initial, final
    finally:
        torch.set_num_threads(original_threads)


def test_stage43_minigpt_fits_the_same_small_dataset(fitted_model):
    """REQ-4.3.6: Measure both losses on identical data with real training."""
    _, _, _, initial, final = fitted_model
    print(f"seed={SEED}, initial={initial:.12f}, final={final:.12f}")
    assert math.isfinite(initial) and math.isfinite(final)
    assert initial > 0
    assert final < 0.8 * initial


def test_stage43_fitted_minigpt_can_resume_training(checkpoint, fitted_model, tmp_path):
    """REQ-4.3.6: The fitted model resumes from its completed epoch and updates."""
    model, optimizer, loader, _, final = fitted_model
    path = tmp_path / "fitted"
    checkpoint.save_checkpoint(path, model, optimizer, epoch=EPOCHS)
    restored = tiny_model()
    restored_optimizer = create_optimizer(restored, learning_rate=LEARNING_RATE)
    assert checkpoint.load_checkpoint(path, restored, restored_optimizer) == EPOCHS
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
    assert evaluate_loss(restored, loader) == pytest.approx(final, rel=1e-6, abs=1e-7)
    before = [parameter.detach().clone() for parameter in restored.parameters()]
    steps_before = [state["step"].item() for state in restored_optimizer.state.values()]
    assert steps_before and all(step == EPOCHS * len(loader) for step in steps_before)

    assert math.isfinite(train_one_epoch(restored, loader, restored_optimizer))

    assert any(
        not torch.equal(parameter, old)
        for parameter, old in zip(restored.parameters(), before, strict=True)
    )
    assert [state["step"].item() for state in restored_optimizer.state.values()] == [
        step + len(loader) for step in steps_before
    ]

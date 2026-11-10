import math

import pytest
import torch
from torch.utils.data import DataLoader

from src.model import MiniGPT


@pytest.fixture
def loss_function():
    from src.loss import next_token_loss

    return next_token_loss


@pytest.fixture
def fixed_inputs():
    logits = torch.tensor(
        [
            [[2, 0, -1], [-1, 1, 2], [0, 0, 0]],
            [[0.5, -0.5, 1.5], [3, 1, -2], [-2, 2, 0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)
    return logits, targets


def scalar_cross_entropy(logits, targets):
    """Stable scalar logsumexp, averaged across every Batch and Token position."""
    losses = []
    for sample, sample_targets in zip(logits.tolist(), targets.tolist(), strict=True):
        for row, target in zip(sample, sample_targets, strict=True):
            largest = max(row)
            losses.append(
                largest + math.log(math.fsum(math.exp(value - largest) for value in row))
                - row[target]
            )
    return math.fsum(losses) / len(losses)


def test_stage41_loss_is_mean_cross_entropy_over_every_position(loss_function, fixed_inputs):
    """REQ-4.1.5: Different losses at six positions expose last-position-only reduction."""
    logits, targets = fixed_inputs
    expected = scalar_cross_entropy(logits, targets)
    last_only = scalar_cross_entropy(logits[:, -1:], targets[:, -1:])
    assert abs(expected - last_only) > 0.1

    actual = loss_function(logits, targets)

    assert isinstance(actual, torch.Tensor)
    assert actual.ndim == 0
    assert actual.is_floating_point()
    assert torch.isfinite(actual)
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    ("logits", "targets", "expected"),
    [
        ([[[0.0, 0.0]]], [[1]], math.log(2)),
        (
            [[[1000.0, 1001.0, -1000.0], [0.0, -1000.0, 1000.0]]],
            [[0, 1]],
            (math.log1p(math.exp(1)) + 2000) / 2,
        ),
    ],
    ids=["minimum-shape", "large-shifted-logits"],
)
def test_stage41_loss_handles_boundary_shape_and_large_logits(
    loss_function, logits, targets, expected
):
    """REQ-4.1.5: Hand calculations expose unstable exponentials and pre-Softmax."""
    logits = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    actual = loss_function(logits, targets)

    assert actual.ndim == 0
    assert actual.is_floating_point()
    assert torch.isfinite(actual)
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-6, atol=1e-6
    )


def test_stage41_loss_logits_gradient_matches_analytic_reference(loss_function, fixed_inputs):
    """REQ-4.1.6: dL/dlogit = (probability - one_hot_target)/(B*T) at every position."""
    logits, targets = fixed_inputs
    logits.requires_grad_()
    count = targets.numel()
    expected_samples = []
    for sample, sample_targets in zip(logits.tolist(), targets.tolist(), strict=True):
        expected_rows = []
        for row, target in zip(sample, sample_targets, strict=True):
            largest = max(row)
            masses = [math.exp(value - largest) for value in row]
            total = math.fsum(masses)
            expected_rows.append(
                [(mass / total - int(index == target)) / count
                 for index, mass in enumerate(masses)]
            )
        expected_samples.append(expected_rows)
    expected = torch.tensor(expected_samples, dtype=torch.float32)

    loss_function(logits, targets).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("layout", ["logits", "targets"])
def test_stage41_loss_accepts_noncontiguous_inputs(loss_function, fixed_inputs, layout):
    """REQ-4.1.5: Valid shape and dtype do not require contiguous Tensor storage."""
    logits, targets = fixed_inputs
    if layout == "logits":
        logits = logits.transpose(0, 1).contiguous().transpose(0, 1)
        assert not logits.is_contiguous()
    else:
        targets = targets.transpose(0, 1).contiguous().transpose(0, 1)
        assert not targets.is_contiguous()
    expected = scalar_cross_entropy(logits, targets)

    actual = loss_function(logits, targets)

    assert actual.ndim == 0
    assert actual.is_floating_point()
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-6, atol=1e-6
    )


def test_stage41_loss_rejects_uint8_targets(loss_function, fixed_inputs):
    """REQ-4.1.5: Targets must be torch.long even if a backend accepts byte indices."""
    logits, targets = fixed_inputs

    with pytest.raises(Exception):
        loss_function(logits, targets.to(torch.uint8))


@pytest.mark.parametrize(
    ("logits", "targets"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long)),
        (torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 1, dtype=torch.long)),
        (torch.zeros(2, 2, 3), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 1, dtype=torch.long)),
        (torch.zeros(0, 2, 3), torch.zeros(0, 2, dtype=torch.long)),
        (torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.long)),
        (torch.zeros(1, 2, 0), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, 1), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.float32)),
        (torch.zeros(1, 2, 3), torch.tensor([[0, -1]], dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.tensor([[0, 3]], dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.tensor([[0, -100]], dtype=torch.long)),
        ([[[0.0, 0.0, 0.0]]], torch.zeros(1, 1, dtype=torch.long)),
        (torch.zeros(1, 2, 3), [[0, 1]]),
    ],
    ids=[
        "logits-rank-two", "logits-rank-four", "targets-rank-one", "targets-rank-three",
        "batch-mismatch", "time-mismatch", "empty-batch", "empty-sequence",
        "empty-vocabulary", "one-class", "integer-logits", "float-targets",
        "negative-target", "target-at-vocab-size", "negative-ignore-index",
        "nontensor-logits", "nontensor-targets",
    ],
)
def test_stage41_loss_rejects_invalid_types_shapes_and_targets(
    loss_function, logits, targets
):
    """REQ-4.1.5: No broadcasting, empty means, target coercion or implicit ignore index."""
    with pytest.raises(Exception):
        loss_function(logits, targets)


def test_stage41_dataset_batch_model_logits_and_loss_integrate(loss_function):
    """REQ-4.1.6: Existing MiniGPT can consume a batch and produce a scalar loss."""
    from src.dataset import NextTokenDataset

    dataset = NextTokenDataset([0, 1, 2, 3, 0, 2, 1], context_length=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = MiniGPT(
            vocab_size=4, context_size=3, embedding_dim=8,
            num_layers=1, num_heads=2, mlp_hidden_dim=12,
            dropout=0.0, causal=True,
        ).eval()

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            assert x.dtype == y.dtype == torch.long
            assert x.shape == y.shape == (2, 3)
            assert logits.shape == (2, 3, 4)
            assert torch.isfinite(logits).all()
            expected = scalar_cross_entropy(logits, y)
            actual = loss_function(logits, y)

            assert actual.ndim == 0
            assert actual.is_floating_point()
            assert torch.isfinite(actual)
            torch.testing.assert_close(
                actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-6, atol=1e-6
            )

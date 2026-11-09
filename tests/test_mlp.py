import math

import pytest
import torch
from torch import nn


FC1_WEIGHT = ((1.0, 0.5), (-0.5, 1.0), (1.0, -1.0))
FC1_BIAS = (0.25, -0.5, 0.75)
FC2_WEIGHT = ((1.0, -0.5, 0.25), (-0.75, 0.5, 1.0))
FC2_BIAS = (0.5, -0.25)


@pytest.fixture
def mlp_class():
    # Defer the import so tests can be collected before the target file exists.
    from src.transformer import MLP

    return MLP


@pytest.fixture
def make_mlp(mlp_class):
    def make(d_model, hidden_dim, **options):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            return mlp_class(d_model=d_model, hidden_dim=hidden_dim, **options)

    return make


@pytest.fixture
def fixed_mlp(make_mlp):
    def make(**options):
        mlp = make_mlp(d_model=2, hidden_dim=3, **options)
        with torch.no_grad():
            mlp.fc1.weight.copy_(torch.tensor(FC1_WEIGHT, dtype=torch.float32))
            mlp.fc1.bias.copy_(torch.tensor(FC1_BIAS, dtype=torch.float32))
            mlp.fc2.weight.copy_(torch.tensor(FC2_WEIGHT, dtype=torch.float32))
            mlp.fc2.bias.copy_(torch.tensor(FC2_BIAS, dtype=torch.float32))
        return mlp

    return make


@pytest.fixture
def hidden_states():
    return torch.tensor(
        [
            [[1, -1], [0, 2], [-2, 0.5]],
            [[0, 2], [-1, -0.5], [1, 0]],
        ],
        dtype=torch.float32,
    )


def exact_gelu(value):
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def scalar_reference(x):
    """Evaluate the fixed small network using Python scalars and exact GELU."""
    batches = []
    for sample in x.tolist():
        outputs = []
        for token in sample:
            hidden = [
                exact_gelu(math.sumprod(token, row) + bias)
                for row, bias in zip(FC1_WEIGHT, FC1_BIAS, strict=True)
            ]
            outputs.append(
                [
                    math.sumprod(hidden, row) + bias
                    for row, bias in zip(FC2_WEIGHT, FC2_BIAS, strict=True)
                ]
            )
        batches.append(outputs)
    return torch.tensor(batches, dtype=torch.float32)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "d_model", "hidden_dim"),
    [(2, 16, 128, 512), (1, 1, 1, 1), (3, 2, 3, 2)],
    ids=["initial-config", "single-token-scalar", "narrow-hidden-layer"],
)
def test_stage31_shapes_and_public_trainable_biased_projections(
    make_mlp, batch_size, sequence_length, d_model, hidden_dim
):
    """REQ-3.1.1, REQ-3.1.2: D need not equal H, and every token keeps D outputs."""
    mlp = make_mlp(d_model=d_model, hidden_dim=hidden_dim)
    assert isinstance(mlp, nn.Module)
    for name, output_dim, input_dim in (
        ("fc1", hidden_dim, d_model), ("fc2", d_model, hidden_dim)
    ):
        projection = getattr(mlp, name)
        assert isinstance(projection, nn.Linear)
        assert projection.weight.shape == (output_dim, input_dim)
        assert projection.bias is not None
        assert projection.bias.shape == (output_dim,)
        assert projection.weight.requires_grad
        assert projection.bias.requires_grad
    assert mlp.fc1 is not mlp.fc2
    shape = (batch_size, sequence_length, d_model)
    x = torch.linspace(-1, 1, math.prod(shape), dtype=torch.float32).reshape(shape)

    actual = mlp(x)

    assert actual.shape == shape
    assert torch.isfinite(actual).all()


def test_stage31_fixed_parameters_and_default_dropout_match_reference(
    fixed_mlp, hidden_states
):
    """REQ-3.1.2, REQ-3.1.3, REQ-3.1.4: Both biases and exact GELU precede output."""
    mlp = fixed_mlp()
    expected = scalar_reference(hidden_states)

    for training in (True, False):
        mlp.train(training)
        for _ in range(2):
            torch.testing.assert_close(
                mlp(hidden_states), expected, rtol=1e-5, atol=1e-6
            )


def test_stage31_gelu_is_exact_and_between_the_projections(make_mlp):
    """REQ-3.1.3: Negative inputs distinguish GELU from ReLU and identity."""
    mlp = make_mlp(d_model=1, hidden_dim=1)
    with torch.no_grad():
        mlp.fc1.weight.fill_(1)
        mlp.fc1.bias.zero_()
        mlp.fc2.weight.fill_(1)
        mlp.fc2.bias.zero_()
    values = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    x = torch.tensor([[[value] for value in values]], dtype=torch.float32)
    expected = torch.tensor(
        [[[exact_gelu(value)] for value in values]], dtype=torch.float32
    )

    torch.testing.assert_close(mlp(x), expected, rtol=1e-6, atol=1e-6)


def test_stage31_tokens_share_weights_without_mixing_positions(
    fixed_mlp, hidden_states
):
    """REQ-3.1.1: Identical tokens share outputs, and an edit stays at its position."""
    mlp = fixed_mlp()
    changed = hidden_states.clone()
    changed[0, 1] = torch.tensor([-3.0, 1.0], dtype=torch.float32)
    original_expected = scalar_reference(hidden_states)
    changed_expected = scalar_reference(changed)
    assert (changed_expected[0, 1] - original_expected[0, 1]).abs().max() > 1e-3

    for x, expected in (
        (hidden_states, original_expected), (changed, changed_expected)
    ):
        torch.testing.assert_close(mlp(x), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("probability", [0.25, 0.75])
def test_stage31_dropout_is_after_fc2_and_obeys_train_eval(
    fixed_mlp, probability
):
    """REQ-3.1.4: Complete output coordinates are dropped or scaled by 1/(1-p)."""
    mlp = fixed_mlp(dropout=probability)
    x = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32).expand(16, 16, 2)
    baseline = scalar_reference(x)
    assert baseline.abs().min().item() > 0.1
    mlp.eval()
    for _ in range(2):
        torch.testing.assert_close(mlp(x), baseline, rtol=1e-5, atol=1e-6)

    mlp.train()
    expected_retained = baseline / (1.0 - probability)
    observations = []
    for seed in (0, 17, 42):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            actual = mlp(x)
        retained = actual != 0
        expected = torch.where(retained, expected_retained, torch.zeros_like(baseline))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        observations.append(retained)
    observed = torch.stack(observations)
    assert observed.any(dim=(0, 1, 2)).all()
    assert (~observed).any(dim=(0, 1, 2)).all()

    # A separate bias-only output makes dropping only the hidden activations
    # incapable of producing either the required zeros or the scaled biases.
    with torch.no_grad():
        mlp.fc2.weight.zero_()
    bias_baseline = torch.tensor(FC2_BIAS, dtype=torch.float32).expand_as(x)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        actual = mlp(x)
    retained = actual != 0
    expected = torch.where(
        retained, bias_baseline / (1.0 - probability), torch.zeros_like(bias_baseline)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_stage31_backward_reaches_input_and_both_weights_and_biases(
    fixed_mlp, hidden_states
):
    """REQ-3.1.5: A nondegenerate weighted loss exercises every trainable part."""
    mlp = fixed_mlp()
    x = hidden_states.requires_grad_()
    coefficients = torch.arange(1, 13, dtype=torch.float32).reshape(2, 3, 2)

    (mlp(x) * coefficients).sum().backward()

    for projection in (mlp.fc1, mlp.fc2):
        for parameter in (projection.weight, projection.bias):
            assert parameter.grad is not None
            assert parameter.grad.shape == parameter.shape
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad).item() > 0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad).item() > 0


@pytest.mark.parametrize(
    ("d_model", "hidden_dim", "dropout"),
    [
        (0, 3, 0.0), (-2, 3, 0.0), (1.5, 3, 0.0),
        (2, 0, 0.0), (2, -3, 0.0), (2, 1.5, 0.0),
        (2, 3, -0.1), (2, 3, 1.0), (2, 3, 1.1),
        (2, 3, float("nan")), (2, 3, float("inf")), (2, 3, -float("inf")),
    ],
    ids=[
        "zero-model", "negative-model", "fractional-model",
        "zero-hidden", "negative-hidden", "fractional-hidden",
        "negative-dropout", "dropout-one", "dropout-above-one",
        "nan-dropout", "positive-inf-dropout", "negative-inf-dropout",
    ],
)
def test_stage31_rejects_invalid_constructor_parameters(
    mlp_class, d_model, hidden_dim, dropout
):
    """REQ-3.1.6: Reject invalid dimensions and probabilities outside [0,1)."""
    with pytest.raises(Exception):
        mlp_class(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)

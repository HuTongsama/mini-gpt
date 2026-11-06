import math

import pytest
import torch
from torch import nn


PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj")
FIXED_WEIGHTS = (
    ((0.5, -0.25, 0.75, 0.0), (0.0, 0.5, -0.25, 0.25)),
    ((-0.5, 0.75, 0.0, 0.25), (0.25, 0.0, 0.5, -0.5)),
    ((0.25, 0.5, -0.5, 0.75), (0.75, -0.25, 0.25, 0.0)),
)


@pytest.fixture
def attention_class():
    # Defer the import so the suite remains collectable before implementation.
    from src.attention import SingleHeadAttention

    return SingleHeadAttention


@pytest.fixture
def make_attention(attention_class):
    def make(d_model, head_dim, weights):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            attention = attention_class(d_model=d_model, head_dim=head_dim)
        with torch.no_grad():
            for name, weight in zip(PROJECTION_NAMES, weights, strict=True):
                projection = getattr(attention, name)
                fixed_weight = torch.tensor(weight, dtype=torch.float32)
                assert projection.weight.shape == fixed_weight.shape
                projection.weight.copy_(fixed_weight)
        return attention

    return make


@pytest.fixture
def hidden_states():
    return torch.tensor(
        [
            [[1, 0, 2, -1], [0, 1, -1, 2], [2, -1, 0, 1]],
            [[-1, 1, 0, 2], [2, 0, 1, -1], [0, -2, 1, 1]],
        ],
        dtype=torch.float32,
    )


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def reference_attention(sample, weights):
    """Compute one sample with scalar arithmetic and explicit key iteration."""
    queries, keys, values = (
        [[dot(token, row) for row in projection] for token in sample]
        for projection in weights
    )
    scale = math.sqrt(len(queries[0]))
    output = []
    for query in queries:
        scores = [dot(query, key) / scale for key in keys]
        largest = max(scores)
        numerators = [math.exp(score - largest) for score in scores]
        denominator = math.fsum(numerators)
        output.append(
            [
                dot(numerators, feature) / denominator
                for feature in zip(*values, strict=True)
            ]
        )
    return output


def test_qkv_projections_are_independent_bias_free_linears(
    make_attention, hidden_states
):
    """REQ-2.1.1: The public projections are trainable and shape-preserving over B,T."""
    attention = make_attention(d_model=4, head_dim=2, weights=FIXED_WEIGHTS)
    assert isinstance(attention, nn.Module)
    projections = [getattr(attention, name) for name in PROJECTION_NAMES]

    for projection in projections:
        assert isinstance(projection, nn.Linear)
        assert projection.bias is None
        assert projection.weight.requires_grad
        assert projection(hidden_states).shape == (2, 3, 2)
    assert len({id(projection.weight) for projection in projections}) == 3


def test_attention_matches_scalar_reference_across_independent_batches(
    make_attention, hidden_states
):
    """REQ-2.1.1–REQ-2.1.4: Distinct batches agree with per-sample scalar results."""
    attention = make_attention(d_model=4, head_dim=2, weights=FIXED_WEIGHTS)
    expected = torch.tensor(
        [reference_attention(sample, FIXED_WEIGHTS) for sample in hidden_states.tolist()],
        dtype=torch.float32,
    )

    actual = attention(hidden_states)

    assert actual.shape == (2, 3, 2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("query_multiplier", [1, 200], ids=["ordinary", "large-logits"])
def test_attention_scales_scores_and_normalizes_over_keys(
    make_attention, query_multiplier
):
    """REQ-2.1.2–REQ-2.1.4: Basis value vectors expose the weight for each key."""
    weights = (
        (
            (query_multiplier, 0, query_multiplier),
            (0, query_multiplier, -query_multiplier),
            (0, 0, 0),
            (0, 0, 0),
        ),
        ((2, 0, -2), (0, 2, 1), (0, 0, 0), (0, 0, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)),
    )
    attention = make_attention(d_model=3, head_dim=4, weights=weights)
    # For one-hot inputs, the hand-calculated dot products divided by sqrt(4).
    scaled_rows = ((1, 0, -1), (0, 1, 0.5), (1, -1, -1.5))
    expected_rows = []
    for row in scaled_rows:
        numerators = [math.exp(query_multiplier * (score - max(row))) for score in row]
        expected_rows.append([value / math.fsum(numerators) for value in numerators])
    expected = torch.tensor([expected_rows], dtype=torch.float32)

    actual = attention(torch.eye(3, dtype=torch.float32).unsqueeze(0))

    assert actual.shape == (1, 3, 4)
    assert torch.isfinite(actual).all()
    actual_weights = actual[..., :3]
    torch.testing.assert_close(actual_weights, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_weights.sum(dim=-1), torch.ones_like(actual[..., 3]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        actual[..., 3], torch.zeros_like(actual[..., 3]), rtol=0, atol=1e-6
    )


def test_attention_stays_finite_when_only_unscaled_scores_overflow(make_attention):
    """REQ-2.1.2–REQ-2.1.4: Scaled scores and expected outputs fit in float32."""
    qk_weight = ((1, 0), (0, 0), (0, 0), (0, 0))
    v_weight = ((0, 1), (0, 0), (0, 0), (0, 0))
    attention = make_attention(
        d_model=2, head_dim=4, weights=(qk_weight, qk_weight, v_weight)
    )
    x = torch.tensor([[[2e19, 1], [-2e19, 3]]], dtype=torch.float32)
    # Dot products are about +/-4e38 before scaling and +/-2e38 after.
    # Each query assigns weight 1 (to float32 precision) to its own value.
    expected = torch.tensor([[[1, 0, 0, 0], [3, 0, 0, 0]]], dtype=torch.float32)

    actual = attention(x)

    assert torch.isfinite(actual).all(), actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_single_token_returns_its_value_projection(make_attention):
    """REQ-2.1.3, REQ-2.1.4: At B=T=d_model=head_dim=1 the only weight is one."""
    attention = make_attention(d_model=1, head_dim=1, weights=(((2,),), ((-3,),), ((0.75,),)))
    x = torch.tensor([[[2.0]]], dtype=torch.float32)
    expected = torch.tensor([[[1.5]]], dtype=torch.float32)

    torch.testing.assert_close(attention(x), expected, rtol=0, atol=1e-6)


def test_backward_reaches_all_three_projection_weights(make_attention, hidden_states):
    """REQ-2.1.5: A nonconstant loss produces finite, nonzero gradients for Q,K,V."""
    attention = make_attention(d_model=4, head_dim=2, weights=FIXED_WEIGHTS)
    coefficients = torch.arange(1, 13, dtype=torch.float32).reshape(2, 3, 2)

    (attention(hidden_states) * coefficients).sum().backward()

    for name in PROJECTION_NAMES:
        weight = getattr(attention, name).weight
        assert weight.grad is not None, name
        assert weight.grad.shape == (2, 4), name
        assert torch.isfinite(weight.grad).all(), name
        assert torch.count_nonzero(weight.grad).item() > 0, name


@pytest.mark.parametrize(
    ("d_model", "head_dim"),
    [(0, 2), (-1, 2), (1.5, 2), (2, 0), (2, -1), (2, 1.5)],
    ids=[
        "zero-d-model",
        "negative-d-model",
        "fractional-d-model",
        "zero-head-dim",
        "negative-head-dim",
        "fractional-head-dim",
    ],
)
def test_constructor_rejects_invalid_dimensions(attention_class, d_model, head_dim):
    """Stage 2.1 §2 and §4: Both dimensions must be positive integers."""
    with pytest.raises(Exception):
        attention_class(d_model=d_model, head_dim=head_dim)

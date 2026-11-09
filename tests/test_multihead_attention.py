import math

import pytest
import torch
from torch import nn


PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj")
FIXED_WEIGHTS = (
    ((1, 0, 0.5, 0), (0, 1, 0, -0.5), (0, 0.5, -1, 0), (0.5, 0, 0, 1)),
    ((0.5, 0, 0, 1), (0, -1, 0.5, 0), (1, 0, 0.5, 0), (0, 1, 0, 0.5)),
    ((1, 0, 0, 0.5), (0, 0.5, 1, 0), (0.5, 0, 0, -1), (0, -0.5, 0.5, 0)),
    ((0, 1, 0.5, 0), (1, 0, 0, -0.5), (0, 0, -1, 0.5), (0.5, 0, 0, 1)),
)


@pytest.fixture
def multihead_class():
    # Keep collection and the older suites usable before the new class exists.
    from src.attention import MultiHeadAttention

    return MultiHeadAttention


@pytest.fixture
def make_multihead(multihead_class):
    def make(d_model, num_heads, weights=None, **options):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            attention = multihead_class(d_model=d_model, num_heads=num_heads, **options)
        if weights is not None:
            with torch.no_grad():
                for name, weight in zip(PROJECTION_NAMES, weights, strict=True):
                    parameter = getattr(attention, name).weight
                    fixed_weight = torch.as_tensor(weight, dtype=torch.float32)
                    assert parameter.shape == fixed_weight.shape
                    parameter.copy_(fixed_weight)
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


def scalar_reference(x, weights, num_heads, causal):
    """Evaluate each head from its weight rows, with no batched reshape or mask."""
    q_weight, k_weight, v_weight, out_weight = weights
    head_dim = len(q_weight) // num_heads
    batches = []
    for sample in x.tolist():
        combined = [[] for _ in sample]
        for start in range(0, len(q_weight), head_dim):
            stop = start + head_dim
            queries, keys, values = (
                [
                    [math.sumprod(token, row) for row in weight[start:stop]]
                    for token in sample
                ]
                for weight in (q_weight, k_weight, v_weight)
            )
            for index, query in enumerate(queries):
                key_count = index + 1 if causal else len(sample)
                scores = [
                    math.sumprod(query, key) / math.sqrt(head_dim)
                    for key in keys[:key_count]
                ]
                largest = max(scores)
                masses = [math.exp(score - largest) for score in scores]
                total = math.fsum(masses)
                for value_column in zip(*values[:key_count], strict=True):
                    combined[index].append(math.sumprod(masses, value_column) / total)
        batches.append(
            [[math.sumprod(token, row) for row in out_weight] for token in combined]
        )
    return torch.tensor(batches, dtype=torch.float32)


@pytest.mark.parametrize(
    ("d_model", "num_heads"),
    [(0, 2), (-4, 2), (4.0, 2), (4, 0), (4, -2), (4, 2.0), (5, 2)],
    ids=[
        "zero-model",
        "negative-model",
        "float-model",
        "zero-heads",
        "negative-heads",
        "float-heads",
        "non-divisible",
    ],
)
def test_stage23_rejects_invalid_dimensions(multihead_class, d_model, num_heads):
    """REQ-2.3.6: Dimensions must be positive integers and divide exactly."""
    with pytest.raises(Exception):
        multihead_class(d_model=d_model, num_heads=num_heads)


def test_stage23_four_head_public_projections_and_variable_shapes(make_multihead):
    """REQ-2.3.1, REQ-2.3.4, REQ-2.3.6: D=128 and H=4 support multiple B,T."""
    attention = make_multihead(d_model=128, num_heads=4)
    assert isinstance(attention, nn.Module)
    for name in PROJECTION_NAMES:
        projection = getattr(attention, name)
        assert isinstance(projection, nn.Linear)
        assert projection.weight.shape == (128, 128)
        assert projection.weight.requires_grad
        assert projection.bias is None

    # Reuse one module across lengths so stale, fixed-size masks are observable.
    for batch_size, sequence_length in ((1, 1), (2, 5), (3, 2)):
        shape = (batch_size, sequence_length, 128)
        x = torch.linspace(-1, 1, math.prod(shape), dtype=torch.float32).reshape(shape)
        actual = attention(x)

        assert actual.shape == shape
        assert actual.dtype == torch.float32
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    ("options", "causal"),
    [
        pytest.param({}, True, id="default-causal"),
        pytest.param({"causal": False}, False, id="explicit-bidirectional"),
    ],
)
def test_stage23_fixed_multihead_output_matches_scalar_reference(
    make_multihead, hidden_states, options, causal
):
    """REQ-2.3.1 through REQ-2.3.5: Distinct heads and samples match the oracle."""
    attention = make_multihead(4, 2, FIXED_WEIGHTS, **options)
    expected = scalar_reference(hidden_states, FIXED_WEIGHTS, num_heads=2, causal=causal)

    actual = attention(hidden_states)

    assert actual.shape == (2, 3, 4)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_stage23_each_head_exposes_normalized_causal_weights(make_multihead):
    """REQ-2.3.2, REQ-2.3.3: Basis Values expose both [T,T] weight matrices."""
    zero_row = (0,) * 6
    q = (
        (1, 2, -1, 0, 0, 0), zero_row, zero_row,
        (2, -1, 1, 0, 0, 0), zero_row, zero_row,
    )
    k = (
        (1, 0, -2, 0, 0, 0), zero_row, zero_row,
        (0, 2, 1, 0, 0, 0), zero_row, zero_row,
    )
    basis = torch.eye(6, dtype=torch.float32)
    v = basis[:3].repeat(2, 1)
    weights = (q, k, v.tolist(), basis.tolist())
    attention = make_multihead(6, 2, weights, causal=True)
    x = basis[:3].unsqueeze(0)
    expected = scalar_reference(x, weights, num_heads=2, causal=True)

    actual = attention(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for observed in actual[0].split(3, dim=-1):
        assert torch.count_nonzero(observed.triu(diagonal=1)).item() == 0
        row_sums = observed.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-6)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
def test_stage23_future_edit_has_the_expected_effect_on_the_prefix(
    make_multihead, hidden_states, causal
):
    """REQ-2.3.3, REQ-2.3.5: A suffix change is visible only in bidirectional mode."""
    attention = make_multihead(4, 2, FIXED_WEIGHTS, causal=causal).eval()
    changed = hidden_states.clone()
    changed[0, 2] = torch.tensor([-3, 2, 4, -1], dtype=torch.float32)
    original_prefix = scalar_reference(hidden_states, FIXED_WEIGHTS, 2, causal)[:, :2]
    changed_prefix = scalar_reference(changed, FIXED_WEIGHTS, 2, causal)[:, :2]
    if causal:
        torch.testing.assert_close(changed_prefix, original_prefix, rtol=0, atol=0)
    else:
        assert (changed_prefix[0] - original_prefix[0]).abs().max().item() > 1e-3

    with torch.no_grad():
        torch.testing.assert_close(
            attention(hidden_states)[:, :2], original_prefix, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            attention(changed)[:, :2], changed_prefix, rtol=1e-5, atol=1e-6
        )


def test_stage23_single_head_still_applies_output_projection(make_multihead, hidden_states):
    """REQ-2.3.2, REQ-2.3.4: One head reduces to single-head math plus out_proj."""
    attention = make_multihead(4, 1, FIXED_WEIGHTS, causal=False)
    expected = scalar_reference(hidden_states, FIXED_WEIGHTS, num_heads=1, causal=False)

    torch.testing.assert_close(attention(hidden_states), expected, rtol=1e-5, atol=1e-6)


def test_stage23_prefix_loss_reaches_all_projections_without_future_gradient(
    make_multihead, hidden_states
):
    """REQ-2.3.5: Prefix loss backpropagates through QKV and output weights."""
    attention = make_multihead(4, 2, FIXED_WEIGHTS, causal=True)
    x = hidden_states.requires_grad_()
    coefficients = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 4)

    (attention(x)[:, :2] * coefficients).sum().backward()

    for name in PROJECTION_NAMES:
        gradient = getattr(attention, name).weight.grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient).item() > 0, name
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :2].abs().max().item() > 1e-6
    future_gradient = x.grad[:, 2:]
    torch.testing.assert_close(
        future_gradient, torch.zeros_like(future_gradient), rtol=0, atol=1e-7
    )


def test_stage23_causal_output_stays_finite_at_scaling_boundary(make_multihead):
    """REQ-2.3.2, REQ-2.3.3, REQ-2.3.6: Scaled scores fit despite raw overflow."""
    qk = torch.zeros((8, 8), dtype=torch.float32)
    qk[[0, 4], 0] = 1
    v = torch.zeros_like(qk)
    v[0, 1] = 1
    v[4, 1] = -2
    attention = make_multihead(8, 2, (qk, qk, v, torch.eye(8)), causal=True)
    x = torch.zeros((1, 2, 8), dtype=torch.float32)
    x[0, :, :2] = torch.tensor([[2e19, 1], [2.4e19, 3]], dtype=torch.float32)
    # Each head selects Key 0 in the first row and Key 1 in the second row.
    expected = torch.tensor(
        [[[1, 0, 0, 0, -2, 0, 0, 0], [3, 0, 0, 0, -6, 0, 0, 0]]], dtype=torch.float32
    )

    actual = attention(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

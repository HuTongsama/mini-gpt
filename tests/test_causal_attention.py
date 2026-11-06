import math

import pytest
import torch

from src.attention import SingleHeadAttention


# For x = I, the first three output coordinates expose the three Key weights.
PROBE_WEIGHTS = (
    ((1, 0, 1), (0, 1, -1), (0, 0, 0), (0, 0, 0)),
    ((2, 0, -2), (0, 2, 1), (0, 0, 0), (0, 0, 0)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)),
)


@pytest.fixture
def make_stage22_attention():
    def make(weights, **options):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            attention = SingleHeadAttention(
                d_model=len(weights[0][0]), head_dim=len(weights[0]), **options
            )
        with torch.no_grad():
            for projection, weight in zip(
                (attention.q_proj, attention.k_proj, attention.v_proj),
                weights,
                strict=True,
            ):
                projection.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        return attention

    return make


def expected_probe_output(*score_rows):
    # Score rows contain only allowed Keys. Value vectors are basis vectors,
    # so the weighted output is the normalized row padded with zeros.
    rows = []
    for scores in score_rows:
        exponentials = [math.exp(score) for score in scores]
        denominator = math.fsum(exponentials)
        rows.append(
            [value / denominator for value in exponentials] + [0.0] * (4 - len(scores))
        )
    return torch.tensor([rows], dtype=torch.float32)


@pytest.mark.parametrize(
    "options", [{}, {"causal": False}], ids=["default", "explicit-false"]
)
def test_stage22_noncausal_modes_preserve_bidirectional_output(
    make_stage22_attention, options
):
    """REQ-2.2.1: Default and explicit False include future Keys."""
    attention = make_stage22_attention(PROBE_WEIGHTS, **options)
    x = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    expected = expected_probe_output((1, 0, -1), (0, 1, 0.5), (1, -1, -1.5))

    torch.testing.assert_close(attention(x), expected, rtol=1e-5, atol=1e-6)


def test_stage22_causal_weights_are_zero_in_the_future_and_normalized(
    make_stage22_attention,
):
    """REQ-2.2.2, REQ-2.2.3: Each Query normalizes over its inclusive prefix."""
    attention = make_stage22_attention(PROBE_WEIGHTS, causal=True)
    x = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    # Hand-computed QK scores divided by sqrt(4), keeping only the legal Keys.
    expected = expected_probe_output((1,), (0, 1), (1, -1, -1.5))

    actual = attention(x)

    assert actual.shape == (1, 3, 4)
    observed_weights = actual[..., :3]
    assert torch.count_nonzero(observed_weights.triu(diagonal=1)).item() == 0
    row_sums = observed_weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_stage22_causal_output_preserves_prefix_when_suffix_changes(
    make_stage22_attention,
):
    """REQ-2.2.2, REQ-2.2.4: Two future tokens cannot affect either prefix row."""
    attention = make_stage22_attention(PROBE_WEIGHTS, causal=True).eval()
    x = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]]], dtype=torch.float32)
    changed = x.clone()
    changed[0, 2:] = torch.tensor([[-2, 3, 4], [3, -1, 5]], dtype=torch.float32)
    expected_prefix = expected_probe_output((1,), (0, 1))

    with torch.no_grad():
        for sample in (x, changed):
            torch.testing.assert_close(
                attention(sample)[:, :2], expected_prefix, rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize(
    ("batch_size", "sequence_length"),
    [(1, 1), (2, 2), (3, 5), (2, 64)],
    ids=["single-token", "two-tokens", "five-tokens", "initial-context-length"],
)
def test_stage22_causal_shapes_and_batch_isolation(
    make_stage22_attention, batch_size, sequence_length
):
    """REQ-2.2.5: Uniform scores give a separate running mean for each sample."""
    zero_weight = ((0, 0), (0, 0), (0, 0))
    value_weight = ((1, 0), (0, 2), (0, 0))
    attention = make_stage22_attention(
        (zero_weight, zero_weight, value_weight), causal=True
    )
    positions = torch.arange(1, sequence_length + 1, dtype=torch.float32)
    batch_offsets = 100 * torch.arange(batch_size, dtype=torch.float32).unsqueeze(1)
    first_feature = positions + batch_offsets
    second_feature = 1 - positions - batch_offsets
    x = torch.stack((first_feature, second_feature), dim=-1)
    prefix_lengths = positions.unsqueeze(0)
    expected = torch.stack(
        (
            first_feature.cumsum(dim=1) / prefix_lengths,
            2 * second_feature.cumsum(dim=1) / prefix_lengths,
            torch.zeros_like(first_feature),
        ),
        dim=-1,
    )

    actual = attention(x)

    assert actual.shape == (batch_size, sequence_length, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_stage22_causal_gradients_reach_qkv_without_reaching_future_input(
    make_stage22_attention,
):
    """REQ-2.2.6: Query 1 exercises QKV but has zero gradient at input row 2."""
    attention = make_stage22_attention(PROBE_WEIGHTS, causal=True)
    x = torch.eye(3, dtype=torch.float32).unsqueeze(0).requires_grad_()

    attention(x)[0, 1, 0].backward()

    for name in ("q_proj", "k_proj", "v_proj"):
        gradient = getattr(attention, name).weight.grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient).item() > 0, name
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, 0].abs().max().item() > 1e-6
    torch.testing.assert_close(x.grad[:, 2:], torch.zeros_like(x.grad[:, 2:]), rtol=0, atol=1e-7)


def test_stage22_causal_mode_preserves_stage21_scaling_boundary(make_stage22_attention):
    """REQ-2.2.3 and contract section 3: Finite scaled scores still work with masking."""
    qk_weight = ((1, 0), (0, 0), (0, 0), (0, 0))
    value_weight = ((0, 1), (0, 0), (0, 0), (0, 0))
    attention = make_stage22_attention((qk_weight, qk_weight, value_weight), causal=True)
    x = torch.tensor([[[2e19, 1], [2.4e19, 3]]], dtype=torch.float32)
    # Unscaled products overflow float32; all scores divided by sqrt(4) fit.
    # Query 0 must ignore the larger future score; Query 1 selects its own value.
    expected = torch.tensor([[[1, 0, 0, 0], [3, 0, 0, 0]]], dtype=torch.float32)

    actual = attention(x)

    assert torch.isfinite(actual).all(), actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

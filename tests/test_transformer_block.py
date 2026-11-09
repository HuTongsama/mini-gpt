import math

import pytest
import torch
from torch import nn

from src.attention import MultiHeadAttention
from src.transformer import MLP


NORM_SCALES = ((1.0, 0.5, -0.75, 1.5), (0.5, -1.0, 1.25, 0.75))
NORM_BIASES = ((0.25, -0.5, 0.75, 0.0), (-0.25, 0.5, 0.0, 1.0))


@pytest.fixture
def block_class():
    from src.transformer import TransformerBlock

    return TransformerBlock


@pytest.fixture
def make_block(block_class):
    def make(d_model=4, num_heads=2, mlp_hidden_dim=7, **options):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            return block_class(
                d_model=d_model, num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim, **options
            )

    return make


@pytest.fixture
def hidden_states():
    return torch.tensor(
        [
            [[1, 0, 2, -1], [0, 1, -1, 2], [2, -1, 0, 1]],
            [[-1, 1, 0, 2], [2, 0, 1, -1], [5, 5, 5, 5]],
        ],
        dtype=torch.float32,
    )


def set_norm_parameters(block):
    with torch.no_grad():
        for index, norm in enumerate((block.ln1, block.ln2)):
            norm.weight.copy_(torch.tensor(NORM_SCALES[index], dtype=torch.float32))
            norm.bias.copy_(torch.tensor(NORM_BIASES[index], dtype=torch.float32))


def scalar_layer_norm(x, scales, biases, eps):
    """Normalize each token independently using population variance."""
    batches = []
    for sample in x.tolist():
        tokens = []
        for token in sample:
            mean = math.fsum(token) / len(token)
            variance = math.fsum((value - mean) ** 2 for value in token) / len(token)
            tokens.append(
                [
                    (value - mean) / math.sqrt(variance + eps) * scale + bias
                    for value, scale, bias in zip(token, scales, biases, strict=True)
                ]
            )
        batches.append(tokens)
    return torch.tensor(batches, dtype=torch.float32)


def scalar_affine(x, weight, bias):
    return torch.tensor(
        [
            [
                [
                    math.sumprod(token, row) + offset
                    for row, offset in zip(weight, bias, strict=True)
                ]
                for token in sample
            ]
            for sample in x.tolist()
        ],
        dtype=torch.float32,
    )


def fixed_linear(weight, bias):
    """A test branch attached only through a public Block submodule attribute."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        layer = nn.Linear(len(weight[0]), len(weight), bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return layer


def test_stage32_public_submodules_match_constructor_configuration(make_block):
    """REQ-3.2.2, REQ-3.2.3, REQ-3.2.6: The Block uses the existing module classes."""
    block = make_block(mlp_hidden_dim=9)

    assert isinstance(block, nn.Module)
    assert isinstance(block.attn, MultiHeadAttention)
    assert isinstance(block.mlp, MLP)
    assert block.attn.q_proj.weight.shape == (4, 4)
    assert block.mlp.fc1.weight.shape == (9, 4)
    assert block.mlp.fc2.weight.shape == (4, 9)
    for norm in (block.ln1, block.ln2):
        assert isinstance(norm, nn.LayerNorm)
        assert norm.normalized_shape == (4,)
        assert norm.weight.requires_grad
        assert norm.bias is not None
        assert norm.bias.requires_grad
    assert block.ln1 is not block.ln2
    assert block.ln1.weight is not block.ln2.weight
    assert block.ln1.bias is not block.ln2.bias


def test_stage32_layer_norms_use_independent_affine_parameters_and_token_statistics(
    make_block, hidden_states
):
    """REQ-3.2.3: Distinct affine settings and a constant token expose wrong axes."""
    block = make_block()
    set_norm_parameters(block)

    for index, norm in enumerate((block.ln1, block.ln2)):
        expected = scalar_layer_norm(
            hidden_states, NORM_SCALES[index], NORM_BIASES[index], norm.eps
        )
        torch.testing.assert_close(norm(hidden_states), expected, rtol=1e-5, atol=1e-6)


def test_stage32_preln_branch_inputs_and_two_residuals_match_scalar_reference(
    make_block, hidden_states
):
    """REQ-3.2.1, REQ-3.2.2, REQ-3.2.6: MLP sees ln2 of the updated residual."""
    block = make_block()
    set_norm_parameters(block)
    attn_weight = ((0.5, 1, 0, 0), (0, -0.5, 0.25, 0),
                   (0, 0, 0.75, 0.5), (-1, 0, 0, 0.25))
    attn_bias = (0.5, -0.25, 1.0, 0.75)
    mlp_weight = ((0, 0.5, -0.25, 0), (1, 0, 0, 0.5),
                  (0.25, 0, -1, 0), (0, -0.5, 0, 0.75))
    mlp_bias = (-0.5, 0.25, 0.75, -1.0)
    block.attn = fixed_linear(attn_weight, attn_bias)
    block.mlp = fixed_linear(mlp_weight, mlp_bias)
    expected_attn_input = scalar_layer_norm(
        hidden_states, NORM_SCALES[0], NORM_BIASES[0], block.ln1.eps
    )
    h1 = hidden_states + scalar_affine(expected_attn_input, attn_weight, attn_bias)
    expected_mlp_input = scalar_layer_norm(
        h1, NORM_SCALES[1], NORM_BIASES[1], block.ln2.eps
    )
    expected = h1 + scalar_affine(expected_mlp_input, mlp_weight, mlp_bias)
    observed_inputs = {}

    def capture_attention(module, args):
        observed_inputs["attn"] = args[0].detach().clone()

    def capture_mlp(module, args):
        observed_inputs["mlp"] = args[0].detach().clone()

    attention_hook = block.attn.register_forward_pre_hook(capture_attention)
    mlp_hook = block.mlp.register_forward_pre_hook(capture_mlp)
    try:
        actual = block(hidden_states)
    finally:
        attention_hook.remove()
        mlp_hook.remove()

    torch.testing.assert_close(
        observed_inputs["attn"], expected_attn_input, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        observed_inputs["mlp"], expected_mlp_input, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("attention_bias", "mlp_bias"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((1, -2, 0.5, 3), (0, 0, 0, 0)),
        ((0, 0, 0, 0), (-1, 0.5, 2, -3)),
    ],
    ids=["both-zero", "attention-only", "mlp-only"],
)
def test_stage32_zero_branches_preserve_unscaled_residuals(
    make_block, hidden_states, attention_bias, mlp_bias
):
    """REQ-3.2.1, REQ-3.2.2: Each branch adds directly to the current residual."""
    block = make_block()
    zero_weight = ((0, 0, 0, 0),) * 4
    block.attn = fixed_linear(zero_weight, attention_bias)
    block.mlp = fixed_linear(zero_weight, mlp_bias)
    expected = hidden_states + torch.tensor(attention_bias) + torch.tensor(mlp_bias)

    torch.testing.assert_close(block(hidden_states), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("options", "causal"),
    [
        pytest.param({}, True, id="default-causal"),
        pytest.param({"causal": True}, True, id="explicit-causal"),
        pytest.param({"causal": False}, False, id="bidirectional"),
    ],
)
def test_stage32_real_attention_configuration_batch_isolation_and_prefix(
    make_block, hidden_states, options, causal
):
    """REQ-3.2.2, REQ-3.2.4, REQ-3.2.6: Uniform attention has a known mean output."""
    block = make_block(**options).eval()
    set_norm_parameters(block)
    with torch.no_grad():
        block.attn.q_proj.weight.zero_()
        block.attn.k_proj.weight.zero_()
        block.attn.v_proj.weight.copy_(torch.eye(4))
        block.attn.out_proj.weight.copy_(torch.eye(4))
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
    changed = hidden_states.clone()
    changed[0, 2] = torch.tensor([-4, 3, 1, -2], dtype=torch.float32)
    for sample in (hidden_states, changed):
        normalized = scalar_layer_norm(
            sample, NORM_SCALES[0], NORM_BIASES[0], block.ln1.eps
        )
        if causal:
            lengths = torch.arange(1, 4, dtype=torch.float32).reshape(1, 3, 1)
            expected_branch = normalized.cumsum(dim=1) / lengths
        else:
            expected_branch = normalized.mean(dim=1, keepdim=True).expand_as(sample)
        expected = sample + expected_branch

        torch.testing.assert_close(block(sample), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "d_model", "num_heads", "mlp_hidden_dim"),
    [(1, 1, 4, 2, 7), (3, 5, 4, 2, 7), (2, 16, 128, 4, 512)],
    ids=["single-token", "multiple-batches", "initial-config"],
)
def test_stage32_normal_initialization_preserves_shape_and_finite_values(
    make_block, batch_size, sequence_length, d_model, num_heads, mlp_hidden_dim
):
    """REQ-3.2.2, REQ-3.2.6: Actual submodules compose over several legal B,T,D."""
    block = make_block(d_model, num_heads, mlp_hidden_dim).eval()
    shape = (batch_size, sequence_length, d_model)
    x = torch.linspace(-1, 1, math.prod(shape), dtype=torch.float32).reshape(shape)

    actual = block(x)

    assert actual.shape == shape
    assert torch.isfinite(actual).all()


def test_stage32_modes_propagate_and_dropout_changes_only_the_mlp_branch(make_block):
    """REQ-3.2.5: Output equals the residual plus dropped or scaled MLP bias."""
    block = make_block(dropout=0.5)
    bias = torch.tensor([0.25, -0.5, 1.0, 2.0], dtype=torch.float32)
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.copy_(bias)
    x = torch.tensor([[[1.0, -1.0, 0.5, 2.0]]]).expand(16, 16, 4)
    block.eval()
    assert all(not module.training for module in block.modules())
    torch.testing.assert_close(block(x), x + bias, rtol=0, atol=1e-6)
    block.train()
    assert all(module.training for module in block.modules())
    retained_masks = []
    for seed in (0, 17, 42):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            actual = block(x)
        retained = actual - x != 0
        expected = x + torch.where(retained, 2 * bias, torch.zeros_like(bias))
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        retained_masks.append(retained)
    masks = torch.stack(retained_masks)
    assert masks.any(dim=(0, 1, 2)).all()
    assert (~masks).any(dim=(0, 1, 2)).all()
    block.eval()
    torch.testing.assert_close(block(x), x + bias, rtol=0, atol=1e-6)


def test_stage32_backward_reaches_input_norm_attention_and_mlp(make_block, hidden_states):
    """REQ-3.2.5, REQ-3.2.6: All component parameters have finite gradients."""
    block = make_block()
    x = hidden_states.requires_grad_()
    coefficients = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)

    (block(x) * coefficients).sum().backward()

    for name in ("ln1", "attn", "ln2", "mlp"):
        parameters = list(getattr(block, name).parameters())
        assert parameters, name
        for parameter in parameters:
            assert parameter.grad is not None, name
            assert parameter.grad.shape == parameter.shape, name
            assert torch.isfinite(parameter.grad).all(), name
        assert any(
            torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        ), name
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad).item() > 0


@pytest.mark.parametrize(
    ("d_model", "num_heads", "mlp_hidden_dim", "dropout"),
    [
        (0, 2, 7, 0.0), (-4, 2, 7, 0.0), (1.5, 2, 7, 0.0),
        (4, 0, 7, 0.0), (4, -2, 7, 0.0), (4, 1.5, 7, 0.0),
        (4, 3, 7, 0.0), (4, 2, 0, 0.0), (4, 2, -7, 0.0),
        (4, 2, 1.5, 0.0), (4, 2, 7, -0.1), (4, 2, 7, 1.0),
    ],
    ids=[
        "zero-model", "negative-model", "fractional-model",
        "zero-heads", "negative-heads", "fractional-heads", "non-divisible",
        "zero-hidden", "negative-hidden", "fractional-hidden",
        "negative-dropout", "dropout-one",
    ],
)
def test_stage32_rejects_invalid_constructor_parameters(
    block_class, d_model, num_heads, mlp_hidden_dim, dropout
):
    """REQ-3.2.6 and section 2.1: Existing constructor constraints remain in force."""
    with pytest.raises(Exception):
        block_class(d_model=d_model, num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim, dropout=dropout)

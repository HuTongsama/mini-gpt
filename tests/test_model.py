import itertools

import pytest
import torch
from torch import nn

from src.embedding import InputEmbedding
from src.transformer import TransformerBlock
from tests.test_mlp import exact_gelu
from tests.test_multihead_attention import FIXED_WEIGHTS as ATTENTION_WEIGHTS
from tests.test_multihead_attention import scalar_reference as attention_reference
from tests.test_transformer_block import scalar_affine, scalar_layer_norm


TOKEN_WEIGHTS = (
    (1, 0, 2, -1), (0, 2, -1, 1), (-2, 1, 0, 3),
    (2, -1, 1, 0), (-1, 3, 2, -2),
)
POSITION_WEIGHTS = ((0, 0.5, -0.5, 1), (1, 0, 0.5, -1),
                    (-0.5, 1, 0, 0.5), (0.5, -1, 1, 0))
MLP_WEIGHT1 = ((0.5, 0, -0.25, 1), (0, -1, 0.5, 0.25), (1, 0.25, 0, -0.5))
MLP_WEIGHT2 = ((1, 0.5, 0), (0, -0.25, 0.5), (-0.5, 0, 1), (0.25, 1, -0.5))
FINAL_SCALE = (0.5, -1.0, 1.5, 0.75)
FINAL_BIAS = (0.25, -0.5, 0.75, 1.0)
HEAD_WEIGHT = ((2, 0, 0, -1), (-1, 1, 0.5, 0), (0, -2, 1, 0.5),
               (0.5, 0, -1, 2), (1, 0.5, 0, -0.5))


@pytest.fixture
def model_class():
    from src.model import MiniGPT

    return MiniGPT


@pytest.fixture
def make_model(model_class):
    def make(**options):
        configuration = dict(
            vocab_size=5, context_size=4, embedding_dim=4,
            num_layers=2, num_heads=2, mlp_hidden_dim=3,
        )
        configuration.update(options)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            return model_class(**configuration)

    return make


@pytest.fixture
def token_ids():
    return torch.tensor([[0, 1, 2], [3, 0, 4]], dtype=torch.long)


def components(model):
    """Locate contracted module types without choosing new public attribute names."""
    modules = list(model.modules())
    embeddings = [module for module in modules if isinstance(module, InputEmbedding)]
    blocks = [module for module in modules if isinstance(module, TransformerBlock)]
    block_children = {id(child) for block in blocks for child in block.modules()}
    final_norms = [
        module for module in modules
        if isinstance(module, nn.LayerNorm) and id(module) not in block_children
    ]
    heads = [
        module for module in modules
        if isinstance(module, nn.Linear) and id(module) not in block_children
    ]
    assert len(embeddings) == 1
    assert len(final_norms) == 1
    assert len(heads) == 1
    return embeddings[0], blocks, final_norms[0], heads[0]


def copy_parameter(parameter, values):
    with torch.no_grad():
        parameter.copy_(torch.as_tensor(values, dtype=torch.float32))


def configure_fixed_pipeline(model):
    embedding, blocks, final_norm, head = components(model)
    copy_parameter(embedding.token_embedding.weight, TOKEN_WEIGHTS)
    copy_parameter(embedding.position_embedding.weight, POSITION_WEIGHTS)
    for index, block in enumerate(blocks):
        copy_parameter(block.ln1.weight, (1, 0.5, -0.75, 1.25))
        copy_parameter(block.ln1.bias, (0.25, -0.5, 0, 0.75))
        copy_parameter(block.ln2.weight, (0.5, 1.25, -1, 0.75))
        copy_parameter(block.ln2.bias, (-0.25, 0, 0.5, 0.25))
        copy_parameter(block.attn.q_proj.weight, ATTENTION_WEIGHTS[0])
        copy_parameter(block.attn.k_proj.weight, ATTENTION_WEIGHTS[1])
        copy_parameter(block.attn.v_proj.weight, torch.eye(4))
        copy_parameter(block.attn.out_proj.weight, (index + 1) * 0.25 * torch.eye(4))
        copy_parameter(block.mlp.fc1.weight, MLP_WEIGHT1)
        copy_parameter(block.mlp.fc1.bias, (0.25, -0.5, 0.75))
        copy_parameter(
            block.mlp.fc2.weight,
            (index + 1) * torch.tensor(MLP_WEIGHT2, dtype=torch.float32),
        )
        copy_parameter(block.mlp.fc2.bias, (0.5, -0.25, 0.75, -0.5))
    copy_parameter(final_norm.weight, FINAL_SCALE)
    copy_parameter(final_norm.bias, FINAL_BIAS)
    copy_parameter(head.weight, HEAD_WEIGHT)
    return embedding, blocks, final_norm, head


def pipeline_reference(ids, blocks, final_norm, causal):
    """Compose independent per-head scalar attention and scalar nonlinear maps."""
    tokens = torch.tensor(TOKEN_WEIGHTS, dtype=torch.float32)[ids]
    positions = torch.tensor(POSITION_WEIGHTS, dtype=torch.float32)[:ids.shape[1]]
    states = [tokens + positions]
    for index, block in enumerate(blocks):
        normalized = scalar_layer_norm(
            states[-1], (1, 0.5, -0.75, 1.25), (0.25, -0.5, 0, 0.75), block.ln1.eps
        )
        attention_weights = (
            ATTENTION_WEIGHTS[0], ATTENTION_WEIGHTS[1],
            torch.eye(4).tolist(), ((index + 1) * 0.25 * torch.eye(4)).tolist(),
        )
        attention = attention_reference(normalized, attention_weights, 2, causal)
        updated = states[-1] + attention
        mlp_input = scalar_layer_norm(
            updated, (0.5, 1.25, -1, 0.75), (-0.25, 0, 0.5, 0.25), block.ln2.eps
        )
        hidden = scalar_affine(mlp_input, MLP_WEIGHT1, (0.25, -0.5, 0.75))
        activated = torch.tensor(
            [[[exact_gelu(value) for value in token] for token in sample]
             for sample in hidden.tolist()],
            dtype=torch.float32,
        )
        second_weight = tuple(
            tuple((index + 1) * value for value in row) for row in MLP_WEIGHT2
        )
        states.append(updated + scalar_affine(
            activated, second_weight, (0.5, -0.25, 0.75, -0.5)
        ))
    normalized = scalar_layer_norm(states[-1], FINAL_SCALE, FINAL_BIAS, final_norm.eps)
    logits = scalar_affine(normalized, HEAD_WEIGHT, (0,) * 5)
    return states, normalized, logits


@pytest.mark.parametrize("num_layers", [1, 3])
def test_stage33_stack_and_output_parameters_are_independent(make_model, num_layers):
    """REQ-3.3.1 through REQ-3.3.4: Real blocks and final layers have distinct storage."""
    model = make_model(num_layers=num_layers)
    assert isinstance(model, nn.Module)
    embedding, blocks, final_norm, head = components(model)
    assert len(blocks) == num_layers
    storage_sets = [
        {parameter.data_ptr() for parameter in block.parameters()} for block in blocks
    ]
    for index, storage in enumerate(storage_sets):
        assert all(storage.isdisjoint(other) for other in storage_sets[index + 1:])
    assert final_norm.normalized_shape == (4,)
    assert final_norm.weight.requires_grad and final_norm.bias.requires_grad
    norm_storage = {final_norm.weight.data_ptr(), final_norm.bias.data_ptr()}
    assert all(norm_storage.isdisjoint(storage) for storage in storage_sets)
    assert head.weight.shape == (5, 4)
    assert head.bias is None
    assert head.weight.requires_grad
    assert head.weight.data_ptr() != embedding.token_embedding.weight.data_ptr()
    for block in blocks:
        assert block.attn.q_proj.weight.shape == (4, 4)
        assert block.mlp.fc1.weight.shape == (3, 4)
        assert block.mlp.fc2.weight.shape == (4, 3)


def test_stage33_fixed_pipeline_embedding_block_order_final_norm_and_raw_logits(
    make_model, token_ids
):
    """REQ-3.3.1 through REQ-3.3.4: Intermediate states and logits match known arithmetic."""
    model = make_model()
    embedding, blocks, final_norm, head = configure_fixed_pipeline(model)
    expected_states, expected_norm, expected_logits = pipeline_reference(
        token_ids, blocks, final_norm, causal=True
    )
    assert expected_logits.min().item() < 0
    assert expected_logits.max().item() > 1
    observed = []

    def record_stage(module, args, output):
        observed.append((module, args[0].detach().clone(), output.detach().clone()))

    hooks = [
        module.register_forward_hook(record_stage)
        for module in (embedding, *blocks, final_norm, head)
    ]
    try:
        actual = model(token_ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert [item[0] for item in observed] == [embedding, *blocks, final_norm, head]
    torch.testing.assert_close(observed[0][2], expected_states[0], rtol=0, atol=1e-6)
    for index in range(len(blocks)):
        torch.testing.assert_close(
            observed[index + 1][1], expected_states[index], rtol=1e-5, atol=3e-6
        )
        torch.testing.assert_close(
            observed[index + 1][2], expected_states[index + 1], rtol=1e-5, atol=3e-6
        )
    torch.testing.assert_close(observed[-2][1], expected_states[-1], rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(observed[-1][1], expected_norm, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(actual, expected_logits, rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize(
    ("options", "causal"),
    [pytest.param({}, True, id="default-causal"),
     pytest.param({"causal": False}, False, id="bidirectional")],
)
def test_stage33_future_information_and_batch_isolation(
    make_model, token_ids, options, causal
):
    """REQ-3.3.2, REQ-3.3.5: Configured real Blocks follow the permitted Key set."""
    model = make_model(**options).eval()
    _, blocks, final_norm, _ = configure_fixed_pipeline(model)
    changed = token_ids.clone()
    changed[0, 2] = 4
    original_expected = pipeline_reference(token_ids, blocks, final_norm, causal)[2]
    changed_expected = pipeline_reference(changed, blocks, final_norm, causal)[2]
    if causal:
        torch.testing.assert_close(
            original_expected[:, :2], changed_expected[:, :2], rtol=0, atol=0
        )
    else:
        assert (original_expected[0, :2] - changed_expected[0, :2]).abs().max() > 1e-3
    torch.testing.assert_close(original_expected[1], changed_expected[1], rtol=0, atol=0)

    with torch.no_grad():
        for ids, expected in ((token_ids, original_expected), (changed, changed_expected)):
            for _ in range(2):
                torch.testing.assert_close(model(ids), expected, rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "context_size", "embedding_dim", "num_heads"),
    [(1, 1, 4, 4, 2), (3, 4, 4, 4, 2), (2, 64, 64, 128, 4)],
    ids=["single-token", "context-boundary", "initial-config"],
)
def test_stage33_shapes_and_finite_logits(
    make_model, batch_size, sequence_length, context_size, embedding_dim, num_heads
):
    """REQ-3.3.1, REQ-3.3.4, REQ-3.3.6: Legal boundary shapes retain the vocabulary axis."""
    model = make_model(
        context_size=context_size, embedding_dim=embedding_dim,
        num_heads=num_heads, mlp_hidden_dim=4 * embedding_dim,
    ).eval()
    ids = torch.arange(batch_size * sequence_length).reshape(batch_size, sequence_length) % 5

    actual = model(ids)

    assert actual.shape == (batch_size, sequence_length, 5)
    assert actual.is_floating_point()
    assert torch.isfinite(actual).all()


def test_stage33_gradients_reach_every_component_without_future_representation_gradient(
    make_model, token_ids
):
    """REQ-3.3.5, REQ-3.3.6: Prefix loss reaches all layers but not future hidden states."""
    model = make_model()
    embedding, blocks, final_norm, head = components(model)
    representations = []

    def retain_representation(module, args, output):
        output.retain_grad()
        representations.append(output)

    hook = embedding.register_forward_hook(retain_representation)
    try:
        logits = model(token_ids)
        coefficients = torch.arange(1, 21, dtype=torch.float32).reshape(2, 2, 5)
        (logits[:, :2] * coefficients).sum().backward()
    finally:
        hook.remove()

    for parameter in model.parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    groups = [embedding, final_norm, head]
    for block in blocks:
        groups.extend((block.ln1, block.attn, block.ln2, block.mlp))
    for group in groups:
        assert any(torch.count_nonzero(p.grad).item() > 0 for p in group.parameters())
    assert len(representations) == 1
    gradient = representations[0].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[:, :2].abs().max().item() > 1e-6
    torch.testing.assert_close(
        gradient[:, 2:], torch.zeros_like(gradient[:, 2:]), rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("active_layer", [0, 1], ids=["first-block", "second-block"])
def test_stage33_train_eval_propagate_and_output_matches_dropout_candidates(
    make_model, active_layer
):
    """REQ-3.3.6: Known MLP bias dropout yields a finite set of independently known logits."""
    model = make_model(dropout=0.5)
    embedding, blocks, final_norm, head = components(model)
    base = torch.tensor([1.0, 0.0, 2.0, -1.0])
    bias = torch.tensor([0.25, -0.5, 1.0, 2.0])
    copy_parameter(embedding.token_embedding.weight, base.repeat(5, 1))
    copy_parameter(embedding.position_embedding.weight, torch.zeros(4, 4))
    for block in blocks:
        copy_parameter(block.attn.out_proj.weight, torch.zeros(4, 4))
        copy_parameter(block.mlp.fc2.weight, torch.zeros(4, 3))
        copy_parameter(block.mlp.fc2.bias, torch.zeros(4))
    copy_parameter(blocks[active_layer].mlp.fc2.bias, bias)
    copy_parameter(final_norm.weight, FINAL_SCALE)
    copy_parameter(final_norm.bias, FINAL_BIAS)
    copy_parameter(head.weight, HEAD_WEIGHT)
    ids = torch.zeros((16, 4), dtype=torch.long)
    expected_eval = scalar_affine(
        scalar_layer_norm(
            (base + bias).reshape(1, 1, 4), FINAL_SCALE, FINAL_BIAS, final_norm.eps
        ), HEAD_WEIGHT, (0,) * 5,
    ).expand(16, 4, 5)
    masks = torch.tensor(list(itertools.product((0, 1), repeat=4)), dtype=torch.float32)
    candidates = scalar_affine(
        scalar_layer_norm(
            (base + 2 * bias * masks).unsqueeze(0), FINAL_SCALE, FINAL_BIAS, final_norm.eps
        ), HEAD_WEIGHT, (0,) * 5,
    )[0]
    model.eval()
    assert all(not module.training for module in model.modules())
    torch.testing.assert_close(model(ids), expected_eval, rtol=1e-5, atol=3e-6)
    model.train()
    assert all(module.training for module in model.modules())
    selections = []
    for seed in (0, 17, 42):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            actual = model(ids).reshape(-1, 5)
        distances = (actual[:, None, :] - candidates[None, :, :]).abs().amax(dim=-1)
        assert (distances.min(dim=1).values < 1e-5).all()
        selections.append(distances.argmin(dim=1))
    assert torch.cat(selections).unique().numel() > 1
    model.eval()
    torch.testing.assert_close(model(ids), expected_eval, rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param(name, value, id=f"{name}-{label}")
        for name in (
            "vocab_size", "context_size", "embedding_dim",
            "num_layers", "num_heads", "mlp_hidden_dim",
        )
        for label, value in (("zero", 0), ("negative", -1), ("fractional", 1.5))
    ] + [
        pytest.param("num_heads", 3, id="non-divisible"),
        pytest.param("dropout", -0.1, id="negative-dropout"),
        pytest.param("dropout", 1.0, id="dropout-one"),
        pytest.param("dropout", float("nan"), id="nan-dropout"),
        pytest.param("dropout", float("inf"), id="infinite-dropout"),
        pytest.param("causal", "yes", id="nonboolean-causal"),
    ],
)
def test_stage33_rejects_invalid_constructor_parameters(make_model, parameter, value):
    """REQ-3.3.6: Reject invalid dimensions, head divisibility, probability and mode."""
    with pytest.raises(Exception):
        make_model(**{parameter: value})


@pytest.mark.parametrize("invalid_id", [-1, 5], ids=["negative-id", "id-at-vocab-size"])
def test_stage33_rejects_out_of_vocabulary_ids(make_model, invalid_id):
    """REQ-3.3.6: Invalid IDs cannot be silently clamped or wrapped."""
    model = make_model()
    ids = torch.tensor([[0, invalid_id]], dtype=torch.long)

    with pytest.raises(Exception):
        model(ids)


def test_stage33_rejects_sequence_beyond_context(make_model):
    """REQ-3.3.6: Context overflow cannot be silently truncated."""
    model = make_model()

    with pytest.raises(Exception):
        model(torch.zeros((1, 5), dtype=torch.long))

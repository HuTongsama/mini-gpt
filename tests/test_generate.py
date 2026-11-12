from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.model import MiniGPT


@pytest.fixture
def generate():
    from src.generate import generate_greedy

    return generate_greedy


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("logits", torch.tensor([
            [[0, 1, 2, 3, 9], [-3, 0, 9, 1, -1]],
            [[0, 1, 2, 9, 3], [0, 8, 3, 8, -2]],
        ], dtype=torch.float32))

    def forward(self, inputs):
        return self.logits


class RecordingCycle(nn.Module):
    """Each position prefers the next ID in the cycle 0, 1, 2, 3, 4, 0."""

    def __init__(self, fail_on=None):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("offset", torch.tensor(-1.0))
        self.dropout = nn.Dropout(p=0.5)
        self.fail_on = fail_on
        self.windows = []
        self.modes = []

    def forward(self, inputs):
        self.windows.append(inputs.detach().clone())
        self.modes.append((self.training, self.dropout.training, torch.is_grad_enabled()))
        if len(self.windows) == self.fail_on:
            raise RuntimeError("controlled forward failure")
        preferred = (inputs + 1).remainder(5)
        return self.dropout(F.one_hot(preferred, num_classes=5) * self.scale + self.offset)


@pytest.fixture
def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        return MiniGPT(
            vocab_size=5, context_size=3, embedding_dim=8, num_layers=1,
            num_heads=2, mlp_hidden_dim=16, dropout=0.25, causal=True,
        )


def assert_output(result, prompt, original, expected):
    assert isinstance(result, torch.Tensor)
    assert result.dtype == prompt.dtype == torch.long
    assert result.device == prompt.device
    assert result.shape == (len(expected), len(expected[0]))
    assert result.tolist() == expected
    assert torch.equal(prompt, original)
    assert result.untyped_storage().data_ptr() != prompt.untyped_storage().data_ptr()


@pytest.mark.parametrize("training", [True, False], ids=["from-train", "from-eval"])
def test_stage51_zero_tokens_returns_independent_prompt_without_forward_or_rng(
    generate, training,
):
    """REQ-5.1.1, REQ-5.1.5: Zero generation is a state-preserving copy."""
    model = RecordingCycle().train(training)
    model.scale.grad = torch.tensor(3.0)
    original_train = model.train
    mode_changes = []

    def track_mode(mode=True):
        if mode != model.training:
            mode_changes.append(mode)
        return original_train(mode)

    model.train = track_mode
    prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    original = prompt.clone()
    with torch.random.fork_rng(devices=[]):
        rng = torch.get_rng_state()
        result = generate(model, prompt, max_new_tokens=0, context_size=1)
        assert torch.equal(torch.get_rng_state(), rng)

    assert_output(result, prompt, original, [[1, 2], [3, 4]])
    assert model.windows == []
    assert mode_changes == []
    assert model.training is training
    assert torch.equal(model.scale, torch.tensor(2.0))
    assert torch.equal(model.offset, torch.tensor(-1.0))
    assert torch.equal(model.scale.grad, torch.tensor(3.0))


def test_stage51_zero_tokens_preserves_buffer_updated_by_train(generate):
    """REQ-5.1.5: Even setting the same mode can change persistent state."""
    class ModeCountingModel(RecordingCycle):
        def __init__(self):
            super().__init__()
            self.register_buffer("mode_calls", torch.zeros((), dtype=torch.long))

        def train(self, mode=True):
            self.mode_calls.add_(1)
            return super().train(mode)

    model = ModeCountingModel()
    before = model.mode_calls.clone()
    prompt = torch.tensor([[1]], dtype=torch.long)

    result = generate(model, prompt, max_new_tokens=0, context_size=1)

    assert result.tolist() == [[1]]
    assert model.windows == []
    assert torch.equal(model.mode_calls, before)


@pytest.mark.parametrize(
    "prompt",
    [
        [[0, 1]],
        torch.tensor([0, 1]),
        torch.zeros((1, 1, 1), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
    ],
    ids=["non-tensor", "rank-one", "rank-three", "float", "int32",
         "empty-batch", "empty-prompt"],
)
def test_stage51_rejects_invalid_prompt(generate, prompt):
    """REQ-5.1.1: Shape and dtype validation also applies to the zero-step path."""
    with pytest.raises(Exception):
        generate(RecordingCycle(), prompt, max_new_tokens=0, context_size=2)


@pytest.mark.parametrize(
    "name,value",
    [
        pytest.param("max_new_tokens", -1, id="steps-negative"),
        pytest.param("max_new_tokens", True, id="steps-true"),
        pytest.param("max_new_tokens", False, id="steps-false"),
        pytest.param("max_new_tokens", 1.0, id="steps-float"),
        pytest.param("max_new_tokens", "1", id="steps-string"),
        pytest.param("context_size", 0, id="context-zero"),
        pytest.param("context_size", -1, id="context-negative"),
        pytest.param("context_size", True, id="context-true"),
        pytest.param("context_size", False, id="context-false"),
        pytest.param("context_size", 1.0, id="context-float"),
        pytest.param("context_size", "1", id="context-string"),
    ],
)
def test_stage51_rejects_invalid_generation_options(generate, name, value):
    """REQ-5.1.1: Lengths must be Python integers with the specified bounds."""
    options = {"max_new_tokens": 1, "context_size": 2}
    options[name] = value
    with pytest.raises(Exception):
        generate(RecordingCycle(), torch.tensor([[1]]), **options)


def test_stage51_selects_last_position_vocab_argmax_and_lowest_tie(generate):
    """REQ-5.1.2: The first row, last input ID and last-row argmax disagree."""
    prompt = torch.tensor([[0, 1], [2, 4]], dtype=torch.long)
    original = prompt.clone()
    result = generate(FixedLogits(), prompt, max_new_tokens=1, context_size=2)

    # Batch 0 prefers 2 on the last row. Batch 1 ties IDs 1 and 3.
    assert_output(result, prompt, original, [[0, 1, 2], [2, 4, 1]])


@pytest.mark.parametrize(
    "prompt_values,capacity,windows,expected",
    [
        (
            [[0, 1], [2, 3]], 3,
            [[[0, 1], [2, 3]], [[0, 1, 2], [2, 3, 4]], [[1, 2, 3], [3, 4, 0]]],
            [[0, 1, 2, 3, 4], [2, 3, 4, 0, 1]],
        ),
        ([[1, 3]], 2, [[[1, 3]], [[3, 4]], [[4, 0]]], [[1, 3, 4, 0, 1]]),
        ([[4, 0, 1, 2]], 2, [[[1, 2]], [[2, 3]], [[3, 4]]], [[4, 0, 1, 2, 3, 4, 0]]),
        ([[2]], 1, [[[2]], [[3]], [[4]]], [[2, 3, 4, 0]]),
    ],
    ids=["crosses-capacity-batched", "at-capacity", "over-capacity", "capacity-one"],
)
def test_stage51_feeds_generated_tokens_and_uses_exact_rolling_windows(
    generate, prompt_values, capacity, windows, expected,
):
    """REQ-5.1.1, REQ-5.1.3, REQ-5.1.4: Compare every window and the entire output."""
    model = RecordingCycle()
    prompt = torch.tensor(prompt_values, dtype=torch.long)
    original = prompt.clone()
    result = generate(model, prompt, max_new_tokens=3, context_size=capacity)

    assert [window.tolist() for window in model.windows] == windows
    assert_output(result, prompt, original, expected)


@pytest.mark.parametrize("training", [True, False], ids=["from-train", "from-eval"])
def test_stage51_inference_preserves_state_and_is_deterministic(generate, training):
    """REQ-5.1.5: Both calls run in eval/no-grad without advancing CPU RNG."""
    model = RecordingCycle().train(training)
    model.scale.grad = torch.tensor(7.0)
    backward = Mock()
    model.scale.register_hook(backward)
    placement = Mock(wraps=model.to)
    model.to = placement
    prompt = torch.tensor([[2]], dtype=torch.long)
    original = prompt.clone()

    with torch.random.fork_rng(devices=[]), torch.enable_grad():
        for _ in range(2):
            rng = torch.get_rng_state()
            result = generate(model, prompt, max_new_tokens=2, context_size=2)
            assert torch.equal(torch.get_rng_state(), rng)
            assert model.training is training
            assert_output(result, prompt, original, [[2, 3, 4]])

    assert model.modes == [(False, False, False)] * 4
    assert torch.equal(model.scale, torch.tensor(2.0))
    assert torch.equal(model.offset, torch.tensor(-1.0))
    assert torch.equal(model.scale.grad, torch.tensor(7.0))
    backward.assert_not_called()
    placement.assert_not_called()


@pytest.mark.parametrize("training", [True, False], ids=["from-train", "from-eval"])
def test_stage51_forward_failure_restores_mode_without_returning_partial_result(
    generate, training,
):
    """REQ-5.1.5, REQ-5.1.6: A second-round error still restores caller state."""
    model = RecordingCycle(fail_on=2).train(training)
    model.scale.grad = torch.tensor(7.0)
    prompt = torch.tensor([[1]], dtype=torch.long)
    original = prompt.clone()
    with torch.random.fork_rng(devices=[]):
        rng = torch.get_rng_state()
        with pytest.raises(Exception):
            generate(model, prompt, max_new_tokens=3, context_size=2)
        assert torch.equal(torch.get_rng_state(), rng)

    assert [window.tolist() for window in model.windows] == [[[1]], [[1, 2]]]
    assert model.modes == [(False, False, False)] * 2
    assert model.training is training
    assert torch.equal(prompt, original)
    assert torch.equal(model.scale, torch.tensor(2.0))
    assert torch.equal(model.offset, torch.tensor(-1.0))
    assert torch.equal(model.scale.grad, torch.tensor(7.0))


@pytest.mark.parametrize("token", [-1, 5], ids=["negative", "at-vocab-size"])
def test_stage51_invalid_id_in_consumed_window_raises(generate, tiny_model, token):
    """REQ-5.1.1, REQ-5.1.6: An illegal ID in the retained suffix reaches model validation."""
    prompt = torch.tensor([[0, 1, 2, token]], dtype=torch.long)
    original = prompt.clone()
    tiny_model.train()

    with pytest.raises(Exception):
        generate(tiny_model, prompt, max_new_tokens=1, context_size=3)

    assert tiny_model.training
    assert torch.equal(prompt, original)


def test_stage51_device_mismatch_fails_without_moving_inputs_or_model(generate):
    """REQ-5.1.6: A meta buffer supplies a device mismatch without accelerator computation."""
    class IncompatibleDevice(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchor", torch.empty(1, device="meta"))
            self.input_devices = []

        def forward(self, inputs):
            self.input_devices.append(inputs.device)
            if inputs.device != self.anchor.device:
                raise RuntimeError("controlled device mismatch")
            return torch.zeros((*inputs.shape, 5), device=inputs.device)

    model = IncompatibleDevice().train()
    placement = Mock(wraps=model.to)
    model.to = placement
    prompt = torch.tensor([[1]], dtype=torch.long)

    with pytest.raises(Exception):
        generate(model, prompt, max_new_tokens=1, context_size=2)

    assert all(device == prompt.device for device in model.input_devices)
    assert prompt.device.type == "cpu"
    assert model.anchor.device.type == "meta"
    assert model.training
    placement.assert_not_called()


@pytest.mark.parametrize(
    "shape", [(2, 5), (1, 2, 5), (2, 1, 5)],
    ids=["missing-time-axis", "wrong-batch", "wrong-time-length"],
)
def test_stage51_rejects_incompatible_forward_output(generate, shape):
    """REQ-5.1.6: An incompatible forward result cannot stand in for [B,T,V]."""
    model = FixedLogits()
    model.logits = torch.zeros(shape, dtype=torch.float32)
    prompt = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    with pytest.raises(Exception):
        generate(model, prompt, max_new_tokens=1, context_size=2)


def test_stage51_real_minigpt_with_long_prompt_produces_legal_deterministic_ids(
    generate, tiny_model,
):
    """REQ-5.1.6: Use the unchanged MiniGPT forward and explicit context capacity."""
    prompt = torch.tensor([[0, 1, 2, 3], [4, 2, 0, 1]], dtype=torch.long)
    original = prompt.clone()
    tiny_model.train()
    first = generate(tiny_model, prompt, max_new_tokens=3, context_size=3)
    second = generate(tiny_model, prompt, max_new_tokens=3, context_size=3)

    assert first.shape == second.shape == (2, 7)
    assert first.dtype == second.dtype == prompt.dtype
    assert first.device == second.device == prompt.device
    assert torch.equal(first[:, :4], original)
    assert torch.equal(prompt, original)
    assert torch.equal(first, second)
    assert torch.all((0 <= first[:, 4:]) & (first[:, 4:] < 5))
    assert tiny_model.training

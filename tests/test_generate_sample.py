import math
from copy import deepcopy
from fractions import Fraction
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from src.generate import generate_greedy
from src.model import MiniGPT


@pytest.fixture
def generate():
    from src.generate import generate_sample

    return generate_sample


class LogitModel(nn.Module):
    def __init__(self, rows=None, *, fail_on=None, device="cpu"):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, device=device))
        self.register_buffer("offset", torch.zeros(1, device=device))
        self.rows = None if rows is None else torch.tensor(rows, dtype=torch.float32)
        self.dropout = nn.Dropout(0.25)
        self.fail_on = fail_on
        self.windows = []
        self.modes = []

    def forward(self, ids):
        self.windows.append(ids.detach().clone())
        self.modes.append((self.training, self.dropout.training, torch.is_grad_enabled()))
        if ids.device != self.scale.device:
            raise RuntimeError("controlled device mismatch")
        if len(self.windows) == self.fail_on:
            raise RuntimeError("controlled forward failure")
        if self.rows is not None:
            return self.rows
        first = nn.functional.one_hot((ids + 1) % 5, num_classes=5)
        second = nn.functional.one_hot((ids + 2) % 5, num_classes=5)
        return self.dropout((first * 3 + second * 2) * self.scale + self.offset)


@pytest.fixture
def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(52)
        return MiniGPT(5, 3, 8, 1, 2, 16, dropout=0.25, causal=True)


def call(generate, model, prompt, **options):
    settings = {"max_new_tokens": 1, "context_size": 2}
    settings.update(options)
    return generate(model, prompt, **settings)


def assert_sequence(result, prompt, original, expected):
    assert isinstance(result, torch.Tensor)
    assert result.dtype == prompt.dtype == torch.long
    assert result.device == prompt.device
    assert result.tolist() == expected
    assert torch.equal(prompt, original)
    assert result.untyped_storage().data_ptr() != prompt.untyped_storage().data_ptr()


def snapshot(model):
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    return deepcopy(model.state_dict()), gradients


def assert_unchanged(model, before):
    state, gradients = before
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    for param, expected in zip(model.parameters(), gradients, strict=True):
        if expected is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, expected, rtol=0, atol=0)


class SamplingProbe:
    """Observe multinomial weights and select a positive column of known rank."""

    def __init__(self):
        self.rank = 0
        self.selected_weights = []
        self.support_sizes = []
        self.error = None
        self.allow_backend_error = False
        self.real_sample = torch.multinomial

    def sample(self, input, num_samples, replacement=False, *, generator=None, out=None):
        if self.error is not None:
            raise self.error
        assert num_samples == 1
        assert input.ndim in (1, 2)
        rows = input.detach().reshape(-1, input.shape[-1])
        usable = torch.isfinite(rows).all() and (rows >= 0).all() and (rows.sum(dim=-1) > 0).all()
        if not usable and self.allow_backend_error:
            return self.real_sample(input, num_samples, replacement, generator=generator, out=out)
        assert usable
        torch.testing.assert_close(
            rows.sum(dim=-1), torch.ones(rows.shape[0], dtype=rows.dtype, device=rows.device)
        )
        order = rows.argsort(dim=-1, descending=True, stable=True)
        support = (rows > 0).sum(dim=-1)
        assert (support > self.rank).all()
        indices = order[:, self.rank:self.rank + 1]
        self.selected_weights.extend(rows.gather(1, indices).flatten().tolist())
        self.support_sizes.extend(support.tolist())
        result = indices.reshape(*input.shape[:-1], 1)
        return result if out is None else out.resize_as_(result).copy_(result)


@pytest.fixture
def sampler(monkeypatch, generate):
    probe = SamplingProbe()
    original = torch.multinomial
    # Cover function, method and imported aliases of the public sampling API.
    import src.generate as module

    for name, value in tuple(vars(module).items()):
        if value is original:
            monkeypatch.setattr(module, name, probe.sample)
    monkeypatch.setattr(torch, "multinomial", probe.sample)
    monkeypatch.setattr(torch.Tensor, "multinomial", lambda tensor, *a, **kw: probe.sample(tensor, *a, **kw))
    return probe


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_stage52_zero_steps_does_not_touch_model_or_rng(generate, training):
    """REQ-5.2.1, REQ-5.2.5: Static validation followed by an independent copy."""
    model = LogitModel().train(training)
    model.scale.grad = torch.full_like(model.scale, 4.0)
    model.train = Mock(wraps=model.train)
    before = snapshot(model)
    prompt = torch.tensor([[1, 2], [3, 0]])
    original = prompt.clone()
    rng = torch.Generator().manual_seed(52)
    generator_state = rng.get_state()
    with torch.random.fork_rng(devices=[]):
        global_state = torch.get_rng_state()
        result = call(generate, model, prompt, max_new_tokens=0, context_size=1,
                      top_k=100, generator=rng)
        assert torch.equal(torch.get_rng_state(), global_state)

    assert_sequence(result, prompt, original, [[1, 2], [3, 0]])
    assert model.windows == []
    model.train.assert_not_called()
    assert model.training is training
    assert torch.equal(rng.get_state(), generator_state)
    assert_unchanged(model, before)


@pytest.mark.parametrize(
    "prompt", [[[0]], torch.tensor([0]), torch.zeros((1, 1, 1), dtype=torch.long),
               torch.zeros((1, 1)), torch.zeros((1, 1), dtype=torch.int32),
               torch.empty((0, 1), dtype=torch.long), torch.empty((1, 0), dtype=torch.long)],
    ids=["list", "rank-one", "rank-three", "float", "int32", "empty-batch", "empty-prompt"],
)
def test_stage52_rejects_invalid_prompt_even_at_zero_steps(generate, prompt):
    """REQ-5.2.1: Reject invalid prompt structure before using the model."""
    with pytest.raises(Exception):
        call(generate, LogitModel(), prompt, max_new_tokens=0)


@pytest.mark.parametrize(
    "name,value",
    [
        pytest.param(name, value, id=f"{name}-{label}")
        for name, cases in [
            ("max_new_tokens", [("negative", -1), ("bool", False), ("float", 1.0), ("string", "1")]),
            ("context_size", [("zero", 0), ("negative", -1), ("bool", True), ("float", 2.0), ("string", "2")]),
            ("temperature", [("bool", True), ("zero", 0), ("negative", -1), ("nan", float("nan")),
                             ("inf", float("inf")), ("string", "1"), ("complex", 1j)]),
            ("top_k", [("zero", 0), ("negative", -1), ("bool", True), ("float", 2.0), ("string", "2")]),
            ("generator", [("integer", 52), ("string", "cpu")]),
        ]
        for label, value in cases
    ],
)
def test_stage52_rejects_invalid_static_options(generate, name, value):
    """REQ-5.2.1: Validate static options even on the zero-step path."""
    options = {"max_new_tokens": 0}
    options[name] = value
    with pytest.raises(Exception):
        call(generate, LogitModel(), torch.tensor([[0]]), **options)


def test_stage52_rejects_top_k_above_vocab(generate):
    """REQ-5.2.1: Infer vocabulary width from actual logits."""
    with pytest.raises(Exception):
        call(generate, LogitModel([[[0, 1, 2]]]), torch.tensor([[0]]), top_k=4)


@pytest.mark.parametrize("temperature", [1, 0.5, Fraction(3, 2)], ids=["int", "float", "fraction"])
def test_stage52_real_sampling_accepts_valid_temperature_types(generate, temperature):
    """REQ-5.2.1: Exercise the real sampler, including Fraction conversion."""
    model = LogitModel([[[0, math.log(2), math.log(4)]]])
    rng = torch.Generator().manual_seed(19)
    before = rng.get_state()
    output = call(generate, model, torch.tensor([[0]]), temperature=temperature, generator=rng)
    assert output.shape == (1, 2)
    assert output[0, 1].item() in (0, 1, 2)
    assert not torch.equal(rng.get_state(), before)


@pytest.mark.parametrize(
    "temperature,top_k,weights",
    [
        (1.0, None, (1, 2, 4)),
        (2.0, None, (1, math.sqrt(2), 2)),
        (0.5, None, (1, 4, 16)),
        (1.0, 2, (0, 2, 4)),
        (1.0, 3, (1, 2, 4)),
        (2.0, 1, (0, 0, 1)),
    ],
    ids=["default", "warm", "cold", "top-two", "top-v", "top-one"],
)
def test_stage52_last_position_temperature_and_topk_probabilities(
    generate, sampler, temperature, top_k, weights,
):
    """REQ-5.2.2, REQ-5.2.3: Reconstruct vocabulary probabilities from sampler choices."""
    z = [0, math.log(2), math.log(4)]
    model = LogitModel([[[9, -3, -2], z], [[-3, -2, 9], z[::-1]]])
    logits_before = model.rows.clone()
    prompt = torch.tensor([[0, 1], [2, 0]])
    original = prompt.clone()
    expected = torch.tensor([weights, weights[::-1]], dtype=torch.float32)
    expected /= expected.sum(dim=-1, keepdim=True)
    support = sum(weight > 0 for weight in weights)
    observed = torch.zeros_like(expected)
    for rank in range(support):
        sampler.rank = rank
        sampler.selected_weights.clear()
        sampler.support_sizes.clear()
        result = call(generate, model, prompt, temperature=temperature, top_k=top_k)
        expected_ids = expected.argsort(dim=-1, descending=True)[:, rank].tolist()
        assert_sequence(result, prompt, original, [
            [*row, token] for row, token in zip(original.tolist(), expected_ids, strict=True)
        ])
        if top_k == 1 and not sampler.selected_weights:
            # Direct selection from a singleton support has probability one.
            assert torch.equal(result, generate_greedy(
                model, prompt, max_new_tokens=1, context_size=2
            ))
            observed[range(2), expected_ids] = 1
        else:
            assert sampler.support_sizes == [support, support]
            observed[range(2), expected_ids] = torch.tensor(sampler.selected_weights)

    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(model.rows, logits_before)


@pytest.mark.parametrize(
    "row,k,allowed_ids,expected_weight",
    [
        ([3, 2, 2, 0], 2, {1, 2}, 1 / (1 + math.e)),
        ([1, 1, 1], 1, {0, 1, 2}, 1.0),
    ],
    ids=["top-two-cutoff-tie", "top-one-all-tied"],
)
def test_stage52_topk_cutoff_ties_keep_exactly_k_candidates(
    generate, sampler, row, k, allowed_ids, expected_weight,
):
    """REQ-5.2.3: The cutoff may retain any tied ID, subject to the candidate limit."""
    sampler.rank = k - 1
    prompt = torch.tensor([[0]])

    result = call(generate, LogitModel([[row]]), prompt, top_k=k)

    assert result.shape == (1, 2)
    assert result[0, 0].item() == 0
    assert result[0, 1].item() in allowed_ids
    if k == 1 and not sampler.selected_weights:
        # Direct selection from a singleton support is also valid.
        return
    assert sampler.support_sizes == [k]
    assert sampler.selected_weights == pytest.approx([expected_weight], rel=1e-6)


def test_stage52_large_finite_logit_offset_is_stable(generate, sampler):
    """REQ-5.2.2: A common offset leaves ratios equal to 1:e:e²."""
    model = LogitModel([[[100, 101, 102]]])
    sampler.rank = 1
    result = call(generate, model, torch.tensor([[0]]))
    assert result.tolist() == [[0, 1]]
    assert sampler.support_sizes == [3]
    assert sampler.selected_weights == pytest.approx([math.e / (1 + math.e + math.e**2)])


@pytest.mark.parametrize(
    "prompt,capacity,windows,expected",
    [
        ([[0, 1], [2, 3]], 3,
         [[[0, 1], [2, 3]], [[0, 1, 3], [2, 3, 0]], [[1, 3, 0], [3, 0, 2]]],
         [[0, 1, 3, 0, 2], [2, 3, 0, 2, 4]]),
        ([[1, 3]], 2, [[[1, 3]], [[3, 0]], [[0, 2]]], [[1, 3, 0, 2, 4]]),
        ([[4, 0, 1, 2]], 2, [[[1, 2]], [[2, 4]], [[4, 1]]], [[4, 0, 1, 2, 4, 1, 3]]),
        ([[2]], 1, [[[2]], [[4]], [[1]]], [[2, 4, 1, 3]]),
    ],
    ids=["crosses-capacity-batched", "at-capacity", "over-capacity", "capacity-one"],
)
def test_stage52_sampled_nongreedy_ids_feed_each_exact_window(
    generate, sampler, prompt, capacity, windows, expected,
):
    """REQ-5.2.4: Force the second highest candidate and check every feedback window."""
    model = LogitModel()
    sampler.rank = 1
    ids = torch.tensor(prompt)
    original = ids.clone()
    result = call(generate, model, ids, max_new_tokens=3, context_size=capacity)
    assert [window.tolist() for window in model.windows] == windows
    assert sampler.support_sizes == [5] * (3 * len(prompt))
    assert_sequence(result, ids, original, expected)


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_stage52_explicit_generators_reproduce_without_using_global_rng(generate, training):
    """REQ-5.2.5: Real draws use independent same-seed generators and preserve model state."""
    model = LogitModel().train(training)
    model.scale.grad = torch.full_like(model.scale, 7)
    model.to = Mock(wraps=model.to)
    backward = Mock()
    model.scale.register_hook(backward)
    before = snapshot(model)
    prompt = torch.tensor([[0, 1], [3, 4]])
    original = prompt.clone()
    first_rng, second_rng = (torch.Generator().manual_seed(91) for _ in range(2))
    first_state = first_rng.get_state()
    with torch.random.fork_rng(devices=[]), torch.enable_grad():
        global_state = torch.get_rng_state()
        first = call(generate, model, prompt, max_new_tokens=4, generator=first_rng)
        assert torch.equal(torch.get_rng_state(), global_state)
        assert model.training is training
        assert torch.equal(second_rng.get_state(), first_state)
        second = call(generate, model, prompt, max_new_tokens=4, generator=second_rng)
        assert torch.equal(torch.get_rng_state(), global_state)
    assert not torch.equal(first_rng.get_state(), first_state)
    assert not torch.equal(second_rng.get_state(), first_state)
    assert torch.equal(first, second)
    assert first.shape == (2, 6)
    assert torch.equal(first[:, :2], original)
    assert torch.equal(prompt, original)
    assert model.modes == [(False, False, False)] * 8
    assert model.training is training
    assert_unchanged(model, before)
    model.to.assert_not_called()
    backward.assert_not_called()


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("failure", ["forward", "sampler"])
def test_stage52_restores_mode_on_forward_or_sampling_exception(
    generate, sampler, training, failure,
):
    """REQ-5.2.5, REQ-5.2.6: Errors never yield a partial sequence."""
    model = LogitModel(fail_on=2 if failure == "forward" else None).train(training)
    model.scale.grad = torch.full_like(model.scale, 7)
    before = snapshot(model)
    if failure == "sampler":
        sampler.error = RuntimeError("controlled sampler failure")
    ids = torch.tensor([[1]])
    original = ids.clone()

    with pytest.raises(Exception):
        call(generate, model, ids, max_new_tokens=3)

    windows = [[[1]], [[1, 2]]] if failure == "forward" else [[[1]]]
    assert [window.tolist() for window in model.windows] == windows
    assert model.modes == [(False, False, False)] * len(windows)
    assert model.training is training
    assert torch.equal(ids, original)
    assert_unchanged(model, before)


@pytest.mark.parametrize("token", [-1, 5])
def test_stage52_rejects_illegal_id_in_consumed_window(generate, tiny_model, token):
    """REQ-5.2.6: Model vocabulary validation applies to the retained window."""
    with pytest.raises(Exception):
        call(generate, tiny_model, torch.tensor([[0, 1, 2, token]]), context_size=3)
    assert tiny_model.training


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3), (2, 1, 3), (2, 2, 0)])
def test_stage52_rejects_incompatible_logits_shape(generate, shape):
    """REQ-5.2.6: The model result must match [B,T,V]."""
    model = LogitModel()
    model.rows = torch.zeros(shape)
    with pytest.raises(Exception):
        call(generate, model, torch.tensor([[0, 1], [1, 2]]))
    assert model.training


def test_stage52_device_mismatch_does_not_move_inputs_or_model(generate):
    """REQ-5.2.6: The meta device serves only as a device marker."""
    model = LogitModel(device="meta")
    model.to = Mock(wraps=model.to)
    prompt = torch.tensor([[1]])
    with pytest.raises(Exception):
        call(generate, model, prompt)
    assert model.scale.device.type == "meta"
    assert all(window.device == prompt.device for window in model.windows)
    model.to.assert_not_called()
    assert model.training


@pytest.mark.parametrize(
    "row", [[0, float("nan"), 1], [0, float("inf"), 1], [-float("inf")] * 3],
    ids=["nan", "positive-inf", "all-negative-inf"],
)
def test_stage52_rejects_unusable_nonfinite_logits(generate, row):
    """REQ-5.2.6: Unusable logits cannot be replaced by a fabricated prediction."""
    model = LogitModel([[row]])
    with pytest.raises(Exception):
        call(generate, model, torch.tensor([[0]]))
    assert model.training


def test_stage52_extreme_finite_scaling_fails_explicitly_or_samples_valid_distribution(
    generate, sampler,
):
    """REQ-5.2.6: Permit stable calculation at extreme scales but reject silent Greedy fallback."""
    large = torch.finfo(torch.float32).max / 4
    model = LogitModel([[[large, 0, large]]])
    # The exact limiting distribution is [1/2, 0, 1/2].
    sampler.rank = 1
    sampler.allow_backend_error = True
    try:
        result = call(generate, model, torch.tensor([[1]]), temperature=1e-50)
    except (ValueError, RuntimeError, OverflowError, FloatingPointError, ZeroDivisionError):
        pass
    else:
        assert result.tolist() == [[1, 2]]
        assert sampler.support_sizes == [2]
        assert sampler.selected_weights == pytest.approx([0.5])
    assert model.training


def test_stage52_real_minigpt_accepts_default_rng_and_preserves_weights(generate, tiny_model):
    """REQ-5.2.5, REQ-5.2.6: Real CPU sampling with a long batched prompt."""
    parameter = next(tiny_model.parameters())
    parameter.grad = torch.ones_like(parameter)
    before = snapshot(tiny_model)
    prompt = torch.tensor([[0, 1, 2, 3], [4, 2, 0, 1]])
    original = prompt.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(520)
        result = call(generate, tiny_model, prompt, max_new_tokens=3, context_size=3,
                      temperature=0.75, top_k=3, generator=None)
    assert result.shape == (2, 7)
    assert result.dtype == prompt.dtype and result.device == prompt.device
    assert torch.equal(result[:, :4], original) and torch.equal(prompt, original)
    assert (result[:, 4:] >= 0).all() and (result[:, 4:] < 5).all()
    assert tiny_model.training
    assert_unchanged(tiny_model, before)

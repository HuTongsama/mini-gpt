import importlib
import math
from fractions import Fraction

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

import src.loss as loss_module
from src.dataset import NextTokenDataset
from src.model import MiniGPT


@pytest.fixture
def train_module():
    return importlib.import_module("src.train")


class ScalarLogits(nn.Module):
    """A single learned logit gives analytically checkable losses and SGD updates."""

    def __init__(self, initial=0.0, require_cleared=False):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.require_cleared = require_cleared
        self.events = []
        self.forward_states = []

    def forward(self, inputs):
        if self.require_cleared:
            gradient = self.value.grad
            assert gradient is None or torch.count_nonzero(gradient).item() == 0
            self.events.append("cleared-before-forward")
        self.events.append("forward")
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        positive = self.value.expand_as(inputs)
        return torch.stack((positive, torch.zeros_like(positive)), dim=-1)


class ObservedSGD(torch.optim.SGD):
    """Use real SGD arithmetic while recording the gradients supplied to each step."""

    def __init__(self, model):
        super().__init__(model.parameters(), lr=1.0)
        self.model = model
        self.step_gradients = []

    def step(self, closure=None):
        self.model.events.append("step")
        gradient = self.model.value.grad
        assert gradient is not None
        self.step_gradients.append(gradient.detach().clone())
        return super().step(closure)


class BrokenDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError("controlled data failure")


@pytest.fixture
def uneven_loader():
    # Four target tokens in batch 0 and two in the non-full final batch.
    inputs = torch.ones((3, 2), dtype=torch.long)
    targets = torch.tensor([[0, 0], [0, 0], [1, 1]], dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False, num_workers=0)


def replace_public_loss(monkeypatch, train_module, replacement):
    """Observe the existing loss API, supporting both module and direct imports."""
    original = loss_module.next_token_loss
    monkeypatch.setattr(loss_module, "next_token_loss", replacement)
    for name, value in tuple(vars(train_module).items()):
        if value is original:
            monkeypatch.setattr(train_module, name, replacement)


@pytest.mark.parametrize(
    ("learning_rate", "options", "expected_decay"),
    [
        (0.05, {}, 0.0),
        (0.05, {"weight_decay": 0.2}, 0.2),
        (Fraction(1, 100), {"weight_decay": Fraction(1, 10)}, 0.1),
        (1, {"weight_decay": 0}, 0.0),
    ],
    ids=["default-zero-decay", "positive-decay", "finite-real-parameters", "integer-real-parameters"],
)
def test_stage42_optimizer_is_adamw_bound_to_current_model(
    train_module, learning_rate, options, expected_decay
):
    """REQ-4.2.1: The optimizer owns all current trainable parameters exactly once."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    expected_parameters = {id(parameter) for parameter in model.parameters()}

    optimizer = train_module.create_optimizer(model, learning_rate=learning_rate, **options)

    assert isinstance(optimizer, torch.optim.AdamW)
    managed_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert {id(parameter) for parameter in managed_parameters} == expected_parameters
    assert len(managed_parameters) == len(expected_parameters)
    assert all(parameter.requires_grad for parameter in model.parameters())
    for group in optimizer.param_groups:
        assert float(group["lr"]) == pytest.approx(float(learning_rate))
        assert float(group["weight_decay"]) == pytest.approx(expected_decay)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param("learning_rate", value, id=f"lr-{label}")
        for label, value in (
            ("zero", 0), ("negative", -0.1), ("nan", float("nan")),
            ("positive-inf", float("inf")), ("negative-inf", -float("inf")),
            ("true", True), ("false", False), ("string", "0.1"), ("none", None),
        )
    ] + [
        pytest.param("weight_decay", value, id=f"decay-{label}")
        for label, value in (
            ("negative", -0.1), ("nan", float("nan")),
            ("positive-inf", float("inf")), ("negative-inf", -float("inf")),
            ("true", True), ("false", False), ("string", "0.1"), ("none", None),
        )
    ],
)
def test_stage42_optimizer_rejects_invalid_hyperparameters(train_module, parameter, value):
    """REQ-4.2.1: Learning rate is strictly positive; decay may be zero."""
    options = dict(learning_rate=0.1, weight_decay=0.0)
    options[parameter] = value

    with pytest.raises(Exception):
        train_module.create_optimizer(ScalarLogits(), **options)


def test_stage42_training_order_updates_gradient_isolation_and_token_weighted_loss(
    train_module, uneven_loader, monkeypatch
):
    """REQ-4.2.2, REQ-4.2.3, REQ-4.2.4: Two exact SGD steps have known gradients."""
    model = ScalarLogits(require_cleared=True).eval()
    model.value.grad = torch.tensor(11.0)
    optimizer = ObservedSGD(model)
    original_loss = loss_module.next_token_loss

    def observed_loss(logits, targets):
        model.events.append("loss")
        loss = original_loss(logits, targets)
        loss.register_hook(lambda gradient: model.events.append("backward"))
        return loss

    replace_public_loss(monkeypatch, train_module, observed_loss)
    expected_second_gradient = 1.0 / (1.0 + math.exp(-0.5))
    expected_parameter = 0.5 - expected_second_gradient
    # First loss is log(2). The second uses theta=0.5 BEFORE its update.
    first_loss = math.log(2)
    second_loss = math.log1p(math.exp(0.5))
    expected_epoch_loss = (4 * first_loss + 2 * second_loss) / 6
    assert abs(expected_epoch_loss - (first_loss + second_loss) / 2) > 0.01

    actual = train_module.train_one_epoch(model, uneven_loader, optimizer)

    assert model.events == [
        "cleared-before-forward", "forward", "loss", "backward", "step",
    ] * 2
    assert model.forward_states == [(True, True), (True, True)]
    assert model.training
    assert len(optimizer.step_gradients) == 2
    torch.testing.assert_close(optimizer.step_gradients[0], torch.tensor(-0.5))
    torch.testing.assert_close(
        optimizer.step_gradients[1], torch.tensor(expected_second_gradient),
        rtol=1e-6, atol=1e-7,
    )
    torch.testing.assert_close(
        model.value.detach(), torch.tensor(expected_parameter), rtol=1e-6, atol=1e-7
    )
    assert type(actual) is float
    assert actual == pytest.approx(expected_epoch_loss, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("initial_training", [True, False], ids=["from-train", "from-eval"])
def test_stage42_evaluation_is_read_only_weighted_and_restores_mode(
    train_module, uneven_loader, monkeypatch, initial_training
):
    """REQ-4.2.4, REQ-4.2.5: Validation preserves both parameters and existing gradients."""
    model = ScalarLogits(initial=math.log(3)).train(initial_training)
    model.value.grad = torch.tensor(7.0)
    before = model.value.detach().clone()
    gradient_before = model.value.grad.clone()

    def forbidden_zero_grad(*args, **kwargs):
        pytest.fail("evaluation must not clear existing gradients")

    monkeypatch.setattr(model, "zero_grad", forbidden_zero_grad)
    original_loss = loss_module.next_token_loss

    def observed_loss(logits, targets):
        assert not torch.is_grad_enabled()
        assert not logits.requires_grad
        return original_loss(logits, targets)

    replace_public_loss(monkeypatch, train_module, observed_loss)
    expected = (4 * math.log(4 / 3) + 2 * math.log(4)) / 6

    actual = train_module.evaluate_loss(model, uneven_loader, device="cpu")

    assert type(actual) is float
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert model.forward_states == [(False, False), (False, False)]
    assert model.training is initial_training
    assert torch.equal(model.value, before)
    assert torch.equal(model.value.grad, gradient_before)


@pytest.mark.parametrize("initial_training", [True, False], ids=["from-train", "from-eval"])
@pytest.mark.parametrize("failure", ["data", "forward", "loss"])
def test_stage42_evaluation_restores_mode_after_exception(
    train_module, uneven_loader, monkeypatch, initial_training, failure
):
    """REQ-4.2.5: Data iteration, forward and loss errors cannot leave the wrong mode."""
    model = ScalarLogits().train(initial_training)
    before = model.value.detach().clone()
    loader = uneven_loader

    def fail(*args, **kwargs):
        raise RuntimeError(f"controlled {failure} failure")

    if failure == "data":
        loader = DataLoader(BrokenDataset(), batch_size=1, num_workers=0)
    elif failure == "forward":
        monkeypatch.setattr(model, "forward", fail)
    else:
        replace_public_loss(monkeypatch, train_module, fail)

    with pytest.raises(Exception):
        train_module.evaluate_loss(model, loader)

    assert model.training is initial_training
    assert torch.equal(model.value, before)


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_stage42_empty_dataloader_is_rejected(train_module, operation):
    """REQ-4.2.6: An empty epoch cannot manufacture zero or NaN loss."""
    model = ScalarLogits()
    empty = torch.empty((0, 2), dtype=torch.long)
    loader = DataLoader(TensorDataset(empty, empty.clone()), batch_size=2)
    before = model.value.detach().clone()

    with pytest.raises(Exception):
        if operation == "train":
            train_module.train_one_epoch(model, loader, ObservedSGD(model))
        else:
            train_module.evaluate_loss(model, loader)

    assert torch.equal(model.value, before)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_stage42_nonfinite_training_loss_stops_before_current_backward_and_step(
    train_module, uneven_loader, monkeypatch, nonfinite
):
    """REQ-4.2.6: Batch 0 may finish; nonfinite batch 1 must not backpropagate or update."""
    model = ScalarLogits(require_cleared=True)
    optimizer = ObservedSGD(model)
    original_loss = loss_module.next_token_loss
    calls = []

    def controlled_loss(logits, targets):
        calls.append(1)
        loss = original_loss(logits, targets)
        if len(calls) == 2:
            loss = loss * nonfinite
        loss.register_hook(lambda gradient: model.events.append("backward"))
        return loss

    replace_public_loss(monkeypatch, train_module, controlled_loss)

    with pytest.raises(Exception):
        train_module.train_one_epoch(model, uneven_loader, optimizer)

    assert len(calls) == 2
    assert model.events.count("backward") == 1
    assert len(optimizer.step_gradients) == 1
    torch.testing.assert_close(model.value.detach(), torch.tensor(0.5))


@pytest.mark.parametrize("initial_training", [True, False], ids=["from-train", "from-eval"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_stage42_nonfinite_validation_loss_reports_error_and_restores_mode(
    train_module, uneven_loader, monkeypatch, initial_training, nonfinite
):
    """REQ-4.2.5, REQ-4.2.6: A nonfinite validation loss still preserves state."""
    model = ScalarLogits().train(initial_training)
    model.value.grad = torch.tensor(7.0)
    before = model.value.detach().clone()
    original_loss = loss_module.next_token_loss

    def nonfinite_loss(logits, targets):
        assert not torch.is_grad_enabled()
        return original_loss(logits, targets) * nonfinite

    replace_public_loss(monkeypatch, train_module, nonfinite_loss)

    with pytest.raises(Exception):
        train_module.evaluate_loss(model, uneven_loader)

    assert model.training is initial_training
    assert torch.equal(model.value, before)
    assert torch.equal(model.value.grad, torch.tensor(7.0))


def test_stage42_dataset_minigpt_adamw_training_and_validation_integrate(train_module):
    """REQ-4.2.1 through REQ-4.2.6: One real CPU batch updates MiniGPT then validates."""
    dataset = NextTokenDataset([0, 1, 2, 3, 0], context_length=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = MiniGPT(
            vocab_size=4, context_size=3, embedding_dim=8,
            num_layers=1, num_heads=2, mlp_hidden_dim=12, dropout=0.0,
        ).eval()
    optimizer = train_module.create_optimizer(model, learning_rate=0.01)
    parameters = list(model.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    forward_states = []
    gradients = []

    def record_forward(module, args, output):
        assert args[0].dtype == torch.long and args[0].shape == (2, 3)
        assert output.shape == (2, 3, 4) and torch.isfinite(output).all()
        forward_states.append((module.training, torch.is_grad_enabled()))

    def record_gradient(gradient):
        gradients.append(gradient.detach().clone())

    handles = [model.register_forward_hook(record_forward)]
    handles.extend(parameter.register_hook(record_gradient) for parameter in parameters)
    try:
        training_loss = train_module.train_one_epoch(
            model, loader, optimizer, device=torch.device("cpu")
        )
        after_training = [parameter.detach().clone() for parameter in parameters]
        assert any(
            not torch.equal(old, new) for old, new in zip(before, after_training, strict=True)
        )
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        gradient_count = len(gradients)

        validation_loss = train_module.evaluate_loss(model, loader)
    finally:
        for handle in handles:
            handle.remove()

    assert type(training_loss) is type(validation_loss) is float
    assert math.isfinite(training_loss) and math.isfinite(validation_loss)
    assert forward_states == [(True, True), (False, False)]
    assert model.training
    assert len(gradients) == gradient_count
    for parameter, snapshot in zip(parameters, after_training, strict=True):
        assert torch.equal(parameter, snapshot)


def test_stage42_adamw_steps_with_fraction_hyperparameters(train_module):
    """REQ-4.2.1: Accepted real hyperparameters must support an actual update."""
    model = ScalarLogits(initial=0.5)
    reference = ScalarLogits(initial=0.5)
    optimizer = train_module.create_optimizer(
        model, learning_rate=Fraction(1, 100), weight_decay=Fraction(1, 10)
    )
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
    model.value.grad = torch.tensor(0.25)
    reference.value.grad = torch.tensor(0.25)
    reference_optimizer.step()

    optimizer.step()

    torch.testing.assert_close(model.value, reference.value)


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_stage42_accepts_dataloader_without_length(train_module, operation):
    """REQ-4.2.6: A DataLoader can yield valid batches without a length."""
    class OneSampleStream(IterableDataset):
        def __iter__(self):
            yield torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long)

    loader = DataLoader(OneSampleStream(), batch_size=1)
    model = ScalarLogits()

    if operation == "train":
        actual = train_module.train_one_epoch(model, loader, ObservedSGD(model))
        torch.testing.assert_close(model.value.detach(), torch.tensor(0.5))
    else:
        actual = train_module.evaluate_loss(model, loader)
        torch.testing.assert_close(model.value.detach(), torch.tensor(0.0))

    assert type(actual) is float
    assert actual == pytest.approx(math.log(2))


def test_stage42_training_leaves_model_device_placement_to_caller(
    train_module, uneven_loader, monkeypatch
):
    """Contract section 2.2: The caller places the model before training."""
    model = ScalarLogits()
    optimizer = ObservedSGD(model)

    def reject_model_move(*args, **kwargs):
        pytest.fail("train_one_epoch must leave model placement to the caller")

    monkeypatch.setattr(model, "to", reject_model_move)

    loss = train_module.train_one_epoch(model, uneven_loader, optimizer, device="cpu")

    assert math.isfinite(loss)
    assert len(optimizer.step_gradients) == 2

import importlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset
from src.model import MiniGPT
from src.train import create_optimizer, train_one_epoch


REQUIRED_FIELDS = (
    "format_version", "completed_epochs", "model_state",
    "optimizer_state", "cpu_rng_state",
)


class UnsupportedCheckpointValue:
    """Inert custom object that does not belong in a weights-only file."""


@pytest.fixture
def checkpoint():
    return importlib.import_module("src.checkpoint")


@pytest.fixture(autouse=True)
def fixed_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        yield


@pytest.fixture
def updated_linear():
    model = nn.Linear(2, 2)
    model.register_buffer("running_scale", torch.tensor([2.0, 3.0]))
    optimizer = create_optimizer(model, learning_rate=0.02, weight_decay=0.1)
    model(torch.tensor([[1.0, -2.0]])).square().sum().backward()
    optimizer.step()
    assert optimizer.state
    return model, optimizer


@pytest.fixture
def valid_payload(updated_linear):
    """Construct files independently of the production writer."""
    model, optimizer = updated_linear
    return {
        "format_version": 1,
        "completed_epochs": 2,
        "model_state": deepcopy(model.state_dict()),
        "optimizer_state": deepcopy(optimizer.state_dict()),
        "cpu_rng_state": torch.get_rng_state(),
    }


def tiny_model(dropout=0.0):
    return MiniGPT(2, 4, 8, 1, 2, 16, dropout=dropout)


def ordered_loader():
    return DataLoader(
        NextTokenDataset([0, 1] * 6, context_length=4),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )


@pytest.mark.parametrize("training,epoch", [(True, 0), (False, 2)])
def test_stage43_save_has_complete_state_and_preserves_caller(
    checkpoint, updated_linear, tmp_path, training, epoch
):
    """REQ-4.3.1: Compare a safe read of the file to independent snapshots."""
    model, optimizer = updated_linear
    model.train(training)
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state()
    path = tmp_path / "state"

    assert checkpoint.save_checkpoint(path, model, optimizer, epoch=epoch) is None

    assert model.training is training
    torch.testing.assert_close(model.state_dict(), model_state, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), optimizer_state, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(REQUIRED_FIELDS) <= payload.keys()
    assert type(payload["format_version"]) is int and payload["format_version"] == 1
    assert type(payload["completed_epochs"]) is int and payload["completed_epochs"] == epoch
    torch.testing.assert_close(payload["model_state"], model_state, rtol=0, atol=0)
    torch.testing.assert_close(payload["optimizer_state"], optimizer_state, rtol=0, atol=0)
    assert torch.equal(payload["cpu_rng_state"], rng)


@pytest.mark.parametrize("epoch", [-1, True, False, 1.0, "1"])
def test_stage43_save_rejects_invalid_epoch_before_writing(checkpoint, updated_linear, tmp_path, epoch):
    """REQ-4.3.1, REQ-4.3.5: Invalid progress must leave the destination untouched."""
    model, optimizer = updated_linear
    path = tmp_path / "existing"
    original = b"existing destination contents"
    path.write_bytes(original)
    with pytest.raises(Exception):
        checkpoint.save_checkpoint(path, model, optimizer, epoch=epoch)
    assert path.read_bytes() == original


@pytest.mark.parametrize("destination", ["missing-parent", "directory"])
def test_stage43_save_rejects_unwritable_destination(checkpoint, updated_linear, tmp_path, destination):
    """REQ-4.3.1: Filesystem errors are observable with portable temporary paths."""
    model, optimizer = updated_linear
    path = tmp_path / "absent" / "state" if destination == "missing-parent" else tmp_path
    with pytest.raises(Exception):
        checkpoint.save_checkpoint(path, model, optimizer, epoch=0)


def test_stage43_model_only_restores_trained_weights_and_logits(checkpoint, tmp_path, monkeypatch):
    """REQ-4.3.2: Load into a different MiniGPT instance without moving it or changing mode."""
    model = tiny_model()
    optimizer = create_optimizer(model, learning_rate=0.02)
    train_one_epoch(model, ordered_loader(), optimizer)
    expected = deepcopy(model.state_dict())
    inputs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        logits = model.eval()(inputs)
    path = str(tmp_path / "model")
    checkpoint.save_checkpoint(path, model, optimizer, epoch=1)
    restored = tiny_model().train()
    assert any(not torch.equal(value, restored.state_dict()[key]) for key, value in expected.items())

    def forbid_move(*args, **kwargs):
        pytest.fail("Checkpoint loading must leave device placement to the caller")

    monkeypatch.setattr(restored, "to", forbid_move)
    epoch = checkpoint.load_checkpoint(
        path, restored, optimizer=None, map_location=torch.device("cpu"), restore_rng=False
    )
    assert type(epoch) is int and epoch == 1
    assert restored.training
    torch.testing.assert_close(restored.state_dict(), expected, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(inputs), logits, rtol=1e-6, atol=1e-7)


def test_stage43_restores_persistent_buffer(checkpoint, updated_linear, tmp_path):
    """REQ-4.3.2: Buffers are restored along with parameters."""
    model, optimizer = updated_linear
    path = tmp_path / "buffer"
    checkpoint.save_checkpoint(path, model, optimizer, epoch=1)
    restored = nn.Linear(2, 2)
    restored.register_buffer("running_scale", torch.zeros(2))
    checkpoint.load_checkpoint(path, restored, restore_rng=False)
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)


def test_stage43_adamw_and_dropout_resume_match_uninterrupted_epoch(checkpoint, tmp_path):
    """REQ-4.3.3, REQ-4.3.4: A real uninterrupted path supplies the reference."""
    model = tiny_model(dropout=0.25)
    optimizer = create_optimizer(model, learning_rate=0.02, weight_decay=0.1)
    loader = ordered_loader()
    train_one_epoch(model, loader, optimizer)
    saved_optimizer = deepcopy(optimizer.state_dict())
    assert saved_optimizer["state"]
    for state in saved_optimizer["state"].values():
        assert state["step"].item() == 2
        assert "exp_avg" in state and "exp_avg_sq" in state
    assert any(state["exp_avg"].count_nonzero() for state in saved_optimizer["state"].values())
    path = tmp_path / "epoch"
    checkpoint.save_checkpoint(path, model, optimizer, epoch=1)
    reference_loss = train_one_epoch(model, loader, optimizer)
    reference_rng = torch.get_rng_state()

    restored = tiny_model(dropout=0.25).eval()
    restored_optimizer = create_optimizer(restored, learning_rate=0.005)
    assert not restored_optimizer.state
    assert checkpoint.load_checkpoint(path, restored, restored_optimizer) == 1
    assert not restored.training
    torch.testing.assert_close(restored_optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
    restored_loss = train_one_epoch(restored, loader, restored_optimizer)

    assert restored_loss == pytest.approx(reference_loss, rel=1e-6, abs=1e-7)
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        restored_optimizer.state_dict(), optimizer.state_dict(), rtol=1e-6, atol=1e-7
    )
    assert torch.equal(torch.get_rng_state(), reference_rng)


@pytest.mark.parametrize("restore_rng", [True, False], ids=["restore", "preserve-caller"])
def test_stage43_load_rng_controls_next_cpu_draws(checkpoint, valid_payload, tmp_path, restore_rng):
    """REQ-4.3.4: A known file and generator determine the next draws independently."""
    torch.manual_seed(71)
    valid_payload["cpu_rng_state"] = torch.get_rng_state()
    path = tmp_path / "rng"
    torch.save(valid_payload, path)
    saved_next = torch.rand(8)
    model = nn.Linear(2, 2)
    model.register_buffer("running_scale", torch.zeros(2))
    torch.manual_seed(99)
    caller_rng = torch.get_rng_state()
    caller_next = torch.rand(8, generator=torch.Generator().set_state(caller_rng))

    epoch = checkpoint.load_checkpoint(path, model, restore_rng=restore_rng)

    assert type(epoch) is int and epoch == 2
    expected_rng = valid_payload["cpu_rng_state"] if restore_rng else caller_rng
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(torch.rand(8), saved_next if restore_rng else caller_next)


@pytest.mark.parametrize("bad_value", [0, 1, None, "true"])
def test_stage43_load_rejects_nonboolean_restore_rng(
    checkpoint, updated_linear, valid_payload, tmp_path, bad_value
):
    """REQ-4.3.4: Truthy and falsy substitutes are outside the Boolean API."""
    model, _ = updated_linear
    path = tmp_path / "valid"
    torch.save(valid_payload, path)
    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, restore_rng=bad_value)


@pytest.mark.parametrize("corrupt", [False, True], ids=["missing", "corrupt"])
def test_stage43_load_rejects_unreadable_files(checkpoint, updated_linear, tmp_path, corrupt):
    """REQ-4.3.5: Missing and damaged files cannot report a successful load."""
    model, optimizer = updated_linear
    path = tmp_path / "invalid"
    if corrupt:
        path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, optimizer)


def test_stage43_load_rejects_custom_objects(checkpoint, updated_linear, valid_payload, tmp_path):
    """REQ-4.3.5 and section 3: The loader rejects nonstandard Python objects."""
    model, optimizer = updated_linear
    valid_payload["extra"] = UnsupportedCheckpointValue()
    path = tmp_path / "custom-object"
    torch.save(valid_payload, path)
    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, optimizer)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_stage43_load_rejects_each_missing_field(
    checkpoint, updated_linear, valid_payload, tmp_path, field
):
    """REQ-4.3.5: Each required field is removed separately from a valid payload."""
    model, optimizer = updated_linear
    del valid_payload[field]
    path = tmp_path / "incomplete"
    torch.save(valid_payload, path)
    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, optimizer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", 2), ("format_version", "1"),
        ("completed_epochs", -1), ("completed_epochs", True), ("completed_epochs", 1.5),
        ("optimizer_state", None), ("optimizer_state", {"state": {}}),
        ("cpu_rng_state", torch.tensor([1.0])),
    ],
    ids=["unknown-version", "string-version", "negative-epoch", "boolean-epoch",
         "fractional-epoch", "null-optimizer", "incomplete-optimizer", "invalid-rng"],
)
def test_stage43_load_rejects_invalid_fields(
    checkpoint, updated_linear, valid_payload, tmp_path, field, value
):
    """REQ-4.3.5: Invalid metadata and state are distinct from file corruption."""
    model, optimizer = updated_linear
    valid_payload[field] = value
    path = tmp_path / "invalid-field"
    torch.save(valid_payload, path)
    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, optimizer)


@pytest.mark.parametrize("mismatch", ["shape", "missing-key", "extra-key"])
def test_stage43_load_rejects_model_mismatch(
    checkpoint, updated_linear, valid_payload, tmp_path, mismatch
):
    """REQ-4.3.2, REQ-4.3.5: Model loading must be strict about shapes and keys."""
    model, _ = updated_linear
    if mismatch == "shape":
        valid_payload["model_state"]["weight"] = torch.zeros(3, 2)
    elif mismatch == "missing-key":
        del valid_payload["model_state"]["bias"]
    else:
        valid_payload["model_state"]["extra"] = torch.zeros(1)
    path = tmp_path / "incompatible"
    torch.save(valid_payload, path)
    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, optimizer=None, restore_rng=False)


def test_stage43_checkpoint_imports_without_test_package():
    """The production checkpoint API is available without the test package."""
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; sys.modules['tests'] = None; import src.checkpoint"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_stage43_save_requires_an_optimizer(checkpoint, updated_linear, tmp_path):
    """REQ-4.3.1: Saving requires the caller's optimizer."""
    model, _ = updated_linear

    with pytest.raises(Exception):
        checkpoint.save_checkpoint(tmp_path / "without-optimizer", model, None, epoch=0)


@pytest.mark.parametrize("field", ["optimizer_state", "cpu_rng_state"])
def test_stage43_model_only_load_still_requires_complete_file(
    checkpoint, updated_linear, valid_payload, tmp_path, field
):
    """REQ-4.3.5: Skipping restoration does not make file fields optional."""
    model, _ = updated_linear
    del valid_payload[field]
    path = tmp_path / "incomplete"
    torch.save(valid_payload, path)

    with pytest.raises(Exception):
        checkpoint.load_checkpoint(path, model, optimizer=None, restore_rng=False)

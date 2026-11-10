import importlib
from array import array
from fractions import Fraction

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharTokenizer


@pytest.fixture
def dataset_module():
    # Keep all tests collectable while the developer implements this module.
    return importlib.import_module("src.dataset")


@pytest.mark.parametrize(
    ("text", "as_string_path"),
    [("English text\nSecond line\n", True),
     ("你好，世界！\n第二行", False),
     ("a中🙂é\u0301\t\n\n终\n", False)],
    ids=["english", "chinese", "mixed-unicode-and-newlines"],
)
def test_stage41_read_utf8_preserves_text(dataset_module, tmp_path, text, as_string_path):
    """REQ-4.1.1: Decode known UTF-8 bytes without dropping characters or LF newlines."""
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    argument = str(path) if as_string_path else path

    assert dataset_module.read_text_utf8(argument) == text


def test_stage41_read_utf8_reports_missing_file(dataset_module, tmp_path):
    """REQ-4.1.1: A missing source file cannot produce an empty corpus silently."""
    with pytest.raises(Exception):
        dataset_module.read_text_utf8(tmp_path / "missing.txt")


def test_stage41_read_utf8_rejects_invalid_encoding(dataset_module, tmp_path):
    """REQ-4.1.1: Invalid UTF-8 cannot be replaced or ignored."""
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"valid prefix\xff\xfe")

    with pytest.raises(Exception):
        dataset_module.read_text_utf8(path)


@pytest.mark.parametrize(
    ("fraction", "expected_train", "expected_validation"),
    [
        (0.1, "a", "bcdef你好🙂9"),
        (0.26, "ab", "cdef你好🙂9"),
        (0.5, "abcde", "f你好🙂9"),
        (0.999, "abcdef你好🙂", "9"),
        (Fraction(1, 4), "ab", "cdef你好🙂9"),
    ],
    ids=["first-nonempty", "floor-index", "half", "near-one", "real-fraction"],
)
def test_stage41_split_is_contiguous_and_uses_character_floor(
    dataset_module, fraction, expected_train, expected_validation
):
    """REQ-4.1.2: Literal disjoint slices expose shuffling, overlap and rounding."""
    text = "abcdef你好🙂9"

    train, validation = dataset_module.split_text(text, train_fraction=fraction)

    assert (train, validation) == (expected_train, expected_validation)
    assert train + validation == text
    assert set(train).isdisjoint(validation)


def test_stage41_split_default_fraction_is_point_nine(dataset_module):
    """REQ-4.1.2: The default split uses the declared 0.9 ratio."""
    assert dataset_module.split_text("abcdefghij") == ("abcdefghi", "j")


@pytest.mark.parametrize(
    "fraction",
    [0, 1, -0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, False, "0.5"],
    ids=["zero", "one", "negative", "above-one", "nan", "positive-inf",
         "negative-inf", "true", "false", "string"],
)
def test_stage41_split_rejects_invalid_fraction(dataset_module, fraction):
    """REQ-4.1.2: Fractions must be finite real numbers strictly between zero and one."""
    with pytest.raises(Exception):
        dataset_module.split_text("abcdefghij", train_fraction=fraction)


@pytest.mark.parametrize(
    ("text", "fraction"), [("", 0.5), ("a", 0.5), ("ab", 0.1)],
    ids=["empty-text", "one-character", "rounded-empty-training"],
)
def test_stage41_split_rejects_empty_result(dataset_module, text, fraction):
    """REQ-4.1.2: Both resulting texts must contain at least one character."""
    with pytest.raises(Exception):
        dataset_module.split_text(text, train_fraction=fraction)


@pytest.mark.parametrize(
    ("ids", "length", "expected"),
    [
        (
            [10, 11, 12, 13, 14, 15], 3,
            [([10, 11, 12], [11, 12, 13]),
             ([11, 12, 13], [12, 13, 14]),
             ([12, 13, 14], [13, 14, 15])],
        ),
        ((0, 1, 2, 3), 3, [([0, 1, 2], [1, 2, 3])]),
        ([2, 0, 1], 1, [([2], [0]), ([0], [1])]),
    ],
    ids=["all-three-windows", "minimum-corpus-tuple", "one-token-context"],
)
def test_stage41_dataset_matches_every_handwritten_window(
    dataset_module, ids, length, expected
):
    """REQ-4.1.3, REQ-4.1.4: Include every window, especially the final target token."""
    dataset = dataset_module.NextTokenDataset(ids, context_length=length)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == len(expected) == len(ids) - length
    for index, (expected_x, expected_y) in enumerate(expected):
        sample = dataset[index]
        assert isinstance(sample, tuple) and len(sample) == 2
        x, y = sample
        assert x.dtype == y.dtype == torch.long
        assert x.shape == y.shape == (length,)
        assert x.tolist() == expected_x
        assert y.tolist() == expected_y


def test_stage41_dataloader_preserves_window_order_and_partial_batch(dataset_module):
    """REQ-4.1.4: The final one-sample batch retains its target alignment."""
    dataset = dataset_module.NextTokenDataset([10, 11, 12, 13, 14, 15], context_length=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    expected = [
        ([[10, 11, 12], [11, 12, 13]], [[11, 12, 13], [12, 13, 14]]),
        ([[12, 13, 14]], [[13, 14, 15]]),
    ]
    batches = list(loader)

    assert len(batches) == 2
    for (x, y), (expected_x, expected_y) in zip(batches, expected, strict=True):
        assert x.dtype == y.dtype == torch.long
        assert x.shape == y.shape == (len(expected_x), 3)
        assert x.tolist() == expected_x
        assert y.tolist() == expected_y


@pytest.mark.parametrize("index", [3, 4], ids=["at-length", "beyond-length"])
def test_stage41_dataset_rejects_out_of_range_index(dataset_module, index):
    """REQ-4.1.4: Only the declared three nonnegative indices are in range."""
    dataset = dataset_module.NextTokenDataset([0, 1, 2, 3, 4, 5], context_length=3)

    with pytest.raises(IndexError):
        dataset[index]


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_stage41_dataset_rejects_invalid_context_length(dataset_module, length):
    """REQ-4.1.4: Context length must be a positive integer and cannot be bool."""
    with pytest.raises(Exception):
        dataset_module.NextTokenDataset([0, 1, 2, 3], context_length=length)


@pytest.mark.parametrize(
    "ids", [[0, -1, 2, 3], [0, True, 2, 3], [0, 1.5, 2, 3], [0, "1", 2, 3]],
    ids=["negative-id", "boolean-id", "fractional-id", "string-id"],
)
def test_stage41_dataset_rejects_invalid_token_ids(dataset_module, ids):
    """REQ-4.1.4: Invalid IDs cannot be cast into valid integers silently."""
    with pytest.raises(Exception):
        dataset_module.NextTokenDataset(ids, context_length=2)


@pytest.mark.parametrize("ids", [[], [0, 1], [0, 1, 2]])
def test_stage41_dataset_requires_context_plus_one_tokens(dataset_module, ids):
    """REQ-4.1.4: Even a corpus of exactly T tokens has no next-token sample."""
    with pytest.raises(Exception):
        dataset_module.NextTokenDataset(ids, context_length=3)


def test_stage41_returned_tensors_do_not_alias_mutable_source_storage(dataset_module):
    """REQ-4.1.4: A valid integer Sequence backed by a buffer stays unchanged."""
    original = [0, 1, 2, 3, 4]
    source = array("q", original)
    dataset = dataset_module.NextTokenDataset(source, context_length=3)
    x, y = dataset[0]
    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]

    x[0] = 99
    y[-1] = 88

    assert source.tolist() == original


def test_stage41_train_validation_windows_stay_inside_their_texts(dataset_module):
    """REQ-4.1.2, REQ-4.1.3: Training-only fitting permits valid separate datasets."""
    train_text, validation_text = dataset_module.split_text("abcacaba", train_fraction=0.5)
    tokenizer = CharTokenizer()
    tokenizer.fit(train_text)
    train = dataset_module.NextTokenDataset(tokenizer.encode(train_text), context_length=2)
    validation = dataset_module.NextTokenDataset(
        tokenizer.encode(validation_text), context_length=2
    )
    expected = [
        (train, [("ab", "bc"), ("bc", "ca")]),
        (validation, [("ca", "ab"), ("ab", "ba")]),
    ]

    for dataset, windows in expected:
        assert len(dataset) == 2
        for index, (expected_x, expected_y) in enumerate(windows):
            x, y = dataset[index]
            assert tokenizer.decode(x.tolist()) == expected_x
            assert tokenizer.decode(y.tolist()) == expected_y


def test_stage41_validation_oov_is_reported_without_expanding_training_vocabulary(
    dataset_module,
):
    """REQ-4.1.2 and contract section 4: Validation cannot teach new characters."""
    train, validation = dataset_module.split_text("abababab界a", train_fraction=0.6)
    tokenizer = CharTokenizer()
    tokenizer.fit(train)
    assert tokenizer.vocab_size == 2

    with pytest.raises(ValueError):
        tokenizer.encode(validation)

    assert tokenizer.vocab_size == 2

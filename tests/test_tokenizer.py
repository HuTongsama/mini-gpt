import pytest

from src.tokenizer import CharTokenizer


VOCABULARY = "ab你好🙂é\u0301 \n"


@pytest.fixture
def tokenizer():
    tokenizer = CharTokenizer()
    tokenizer.fit(VOCABULARY + "aa你🙂")
    return tokenizer


def test_fit_builds_one_id_per_distinct_character(tokenizer):
    """REQ-1.1: Fit learns a bijection from a corpus with repeated characters."""
    ids = tokenizer.encode(VOCABULARY)

    assert len(ids) == len(VOCABULARY)
    assert all(isinstance(token_id, int) for token_id in ids)
    assert len(set(ids)) == len(VOCABULARY) == tokenizer.vocab_size
    for char, token_id in zip(VOCABULARY, ids, strict=True):
        assert tokenizer.encode(char) == [token_id]
        assert tokenizer.decode([token_id]) == char


@pytest.mark.parametrize(
    "sample",
    ["abba", "好你好", "🙂你a\u0301 é\nb🙂", ""],
    ids=["english", "chinese", "mixed-unicode", "empty"],
)
def test_encode_decode_preserves_character_order(tokenizer, sample):
    """REQ-1.1: Encoding preserves the sequence of learned character IDs."""
    expected_ids = [tokenizer.encode(char)[0] for char in sample]

    assert tokenizer.encode(sample) == expected_ids
    assert tokenizer.decode(expected_ids) == sample


def test_encode_rejects_unknown_character(tokenizer):
    """REQ-1.1: An unknown character inside known text raises an error."""
    with pytest.raises(Exception):
        tokenizer.encode("a界好")


def test_save_and_load_preserve_vocabulary(tokenizer, tmp_path):
    """REQ-1.1: A new tokenizer restores both directions of the mapping."""
    path = tmp_path / "vocabulary.json"
    original_ids = tokenizer.encode(VOCABULARY)
    tokenizer.save_vocab(str(path))

    restored = CharTokenizer()
    restored.load_vocab(str(path))

    assert restored.vocab_size == tokenizer.vocab_size
    assert restored.encode(VOCABULARY) == original_ids
    assert restored.decode(original_ids) == VOCABULARY

import pytest
import torch

from src.embedding import InputEmbedding


@pytest.fixture
def make_embedding():
    def make(vocab_size, context_size, embedding_dim):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            return InputEmbedding(vocab_size, context_size, embedding_dim)

    return make


@pytest.fixture
def fixed_embedding(make_embedding):
    embedding = make_embedding(vocab_size=3, context_size=4, embedding_dim=2)
    with torch.no_grad():
        embedding.token_embedding.weight.copy_(
            torch.tensor([[2.0, -1.0], [4.0, 3.0], [-2.0, 6.0]])
        )
        embedding.position_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [0.0, -3.0], [5.0, 1.0], [-4.0, 0.0]])
        )
    return embedding


def test_stage_one_acceptance_shape(make_embedding):
    """REQ-1.2, REQ-1.3: The reference input produces [2, 16, 128]."""
    embedding = make_embedding(vocab_size=5, context_size=64, embedding_dim=128)
    token_ids = torch.arange(32, dtype=torch.long).reshape(2, 16) % 5

    assert embedding.token_embedding(token_ids).shape == (2, 16, 128)
    assert embedding(token_ids).shape == (2, 16, 128)


def test_token_lookup_depends_only_on_id(fixed_embedding):
    """REQ-1.2: Repeated IDs share a vector across positions and samples."""
    token_ids = torch.tensor([[0, 2, 0], [1, 0, 2]], dtype=torch.long)
    expected = torch.tensor(
        [
            [[2.0, -1.0], [-2.0, 6.0], [2.0, -1.0]],
            [[4.0, 3.0], [2.0, -1.0], [-2.0, 6.0]],
        ]
    )

    torch.testing.assert_close(
        fixed_embedding.token_embedding(token_ids), expected, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "invalid_id", [-1, 3], ids=["below-vocabulary", "above-vocabulary"]
)
def test_input_embedding_rejects_out_of_range_ids(fixed_embedding, invalid_id):
    """REQ-1.2: IDs outside the lookup table are invalid."""
    token_ids = torch.tensor([[1, invalid_id]], dtype=torch.long)

    with pytest.raises(Exception):
        fixed_embedding(token_ids)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length"), [(1, 1), (2, 3), (3, 4)]
)
def test_position_vectors_are_added_and_broadcast(
    fixed_embedding, batch_size, sequence_length
):
    """REQ-1.2, REQ-1.3: Position rows start at zero and repeat across the batch."""
    token_ids = torch.tensor(
        [[0, 0, 0, 0], [1, 2, 0, 1], [2, 1, 2, 0]], dtype=torch.long
    )
    expected_positions = torch.tensor(
        [[1.0, 2.0], [0.0, -3.0], [5.0, 1.0], [-4.0, 0.0]]
    )
    # Hand-calculated sums of the fixed token and position rows.
    expected_output = torch.tensor(
        [
            [[3.0, 1.0], [2.0, -4.0], [7.0, 0.0], [-2.0, -1.0]],
            [[5.0, 5.0], [-2.0, 3.0], [7.0, 0.0], [0.0, 3.0]],
            [[-1.0, 8.0], [4.0, 0.0], [3.0, 7.0], [-2.0, -1.0]],
        ]
    )
    positions = torch.arange(sequence_length)

    torch.testing.assert_close(
        fixed_embedding.position_embedding(positions),
        expected_positions[:sequence_length],
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        fixed_embedding(token_ids[:batch_size, :sequence_length]),
        expected_output[:batch_size, :sequence_length],
        rtol=0,
        atol=1e-6,
    )


def test_input_embedding_rejects_sequence_beyond_context(fixed_embedding):
    """REQ-1.3: A sequence of context_size + 1 raises an error."""
    token_ids = torch.zeros((2, 5), dtype=torch.long)

    with pytest.raises(Exception):
        fixed_embedding(token_ids)


def test_backward_accumulates_token_and_position_gradients(make_embedding):
    """REQ-1.2, REQ-1.3: Both trainable tables accumulate gradients on used rows."""
    embedding = make_embedding(vocab_size=4, context_size=4, embedding_dim=2)
    token_ids = torch.tensor([[0, 1, 0], [2, 0, 1]], dtype=torch.long)
    coefficients = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    (embedding(token_ids) * coefficients.unsqueeze(-1)).sum().backward()

    # Each token row sums coefficients at occurrences of that ID.
    expected_token_grad = torch.tensor([[9.0, 9.0], [8.0, 8.0], [4.0, 4.0], [0.0, 0.0]])
    # Each position row sums its column across both samples.
    expected_position_grad = torch.tensor([[5.0, 5.0], [7.0, 7.0], [9.0, 9.0], [0.0, 0.0]])
    torch.testing.assert_close(
        embedding.token_embedding.weight.grad, expected_token_grad, rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        embedding.position_embedding.weight.grad,
        expected_position_grad,
        rtol=0,
        atol=1e-6,
    )

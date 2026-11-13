"""Run the Tiny Shakespeare CPU experiment with python -m src.main."""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

import torch
from torch.utils.data import DataLoader

from . import checkpoint, dataset, generate, model, tokenizer, train


def train_experiment(corpus_path: Path, run_dir: Path) -> None:
    text = dataset.read_text_utf8(corpus_path)
    train_text, validation_text = dataset.split_text(text, train_fraction=0.9)
    train_samples = train_text[:30000]
    validation_samples = validation_text[:2000]

    ch_tokenizer = tokenizer.CharTokenizer()
    ch_tokenizer.fit(train_text)
    ch_tokenizer.save_vocab(run_dir / "tokenizer.json")

    train_ids = ch_tokenizer.encode(train_samples)
    validation_ids = ch_tokenizer.encode(validation_samples)

    context_size = 64
    train_dataset = dataset.NextTokenDataset(
        train_ids,
        context_length=context_size
    )

    validation_dataset = dataset.NextTokenDataset(
        validation_ids,
        context_length=context_size
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    config = {
        "vocab_size": ch_tokenizer.vocab_size,
        "context_size": context_size,
        "embedding_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_hidden_dim": 512,
        "dropout": 0.1,
        "causal": True,
    }

    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    mini_gpt = model.MiniGPT(**config)
    mini_gpt.to('cpu')

    optimizer = train.create_optimizer(
        model=mini_gpt,
        learning_rate=3e-4,
        weight_decay=0.01)

    epochs = 50
    best_val_loss = float("inf")

    bad_epochs = 0
    patience = 5
    for epoch in range(1, epochs + 1):

        train_loss = train.train_one_epoch(
            mini_gpt, train_loader, optimizer, device="cpu"
        )
        train_eval_loss = train.evaluate_loss(
            mini_gpt, train_loader, device="cpu"
        )
        val_loss = train.evaluate_loss(
            mini_gpt, validation_loader, device="cpu"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint.save_checkpoint(run_dir / "best.pt", mini_gpt, optimizer, epoch=epoch)
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Train Eval Loss: {train_eval_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        if bad_epochs >= patience:
            break


def generate_examples(run_dir: Path) -> None:
    restored_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    restored_tokenizer = tokenizer.CharTokenizer()
    restored_tokenizer.load_vocab(run_dir / "tokenizer.json")
    restored_mini_gpt = model.MiniGPT(**restored_config)
    restored_mini_gpt.to('cpu')
    completed_epochs = checkpoint.load_checkpoint(
        run_dir / "best.pt", restored_mini_gpt
    )
    print(f"Restored model after {completed_epochs} epochs.")

    prompt = "ROMEO:\n"

    prompt_ids = restored_tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(
        [prompt_ids],
        dtype=torch.long
    )

    greedy_ids = generate.generate_greedy(
        restored_mini_gpt,
        prompt_tensor,
        max_new_tokens=200,
        context_size=restored_config["context_size"]
    )

    sample_ids = generate.generate_sample(
        restored_mini_gpt,
        prompt_tensor,
        max_new_tokens=200,
        context_size=restored_config["context_size"],
        temperature=0.8,
        top_k=10,
        generator=torch.Generator().manual_seed(42)
    )

    greedy_text = restored_tokenizer.decode(greedy_ids[0].tolist())
    sample_text = restored_tokenizer.decode(sample_ids[0].tolist())

    print(f"Greedy generation:\n{greedy_text}")
    print(f"Sampled generation:\n{sample_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate-only", type=Path, metavar="RUN_DIR",
        help="reuse the tokenizer, config and best checkpoint from an earlier run",
    )
    args = parser.parse_args()
    if args.generate_only is not None:
        run_dir = args.generate_only
    else:
        corpus_path = Path("data/tiny_shakespeare.txt")
        if not corpus_path.is_file():
            parser.error(f"missing {corpus_path}; see README.md for download instructions")
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        run_dir = Path(mkdtemp(prefix="tiny-shakespeare-", dir=runs_dir))
        print(f"Run artifacts: {run_dir}")
        train_experiment(corpus_path, run_dir)
    generate_examples(run_dir)


if __name__ == "__main__":
    main()

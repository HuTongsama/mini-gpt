# Mini-GPT v0.1.0

An educational character-level GPT baseline in PyTorch. Follow the next-token
prediction pipeline from a character vocabulary to a trained model that generates text.

Mini-GPT includes token and position embeddings, causal multi-head attention,
Transformer blocks, a next-token dataset and loss, CPU training and validation,
checkpoint save/load, greedy decoding, and temperature/top-k sampling.
See the [requirements](docs/requirements.md) and [stage contracts](docs/contracts/).

## Setup and tests

Install [uv](https://docs.astral.sh/uv/). The project requires Python 3.12+
and selects Python 3.12 for local development. The lockfile pins CPU PyTorch,
NumPy, and the pytest development dependency.

From the repository root:

```powershell
uv sync --locked
uv run --locked python -m pytest
```

## Train and generate

Download Tiny Shakespeare from Karpathy's char-rnn repository (PowerShell):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -OutFile "data/tiny_shakespeare.txt"
uv run --locked python -m src.main
```

Run the module from the repository root so it can find the corpus. The experiment
trains on a subset of the text, evaluates validation loss, reloads the best
checkpoint, and prints greedy and sampled continuations. Each run stores its
vocabulary, model config, and checkpoint in a new directory under `runs/`.
Git ignores both `data/` and `runs/`.

To generate again without training, pass the run directory printed at startup:

```powershell
uv run --locked python -m src.main --generate-only runs/tiny-shakespeare-REPLACE_ME
```

This CPU baseline uses a small model with a 64-character context and a vocabulary
limited to characters seen in the training split. Generation recomputes attention
at each step; there is no KV cache. The experiment is for learning and can take
time on CPU.

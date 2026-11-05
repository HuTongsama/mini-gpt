# Mini-GPT

An educational GPT implementation. See [project requirements](docs/requirements.md).

## Python environment

The environment is managed with [uv](https://docs.astral.sh/uv/).
Python 3.12+ is required; `.python-version` selects Python 3.12 for local development.
PyTorch uses the official CPU wheel index, with NumPy installed for tensor interoperability.
pytest is a development dependency.

From the project directory:

```powershell
uv sync --locked
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
uv run pytest
```

`uv sync` creates `.venv` and installs development dependencies by default.
No manual environment activation is needed when using `uv run`.
Commit `pyproject.toml`, `.python-version`, and `uv.lock` to reproduce the environment.

No project tests have been implemented yet; pytest exits with code 5 when no tests are collected.

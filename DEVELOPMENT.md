# Development Guide

This project is the `DMSRoute` Python package and command-line interface for acquiring, parsing, validating, harmonizing, and standardizing DMS datasets from MaveDB and ProteinGym.

## Requirements

The project currently declares:

- Python `>=3.10`
- core runtime dependencies in `pyproject.toml`
- the `dev` extra includes `pytest>=8.0`

## Development installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This uses the repository's actual editable-install workflow and dev extra.

## Repository architecture

The package lives directly at the repository root as `dmsroute/`.

- root-level orchestration modules: `builders.py`, `catalog.py`, `config.py`, `downloads.py`, `pipeline.py`, and `cli.py`
- `dmsroute/core/`: parsing, validation, transformation, and domain primitives
- `dmsroute/acquisition/`: cache, fetch, and I/O infrastructure
- `dmsroute/sources/`: MaveDB and ProteinGym source-specific adapters and catalogs
- `tests/`: automated regression and behavior tests
- `examples/`: usage examples and notebooks

## Testing

Run the full suite:

```bash
pytest -q
```

Run a focused test file:

```bash
pytest -q tests/test_parsing.py
```

Run one focused test:

```bash
pytest -q tests/test_parsing.py -k hgvs_to_sequence
```

Run dependency validation:

```bash
python -m pip check
```

Run compile checking:

```bash
python -m compileall -q dmsroute tests examples
```

## CLI smoke testing

The supported public CLI entry point is `dmsroute`.

Use the following commands:

```bash
dmsroute --help
dmsroute download --help
dmsroute download-many --help
dmsroute run --help
dmsroute snapshot --help
dmsroute cache --help
```

`python -m dmsroute.cli` is intentionally not a supported public invocation and is not documented as such.

## External access policy

Unit tests should not depend on live MaveDB, ProteinGym, Zenodo, or other external services.

Use offline deterministic tests unless a specific test is designed to exercise an external integration path intentionally.

## Packaging validation

For release-oriented checks, use the repository's current packaging workflow:

```bash
python -m build --sdist --wheel
python -m twine check --strict dist/*
```

Optional clean-install validation:

```bash
python -m venv /tmp/dmsroute-check
# activate the temporary environment and install the built wheel
python -m pip install dist/dmsroute-0.1.1-py3-none-any.whl
python -m pip check
dmsroute --help
```

Do not add or require `uv.lock` for normal development.

## Contribution and change principles

- Source adapters should keep source-specific behavior inside `dmsroute/sources/`.
- Generic acquisition logic belongs in `dmsroute/acquisition/`.
- Generic parsing, validation, and domain logic belongs in `dmsroute/core/`.
- Orchestration remains in the root-level facade modules.
- Changes to standardized data contracts require explicit tests.
- Avoid unrelated cleanup while implementing a requested change.

## Release checklist

Before tagging or publishing a release candidate, confirm:

- worktree is clean
- full test suite passes
- `python -m pip check` passes
- project builds successfully
- `twine check --strict` passes
- clean install succeeds in a temporary environment
- CLI smoke checks pass
- version and citation metadata are current
- tag only after all checks pass


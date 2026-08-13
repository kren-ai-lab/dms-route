# AGENTS.md

## Project scope

`DMSRoute` acquires, parses, validates, harmonizes, and standardizes Deep Mutational Scanning (DMS) datasets, with current source integrations for MaveDB and ProteinGym workflows.

This project is not a representation, embedding, model-training, or machine-learning library.

## Language and style

- Python source code in English.
- Comments and docstrings in English.
- User and developer documentation in English.
- Prefer small, explicit helpers with clear responsibility boundaries.
- Do not perform unrelated cleanup while implementing a requested change.

## Package structure

Preserve the current repository-root package layout:

- `dmsroute/`
  - `core/`: parsing, domain, validation, and transformation primitives
  - `acquisition/`: cache, fetch, and I/O infrastructure
  - `sources/`: MaveDB and ProteinGym source-specific integrations
  - root-level orchestration modules: `builders.py`, `catalog.py`, `config.py`, `downloads.py`, `pipeline.py`, and `cli.py`

Do not reintroduce `src/dmsroute/`.

Do not move modules across these boundaries without a concrete architectural reason.

## Public contracts

Preserve documented public APIs.

The public console command is:

- `dmsroute`

The console entry point is:

- `dmsroute = dmsroute.cli:main`

`python -m dmsroute.cli` is intentionally not a supported public execution contract.

Do not add `__main__.py` merely to support that invocation.

## Data and output contracts

Do not silently change:

- standardized output columns
- status and error semantics
- parsed variant semantics
- WT handling
- MaveDB and ProteinGym source contracts
- configuration keys
- CLI options
- score transformation defaults
- cache semantics

Behavioral changes require explicit justification and tests.

## Tests and network behavior

- Unit tests should remain deterministic and offline unless a test is explicitly designed as an external integration check.
- Do not contact MaveDB, ProteinGym, Zenodo, or other external services from ordinary unit tests.
- Mock or fake external interactions where appropriate.

## Git behavior

Agents must not:

- commit
- stage
- push
- tag
- rewrite history
- publish packages

unless the user explicitly requests that action.

Leave modifications unstaged by default.

## Required validation

For normal code changes, run relevant focused tests first and then the full suite.

Before considering a release-affecting change complete, verify as appropriate:

- `python -m compileall -q dmsroute tests examples`
- `pytest -q`
- `python -m pip check`
- CLI smoke tests
- packaging/build checks when package structure or metadata changes

Do not require every expensive packaging check for trivial documentation-only changes.

# DMS Pipeline (ProteinGym + MaveDB)

A configuration-driven script built on `dms_parser` for MaveDB and ProteinGym
datasets.

## Usage

```bash
pip install -e .
pip install -r examples/yml_parser/requirements.txt
python examples/yml_parser/run_dms_parser.py \
    --config examples/yml_parser/config.yml
```

Options:
- `--only {all,proteingym,mavedb}` — restrict to one source (default: all)
- `--dry-run` — resolve metadata/WT sequences only, skip downloading scores and writing processed output
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

## Adding datasets

Edit `config.yml` — no code changes needed:
- ProteinGym: add a filename (must exist in `DMS_substitutions.csv`) under
  `proteingym.datasets`.
- MaveDB: add a URN under `mavedb.datasets`.

Each entry can carry its own `build_kwargs` block, deep-merged over
`default_build_kwargs`, to override settings (e.g. `delta`, `relative_method`,
or MaveDB's `hgvs_col`/`score_col`) for just that dataset.

## Output

- Processed CSVs land in `<dir_base>/processed/`, one per dataset, same as
  the notebooks.
- A combined run summary (CSV + JSON) is written to `datasets/summaries/`,
  covering both sources, with per-dataset status (`OK` / `ERROR` /
  `DRY_RUN`), row counts, and error messages for anything that failed —
  a single bad dataset never aborts the batch.
- Exit code is `1` if any dataset errored, `0` otherwise, so it plugs into
  CI/cron.

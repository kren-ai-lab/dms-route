# DMS Pipeline (ProteinGym + MaveDB)

A configuration-driven script built on `dms_parser` for MaveDB and ProteinGym
datasets.

The script is intentionally a thin command-line example. The installed
package's `dms_parser.config` module owns YAML loading and validation, while
`dms_parser.pipeline` owns source orchestration and output generation.
Nothing under `examples/` is required implementation for library callers.

## Usage

```bash
pip install -e .
pip install -r examples/yml_parser/requirements.txt
python examples/yml_parser/run_dms_parser.py \
    --config examples/yml_parser/config.yml
```

The same workflow can be called without `argparse`:

```python
from dms_parser import load_pipeline_config, run_pipeline

config = load_pipeline_config("examples/yml_parser/config.yml")
result = run_pipeline(config)
```

The final packaged `dms-parser` command is intentionally deferred to a later
increment.

Options:
- `--only {all,proteingym,mavedb}` — restrict to one source (default: all)
- `--dry-run` — resolve metadata/WT sequences only, skip downloading scores and writing processed output
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

## Adding datasets

Edit `config.yml` — no code changes needed:

- ProteinGym selects a registered resource and lists canonical `DMS_id` values
  as `dataset_id`. Only `dms_substitutions` is currently processable.
- MaveDB lists canonical score-set URNs as `dataset_id`.
- Include a source section to run it; omit the section to skip that source.

Each entry can carry its own `build_kwargs` block, deep-merged over
`default_build_kwargs`, to override builder settings such as `delta` or
`relative_method` for just that dataset. For MaveDB, `hgvs_col` and `score_col`
are direct keys on the dataset entry, alongside `dataset_id`, because the
runner uses them to select source-table columns before calling the builder.

Score transformations are disabled unless explicitly enabled. Set
`drop_failed: true` to save only rows with `status == "OK"`; with the default
`false`, unsupported and error rows remain in the processed table for
traceability.

## Output

- Processed CSVs land in `<dir_base>/processed/`, one per dataset, same as
  the notebooks.
- A combined run summary (CSV + JSON) is written to `datasets/summaries/`,
  covering both sources, with per-dataset status (`OK` / `ERROR` /
  `DRY_RUN`), row counts, and error messages for anything that failed —
  a single bad dataset never aborts the batch.
- Exit code is `1` if any dataset errored, `0` otherwise, so it plugs into
  CI/cron.

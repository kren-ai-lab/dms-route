# dms-parser

`dms-parser` is a Python package and command-line interface for obtaining and
standardizing published Deep Mutational Scanning (DMS) datasets from
ProteinGym and MaveDB.

The package parses substitution variants, reconstructs protein sequences, keeps
source scores in a common `score_raw` column, and records row-level parsing
outcomes. Score transformations and pseudo-binary labels are optional.

## Installation

`dms-parser` requires Python 3.10 or newer.

```bash
git clone https://github.com/kren-ai-lab/parsing_dms_data.git
cd parsing_dms_data
python -m pip install -e .
```

Confirm that the CLI is available:

```bash
dms-parser --help
```

For development, install the test dependency as well:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Inspect a MaveDB score set and produce a standardized dataset:

```bash
dms-parser metadata \
    --source mavedb \
    --dataset-id urn:mavedb:00000001-a-4

dms-parser download \
    --source mavedb \
    --dataset-id urn:mavedb:00000001-a-4 \
    --output-dir datasets/ube2i
```

A successful single download writes:

```text
datasets/ube2i/
|-- standardized.csv
|-- summary.csv
`-- summary.json
```

The equivalent ProteinGym workflow uses its canonical `DMS_id`:

```bash
dms-parser download \
    --source proteingym \
    --dataset-id BLAT_ECOLX_Jacquier_2013 \
    --output-dir datasets/blat-jacquier
```

ProteinGym obtains an assay from its shared substitutions resources. MaveDB
downloads one score set from the public API unless snapshot acquisition is
selected explicitly.

## Supported data

| Source | Dataset identifier | Standardized acquisition |
| --- | --- | --- |
| ProteinGym | Canonical `DMS_id` | Shared substitutions metadata and data resources |
| MaveDB | Permanent score-set URN | Public score-set API or a prefetched bulk snapshot |

The standardization workflow supports amino-acid substitutions, including
multi-mutants. ProteinGym-style variants such as `A23V` and `A23V;G45D` and
MaveDB protein HGVS variants such as `p.Met1Ala` and
`p.[Met1Ala;Lys2Arg]` are supported. MaveDB complete-target identity (`p.=`)
is treated as wild type. Equality-only bracketed expressions also represent
protein wild type, while equality components in mixed brackets add no mutation.

Insertions, deletions, duplications, frameshifts, extensions, and other
non-substitution variants are not standardized.

## Command-line workflows

### Find datasets

Use `list` to query a source catalog:

```bash
dms-parser list --source proteingym --query BRCA1 --limit 10
dms-parser list --source mavedb --query BRCA1 --limit 10
```

Use `metadata` when the dataset identifier is already known:

```bash
dms-parser metadata \
    --source proteingym \
    --dataset-id BRCA1_HUMAN_Findlay_2018
```

Both commands support text output and JSON output:

```bash
dms-parser list \
    --source proteingym \
    --variant-type substitutions \
    --format json \
    --output proteingym-catalog.json
```

MaveDB catalog commands query the active API. ProteinGym catalog commands use
reference files stored in the local cache. `--variant-type`, `--cache-dir`, and
`--refresh` apply only to the ProteinGym catalog.

### Download one dataset

`download` obtains and standardizes one dataset without a YAML configuration:

```bash
dms-parser download \
    --source mavedb \
    --dataset-id urn:mavedb:00000001-a-4 \
    --output-dir datasets/ube2i
```

Source data are cached under `~/.cache/dms-parser` by default. Use
`--cache-dir` to select another cache root, `--refresh` to reacquire source
artifacts, and `--overwrite` to replace an existing three-file output bundle.

Rows that cannot be standardized remain in the output with their `status` and
`error` values. Pass `--drop-failed` to retain only rows whose status is `OK`.

### Download several datasets

Repeat `--dataset-id` to process an ordered batch from one source:

```bash
dms-parser download-many \
    --source proteingym \
    --dataset-id BRCA1_HUMAN_Findlay_2018 \
    --dataset-id PTEN_HUMAN_Mighell_2018 \
    --output-dir datasets
```

Identifiers can also be read from a UTF-8 text file:

```text
urn:mavedb:00000001-a-4
urn:mavedb:00000665-a-1
urn:mavedb:00000080-a-2
```

```bash
dms-parser download-many \
    --source mavedb \
    --dataset-id-file mavedb_ids.txt \
    --output-dir datasets
```

The file contains one identifier per line. Blank lines are ignored; other
lines are treated as identifiers rather than comments or structured data. If
direct and file inputs are combined, direct identifiers are processed first.
Duplicate identifiers are rejected.

Each dataset is written to a deterministic subdirectory. The batch root also
contains `download-summary.csv` and `download-summary.json`:

```text
datasets/
|-- download-summary.csv
|-- download-summary.json
`-- mavedb/
    |-- id-<portable-dataset-id>/
    |   |-- standardized.csv
    |   |-- summary.csv
    |   `-- summary.json
    `-- id-<portable-dataset-id>/
        |-- standardized.csv
        |-- summary.csv
        `-- summary.json
```

Expected failures are recorded per dataset and do not stop the remaining
batch.

### Apply score transformations

Source scores are copied to `score_raw` without transformation. WT-relative
scores and pseudo-binary labels must be requested explicitly:

```bash
dms-parser download \
    --source proteingym \
    --dataset-id BLAT_ECOLX_Jacquier_2013 \
    --output-dir datasets/blat-difference \
    --add-relative-score \
    --relative-method difference \
    --relative-output-col score_difference
```

Supported relative methods are `ratio`, `log_ratio`, `log2_ratio`, and
`difference`. They require a valid WT score from the dataset or an explicit
`--wt-score` fallback.

Pseudo-binary labels are calculated from a requested relative score:

```bash
dms-parser download \
    --source proteingym \
    --dataset-id BLAT_ECOLX_Jacquier_2013 \
    --output-dir datasets/blat-labelled \
    --add-relative-score \
    --relative-method difference \
    --add-binary-label \
    --delta 0.1
```

Run `dms-parser download --help` or
`dms-parser download-many --help` for the complete set of WT, direction, column,
and output options.

### Use MaveDB bulk snapshots

MaveDB bulk snapshots are versioned Zenodo records. The archive can be about
1.9 GB, so snapshot acquisition is always explicit.

Fetch either the latest available snapshot or a fixed record:

```bash
dms-parser snapshot fetch --latest
dms-parser snapshot fetch --record 20840937
```

Search score-set metadata in a managed snapshot:

```bash
dms-parser discover --snapshot 20840937 --query BRCA1
```

An already extracted `main.json` can be searched directly:

```bash
dms-parser discover \
    --main-json /data/mavedb/main.json \
    --query BRCA1
```

Extract raw archival score and count tables for selected score sets:

```bash
dms-parser snapshot extract \
    --record 20840937 \
    --dataset-id urn:mavedb:00000003-a-1
```

To standardize a score set from a snapshot, first fetch the concrete record,
then select snapshot acquisition:

```bash
dms-parser download \
    --source mavedb \
    --dataset-id urn:mavedb:00000001-a-4 \
    --acquisition snapshot \
    --snapshot-record 20840937 \
    --output-dir datasets/ube2i-snapshot
```

`snapshot extract` returns raw archival tables. `download --acquisition
snapshot` passes the selected score table through the normal MaveDB
standardization workflow. Snapshot acquisition is available for `download` and
`download-many`, not for the YAML pipeline.

### Run a YAML pipeline

Use a configuration file to process named datasets from either or both sources:

```bash
dms-parser run --config examples/pipeline.yml
```

A minimal configuration has source-specific dataset lists and an optional
summary directory:

```yaml
proteingym:
  resource: dms_substitutions
  dir_base: datasets/proteingym
  default_build_kwargs:
    drop_failed: false
  datasets:
    - dataset_id: BLAT_ECOLX_Jacquier_2013

mavedb:
  dir_base: datasets/mavedb
  default_build_kwargs:
    drop_failed: false
  datasets:
    - dataset_id: "urn:mavedb:00000001-a-4"

output:
  summary_dir: datasets/summaries
```

Each dataset is processed independently. Standardized CSV files are written
under each source's `<dir_base>/processed/` directory, and combined timestamped
CSV and JSON summaries are written to `output.summary_dir`.

Select one configured source with `--only proteingym` or `--only mavedb`.
`--dry-run` validates the configuration and resolves source metadata without
downloading score tables or writing processed datasets.

[`examples/config.reference.yml`](examples/config.reference.yml) documents the
complete YAML contract and supported builder options.

### Inspect the cache

The cache inventory is read-only:

```bash
dms-parser cache
dms-parser cache --source proteingym
dms-parser cache --format json
dms-parser cache --cache-dir ./example-cache
```

It reports managed entries and structural problems without downloading,
repairing, or deleting artifacts.

### Exit codes and logging

The CLI uses these exit codes:

| Code | Meaning |
| --- | --- |
| `0` | The command completed successfully |
| `1` | An expected acquisition, processing, cache, or output failure occurred |
| `2` | Command usage was invalid |

Commands log at `INFO` by default. Use `--log-level DEBUG`, `WARNING`, or
`ERROR` to change the level.

## Standardized output

The builders preserve original source columns and add a common set of fields:

| Column | Description |
| --- | --- |
| `dataset_id` | Source dataset identifier |
| `source` | `proteingym` or `mavedb` |
| `protein_id` | Source-derived protein identifier, when available |
| `gene` | Source-derived gene name, when available |
| `uniprot_id` | Source-derived UniProt identifier, when available |
| `wt_sequence` | Wild-type protein sequence used for reconstruction |
| `variant` | Normalized internal substitution notation |
| `mutated_sequence` | Reconstructed protein sequence |
| `is_wildtype` | Whether the row represents wild type |
| `is_synthetic` | Whether the row was generated by the builder |
| `n_mutations` | Number of substitutions in the row |
| `score_raw` | Numeric copy of the selected source score |
| `status` | `OK`, `Error`, or `Unsupported` |
| `error` | Row-level parsing or reconstruction error |

Requested transformations add their configured output columns. ProteinGym
parsing also adds `parsed_*` fields, and source-specific columns remain in the
table.

The accompanying summary files record dataset-level row counts, output paths,
WT provenance, and requested transformations.

## Python API

### Build from a local table

```python
from dms_parser import build_proteingym_dataset

dataset = build_proteingym_dataset(
    input_path="experiment.csv",
    score_col="DMS_score",
    variant_col="mutant",
    wt_sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWQ",
    dataset_id="example-assay",
)
```

Use `build_mavedb_dataset` for a local MaveDB-like table with protein HGVS
variants.

### Download programmatically

```python
from pathlib import Path

from dms_parser import FilesystemCache, download_and_standardize_dataset

result = download_and_standardize_dataset(
    "mavedb",
    "urn:mavedb:00000001-a-4",
    output_dir="datasets/ube2i",
    cache=FilesystemCache(Path.home() / ".cache" / "dms-parser"),
)

print(result.dataset_path)
```

The package also exports catalog, parsing, transformation, validation,
snapshot, cache, and configuration APIs through `dms_parser`.

## Examples

The `examples/` directory contains:

- [`01_quickstart_proteingym.ipynb`](examples/01_quickstart_proteingym.ipynb):
  download and standardize a ProteinGym assay.
- [`02_quickstart_mavedb_download.ipynb`](examples/02_quickstart_mavedb_download.ipynb):
  obtain and standardize a MaveDB score set.
- [`03_variant_parsing_and_reconstruction.ipynb`](examples/03_variant_parsing_and_reconstruction.ipynb):
  parse substitutions and reconstruct sequences.
- [`04_transforms_and_pseudo_labels.ipynb`](examples/04_transforms_and_pseudo_labels.ipynb):
  apply score transformations and pseudo-binary labels.
- [`pipeline.yml`](examples/pipeline.yml): example configuration for
  `dms-parser run`.
- [`config.reference.yml`](examples/config.reference.yml): commented reference
  for the YAML configuration.

## Testing

```bash
python -m pytest
```

## License

This project is distributed under the GNU General Public License v3. See
[`LICENSE`](LICENSE).

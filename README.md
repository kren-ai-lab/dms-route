# 📦 **dms-parser**

> A lightweight and modular Python library for downloading, parsing, standardizing, and transforming Deep Mutational Scanning (DMS) datasets from ProteinGym and MaveDB.

`dms-parser` works with already-published DMS datasets; it does not run DMS
experiments or train predictive models.

---

## 🧠 Motivation

Deep Mutational Scanning (DMS) experiments generate large-scale measurements of variant effects across protein sequences. However, working with DMS data in practice presents several challenges:

* **Heterogeneous formats** (ProteinGym vs MaveDB vs custom datasets)
* **Inconsistent mutation encodings** (internal vs HGVS notation)
* **Missing or implicit wild-type references**
* **Non-standard score definitions**
* **Lack of reproducible preprocessing pipelines**

This library addresses these issues by providing a **unified, minimal, and extensible framework** for:

* Parsing mutation representations
* Reconstructing mutated protein sequences
* Standardizing datasets into a common schema
* Transforming scores relative to wild-type
* Generating ML-ready datasets

---

## 🎯 Scope

`dms-parser` is designed to:

* Support **substitution-based DMS datasets**
* Provide **robust parsing and validation**
* Enable **WT-relative transformations**
* Serve as a **data layer for downstream ML pipelines**

It is intentionally:

* ❌ not a modeling library
* ❌ not tied to specific benchmarks
* ❌ not dependent on deep learning frameworks

It is a **clean data + representation layer**.

---

## 🧩 Core Features

### ✅ Dataset ingestion

* ProteinGym substitution assays (CSV / parquet)
* MaveDB (API-based or explicitly prefetched bulk-snapshot acquisition)

### ✅ Variant parsing

* Internal notation: `A23V`, `M1A;K2R`
* HGVS protein notation: `p.Met1Ala`, `p.[Met1Ala;Lys2Arg]`
* MaveDB complete-target identity: `p.=`

### ✅ Sequence reconstruction

* Apply mutations to WT sequence
* Support single and multi-mutants
* Strict validation (position, residue consistency)

### ✅ Dataset standardization

* Unified schema across sources
* Consistent column naming
* Explicit mutation metadata

### ✅ Score transformations

* WT-relative ratio / log-ratio / log2-ratio / difference
* Z-score normalization
* Min-max scaling

### ✅ Pseudo-label generation

* Threshold-based classification with neutral region
* Configurable delta
* Supports both directions (higher/lower is better)

### ✅ Validation

* Sequence validation
* WT presence checks
* Dataset integrity checks
* Error tracking (`status`, `error`)

---

## 📦 Installation

```bash
git clone https://github.com/kren-ai-lab/parsing_dms_data.git
cd parsing_dms_data
python -m pip install -e .
```

The normal installation includes YAML configuration and ProteinGym Parquet
support.

Inspect the installed interface:

```bash
dms-parser --help
dms-parser run --help
dms-parser list --help
dms-parser metadata --help
dms-parser download --help
dms-parser download-many --help
dms-parser snapshot --help
dms-parser snapshot fetch --help
dms-parser snapshot extract --help
dms-parser discover --help
```

The CLI logs at `INFO` by default; pass `--log-level DEBUG` for diagnostic
output.

Use `list` to discover dataset identifiers and `metadata` to inspect one
normalized `DatasetRecord`:

```bash
dms-parser list --source proteingym --query BRCA1 --limit 10

dms-parser metadata \
    --source mavedb \
    --dataset-id urn:mavedb:00000097-a-1

dms-parser list \
    --source proteingym \
    --variant-type substitutions \
    --format json \
    --output proteingym-catalog.json
```

These commands do not download score tables or run the processing pipeline.
ProteinGym may download and cache lightweight reference CSV files under
`~/.cache/dms-parser` by default; `--cache-dir` changes that location and
`--refresh` refreshes those references. MaveDB instead queries its public
metadata API directly and does not accept the cache, refresh, or variant-type
options. Query semantics differ between the two sources, and `--variant-type`
applies only to ProteinGym. Use `--format json` for machine-readable output.
`--output` writes UTF-8 data and overwrites its target.

### Managed MaveDB bulk snapshots

MaveDB publishes versioned bulk snapshots under the stable Zenodo concept DOI
`10.5281/zenodo.11201736`. Resolve the newest concrete version when following
current releases, or pin a concrete record ID for reproducible work:

```bash
dms-parser snapshot fetch --latest

dms-parser snapshot fetch \
    --record 20840937 \
    --format json
```

Snapshots are stored under
`~/.cache/dms-parser/mavedb/snapshots/<record-id>/` by default;
`--cache-dir` changes the cache root and `--refresh` reacquires the selected
concrete version. A pinned valid version is reused without contacting Zenodo,
while `--latest` resolves the current concrete record before checking its
versioned cache entry.

The bulk archive can be approximately 1.9 GB. Downloads are streamed and
published only after their exact Zenodo size and checksum are verified. The
manager safely extracts only the regular `main.json` member from supported
TAR.GZ or ZIP archives; score tables are not extracted. Neither the archive nor
`main.json` is stored in this repository.

### Discover datasets in a MaveDB bulk snapshot

`discover` searches a reproducible, already-extracted bulk `main.json`, not the
live MaveDB API. Use an exact local file without network or snapshot-manager
access:

```bash
dms-parser discover \
    --main-json /data/mavedb/main.json \
    --query BRCA1
```

Or let the existing snapshot manager resolve the latest version or a pinned
Zenodo record:

```bash
dms-parser discover --snapshot latest --query BRCA1

dms-parser discover \
    --snapshot 20840937 \
    --query BRCA1
```

An uncached managed snapshot may download an approximately 1.9 GB archive.
Managed discovery uses the snapshot cache described above; `--cache-dir`
changes its root and `--refresh` reacquires the selected snapshot. These two
options are intentionally unavailable with `--main-json`.

Text output groups only matching score sets under their experiments and exposes
the score-set URNs needed by other commands:

```text
query: BRCA1
snapshot_title: MaveDB public data dump
as_of: 2026-06-24T18:13:01Z
source: managed snapshot
record_id: 20840937
experiment_count: 1
score_set_count: 1
experiment: urn:mavedb:00000097-a
  experiment_set: urn:mavedb:00000097
  title: BRCA1 saturation editing
  score_set: urn:mavedb:00000097-a-1
    title: BRCA1 function scores
    targets: BRCA1
    n_variants: 3893
    status: current
```

Use deterministic JSON on standard output or write it as UTF-8 to a file:

```bash
dms-parser discover \
    --snapshot 20840937 \
    --query BRCA1 \
    --format json \
    --output brca1-score-sets.json
```

The Python API loads, validates, and indexes the complete file once. Reuse one
catalog for repeated searches without reopening or reparsing `main.json`:

```python
from dms_parser import MaveDBBulkCatalog

catalog = MaveDBBulkCatalog.from_file("/data/mavedb/main.json")
brca1 = catalog.search_by_gene("BRCA1")
tp53 = catalog.search_by_gene("TP53")

for experiment in brca1.experiments:
    for score_set in experiment.score_sets:
        print(score_set.dataset_id)
```

Score sets listed by an experiment's `scoreSetUrns` are current. Older nested
score sets absent from that list are superseded, excluded by default, and
marked explicitly when requested:

```bash
dms-parser discover \
    --main-json /data/mavedb/main.json \
    --query BRCA1 \
    --include-superseded
```

Discovery only searches metadata. It does not download score tables, choose or
merge alternatives, or make compatibility decisions. Pass a returned URN
manually to the existing single- or multi-dataset download interface when
appropriate:

```bash
dms-parser download \
    --source mavedb \
    --dataset-id urn:mavedb:00000097-a-1 \
    --output-dir datasets/mavedb-00000097-a-1
```

### Extract raw tables from a managed MaveDB snapshot

`snapshot extract` copies the raw archival scores CSV, plus the counts CSV when
available, for explicit score-set URNs. It never searches or selects a score
set. Resolve the latest snapshot or pin a concrete Zenodo record for
reproducibility:

```bash
dms-parser snapshot extract \
    --latest \
    --dataset-id urn:mavedb:00000003-a-1

dms-parser snapshot extract \
    --record 20840937 \
    --dataset-id urn:mavedb:00000003-a-1 \
    --dataset-id urn:mavedb:00000003-a-2
```

An uncached selector may download the approximately 1.9 GB managed archive.
Extracted tables use a separate cache under
`<cache-root>/mavedb/snapshot_tables/<record-id>/`. Valid bundles are reused
without reopening the archive. `--refresh` refreshes both the managed snapshot
and every requested extraction bundle. TAR.GZ extraction may scan the archive
metadata even though only explicitly requested files are written.

Official members are derived only from the validated score-set URN by replacing
colons with hyphens:

```text
urn:mavedb:00000003-a-1
csv/urn-mavedb-00000003-a-1.scores.csv
csv/urn-mavedb-00000003-a-1.counts.csv
```

The scores member is required. Counts are optional and are extracted
automatically when present. Superseded score sets are rejected unless requested
explicitly:

```bash
dms-parser snapshot extract \
    --record 20840937 \
    --dataset-id urn:mavedb:00000003-a-1 \
    --include-superseded
```

Text output reports concrete snapshot provenance, current/superseded status,
final file paths, counts availability, and whether each bundle was extracted or
reused:

```text
record_id: 20840937
doi: 10.5281/zenodo.20840937
archive_filename: mavedb-dump.2026062418131.tar.gz
dataset_count: 1
dataset_id: urn:mavedb:00000003-a-1
  status: current
  scores_path: .../scores.csv
  counts_path: .../counts.csv
  cache: extracted
```

Use JSON on standard output or publish it atomically to a UTF-8 file:

```bash
dms-parser snapshot extract \
    --record 20840937 \
    --dataset-id urn:mavedb:00000003-a-1 \
    --format json \
    --output extracted-tables.json
```

The Python API deliberately separates snapshot resolution from extraction:

```python
from dms_parser import (
    FilesystemCache,
    extract_mavedb_snapshot_tables,
    fetch_mavedb_snapshot,
)

cache = FilesystemCache("/data/dms-parser-cache")
snapshot = fetch_mavedb_snapshot("20840937", cache=cache)
result = extract_mavedb_snapshot_tables(
    snapshot,
    ["urn:mavedb:00000003-a-1"],
    cache=cache,
)

print(result.tables[0].scores_path)
print(result.tables[0].counts_path)
```

These files are raw archival CSVs. Extraction does not parse their columns,
standardize variants, transform scores, choose compatible alternatives, or
merge datasets. It also does not use the live MaveDB API or place raw tables in
the standardized download cache.

### Download and standardize one dataset

The `download` command fetches and standardizes one substitutions dataset
without requiring `pipeline.yml`:

```bash
dms-parser download \
    --source proteingym \
    --dataset-id BLAT_ECOLX_Jacquier_2013 \
    --output-dir datasets/blat-jacquier

dms-parser download \
    --source mavedb \
    --dataset-id urn:mavedb:00000097-a-1 \
    --output-dir datasets/mavedb-00000097-a-1
```

Standardized downloads support three acquisition models:

| Source | Acquisition |
| --- | --- |
| ProteinGym | Shared bulk CSV/Parquet resources, then filter by `DMS_id` |
| MaveDB default | Individual API acquisition by score-set URN |
| MaveDB snapshot | Selected score tables from one explicitly prefetched, versioned snapshot |

API mode remains the default and is preferable for one or a few MaveDB URNs.
Snapshot mode is useful for larger batches, offline processing, and
release-level reproducibility. First fetch the concrete snapshot explicitly;
this command may download approximately 1.9 GB:

```bash
dms-parser snapshot fetch --record 20840937
```

Then standardize a selected score set entirely from that cached snapshot:

```bash
dms-parser download \
    --source mavedb \
    --dataset-id urn:mavedb:00000097-a-1 \
    --acquisition snapshot \
    --snapshot-record 20840937 \
    --output-dir datasets/mavedb-00000097-a-1
```

`download` never resolves or downloads the large snapshot implicitly. If the
concrete record is absent or invalid in the cache, it reports the exact
`snapshot fetch --record` command required. In snapshot mode, `--refresh`
re-extracts selected raw tables from the already cached immutable archive; it
does not contact Zenodo or reacquire the archive.

`snapshot extract` returns raw archival tables. In contrast,
`download --acquisition snapshot` feeds those cached score tables through the
normal MaveDB builder and publishes the standard three-file dataset bundle.
Counts tables remain optional raw data and are not used in standardization.
ProteinGym is already bulk-first and needs no snapshot-build equivalent.

Each successful command writes exactly this bundle:

```text
<output-dir>/
├── standardized.csv
├── summary.csv
└── summary.json
```

Original source artifacts remain in `~/.cache/dms-parser` by default.
`--cache-dir` changes the cache root, and `--refresh` reacquires cached source
artifacts. `--drop-failed` removes row-level parse failures;
`--add-wildtype-row` prepends one scoreless WT row only when no valid WT row is
present. `--overwrite` replaces only the three deterministic files above and
preserves unrelated files in the output directory.

No score transformations are applied. Raw source columns are retained, so
"standardized" means that common fields are guaranteed, not that the output has
an exclusive fixed schema. ProteinGym acquisition may download its complete
substitutions benchmark before selecting one assay. Indels are not supported,
and MaveDB datasets with nonstandard score/HGVS columns or no recoverable WT
sequence may remain unsupported.

### Download and standardize several datasets

`download-many` processes several substitutions datasets from one source in
request order, without `pipeline.yml`. Repeat the singular `--dataset-id`
option once per dataset:

```bash
dms-parser download-many \
    --source proteingym \
    --dataset-id BRCA1_HUMAN_Findlay_2018 \
    --dataset-id PTEN_HUMAN_Mighell_2018 \
    --output-dir datasets

dms-parser download-many \
    --source mavedb \
    --dataset-id urn:mavedb:00000097-a-1 \
    --dataset-id urn:mavedb:00000100-a-1 \
    --output-dir datasets
```

Use one prefetched MaveDB snapshot for a reproducible batch by repeating the
existing singular `--dataset-id` option:

```bash
dms-parser download-many \
    --source mavedb \
    --dataset-id urn:mavedb:00000097-a-1 \
    --dataset-id urn:mavedb:00000100-a-1 \
    --acquisition snapshot \
    --snapshot-record 20840937 \
    --output-dir datasets
```

The cached snapshot and its selected tables are shared across the batch while
each standardized bundle is built and published independently. Superseded
score sets require `--include-superseded`.

Each ID receives a portable deterministic directory beneath the source. The
directory name starts with `id-`, contains a sanitized form of the original
ID, and ends with a short SHA-256 digest. The original ID remains in every
aggregate record.

```text
<output-dir>/
|-- download-summary.csv
|-- download-summary.json
`-- <source>/
    |-- <portable-id-1>/
    |   |-- standardized.csv
    |   |-- summary.csv
    |   `-- summary.json
    `-- <portable-id-2>/
        |-- standardized.csv
        |-- summary.csv
        `-- summary.json
```

Expected per-dataset failures are recorded as `ERROR` and processing continues;
successful records use `SUCCESS`. The aggregate CSV and JSON preserve request
order and contain the source, original ID, relative output paths, successful
row counts, and the expected error type and message when processing fails.
Their ordered fields are `source`, `dataset_id`, `status`, `output_dir`,
`dataset_path`, `summary_csv_path`, `summary_json_path`, `target_protein`,
`wt_length`, `raw_rows`, `validated_rows`, `discarded_rows`, `output_rows`,
`wildtype_rows`, `synthetic_wildtype_rows`, `error_type`, and `error`.
Exit code `0` means every dataset succeeded, `1` means an expected operational
failure occurred, and `2` indicates invalid command usage.

The default cache is `~/.cache/dms-parser`; `--cache-dir` changes it.
`--refresh` reacquires each MaveDB score artifact and reacquires the shared
ProteinGym substitutions reference and Parquet once per batch. `--overwrite`
replaces only the three deterministic files in each rebuilt dataset bundle and
the two aggregate summary files. Unrelated files and old bundles whose rebuild
fails before publication are preserved, while successful bundles remain after
another requested dataset fails.

As with single download, no score transformations are applied, original source
columns are retained, and only substitutions are supported. ProteinGym may
load the complete substitutions benchmark once for the batch. Existing MaveDB
WT and score/HGVS-column limitations still apply.

---

## ⚙️ Requirements

* Python ≥ 3.10
* pandas
* numpy
* requests
* pyarrow (installed for ProteinGym Parquet support)
* PyYAML (installed for YAML configuration loading)

---

## 🚀 Quickstart

### ProteinGym example

```python
from dms_parser import build_proteingym_dataset

df = build_proteingym_dataset(
    input_path="experiment.csv",
    variant_col="mutant",
    score_col="DMS_score",
    wt_sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWQ",
    add_relative_score=True,
    add_binary_label=True,
)

df.head()
```

---

### MaveDB example

```python
from dms_parser import build_mavedb_dataset

df = build_mavedb_dataset(
    input_path="scores.csv",
    hgvs_col="hgvs_pro",
    score_col="score",
    wt_sequence=wt_sequence,
)

df.head()
```

---

### Pipeline configuration

Run every source in the example configuration:

```bash
dms-parser run --config examples/pipeline.yml
```

Use `--only proteingym` or `--only mavedb` to select one source. Use
`--dry-run` to validate the configuration and resolve source metadata without
downloading score tables or writing processed datasets.

The YAML example selects the logical ProteinGym resource owned by the library
instead of repeating official URLs:

```yaml
proteingym:
  resource: dms_substitutions
  dir_base: datasets/proteingym
  datasets:
    - dataset_id: BLAT_ECOLX_Jacquier_2013
      build_kwargs:
        drop_failed: true

mavedb:
  dir_base: datasets/mavedb
  datasets:
    - dataset_id: "urn:mavedb:00000001-a-4"
```

Each source section can list multiple `dataset_id` entries, which are processed
independently in one pipeline run.

ProteinGym registers `dms_substitutions`, `dms_indels`,
`clinical_substitutions`, and `clinical_indels`. Only `dms_substitutions` is
currently processable; selecting another registered resource for processing
raises a clear error before downloading. ProteinGym uses its canonical
`DMS_id` as `dataset_id`. MaveDB uses the score-set URN.

Source-level `default_build_kwargs` apply to every dataset. A dataset's own
`build_kwargs` are deep-merged over those defaults. Score transformations are
opt-in. `drop_failed: false` retains unsupported and error rows for
traceability; `true` saves only rows whose status is `OK`.

Both builders and YAML `build_kwargs` accept the opt-in
`add_wildtype_row: true` setting. It prepends one WT sequence row only when a
valid WT row is missing. The generated row has a missing `score_raw` because no
experimental score is inferred, and `is_synthetic` distinguishes it from
published observations. WT-relative transformations still require a numeric
observed WT score.

Processed CSV files are written under each source's
`<dir_base>/processed/` directory. The configured `output.summary_dir`
receives combined CSV and JSON summaries. One failed dataset does not abort
the remaining batch, and any `ERROR` summary row produces exit code `1`.

Configuration loading and pipeline orchestration are also available directly
from the installed package:

```python
from dms_parser import load_pipeline_config, run_pipeline

config = load_pipeline_config("examples/pipeline.yml")
result = run_pipeline(config)

print(result.summary)
raise SystemExit(result.exit_code)
```

`config.py` owns YAML loading and structural validation. `pipeline.py` owns
configuration-driven ProteinGym and MaveDB execution and combined summaries.
`downloads.py` owns independent single- and multi-dataset standardized
downloads. The installed CLI commands are adapters over these public APIs.

---

## 🧬 Standardized Dataset Schema

All datasets are transformed into a common structure:

| Column             | Description                                   |
| ------------------ | --------------------------------------------- |
| `variant`          | Internal mutation notation                    |
| `mutated_sequence` | Reconstructed protein sequence                |
| `score_raw`        | Original score                                |
| `is_wildtype`      | WT flag                                       |
| `is_synthetic`     | Generated-row flag                            |
| `n_mutations`      | Number of substitutions                       |
| `status`           | Parsing status (`OK`, `Error`, `Unsupported`) |
| `error`            | Error message (if any)                        |

Optional:

* `score_log_ratio`
* `score_binary_like`
* `score_zscore`
* `score_minmax`

---

## 🔬 Transformations

Numerical score transformations are opt-in. When transformation arguments are
omitted, both dataset builders preserve source values in `score_raw`. Request
WT-relative scores, pseudo-binary labels, z-scores, or min-max scaling
explicitly when they are needed.

### WT-relative scoring

```python
from dms_parser import add_wt_relative_score

df = add_wt_relative_score(
    df,
    score_col="score_raw",
    method="log_ratio",
)
```

### Pseudo-binary labels

```python
from dms_parser import add_pseudo_binary_label

df = add_pseudo_binary_label(
    df,
    score_col="score_log_ratio",
    delta=0.15,
)
```

---

## 🧠 Architecture

The library is organized into modular components:

```text
dms_parser/
├── builders.py        # High-level dataset construction
├── cache.py           # Validated filesystem artifact cache
├── catalog.py         # Common metadata records and source dispatch
├── cli.py             # Installed command-line interface
├── config.py          # Pipeline YAML loading and validation
├── downloads.py       # Standardized single and batch downloads
├── fetch.py           # Cache-aware staged downloads
├── pipeline.py        # Configuration-driven source orchestration
├── parsing.py         # Variant parsing logic
├── transforms.py      # Score transformations
├── validation.py      # Dataset and sequence validation
├── io.py              # File I/O and downloads
├── constants.py       # Shared constants
├── exceptions.py      # Custom exceptions
├── types.py           # Type definitions
└── sources/
    ├── mavedb.py
    ├── mavedb_catalog.py
    ├── proteingym.py
    ├── proteingym_catalog.py
    └── proteingym_resources.py
```

---

## 🔁 Design Principles

* **Modularity**: each component is independent
* **Explicitness**: no hidden assumptions
* **Robustness**: strict validation and error tracking
* **Reproducibility**: deterministic transformations
* **Extensibility**: easy to add new sources or transformations

---

## ⚠️ Limitations

* Only supports **substitution mutations**
* HGVS support excludes:

  * insertions
  * deletions
  * frameshifts
  * duplications
* WT sequence must be provided or recoverable
* Score interpretation depends on the dataset

---

## 📓 Examples

See the `examples/` directory:

```text
examples/01_quickstart_proteingym.ipynb
examples/02_quickstart_mavedb_download.ipynb
examples/03_variant_parsing_and_reconstruction.ipynb
examples/04_transforms_and_pseudo_labels.ipynb
examples/pipeline.yml
```

---

## 🧪 Testing

Run the full test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

---

## 🔮 Roadmap

* [ ] Support for indels (partial HGVS)
* [ ] Automatic WT extraction improvements
* [ ] Integration with representation libraries (e.g., Sylphy)
* [ ] Dataset versioning utilities
* [x] Configuration-driven `run` command

---

## 🤝 Integration in a larger ecosystem

This library is designed to integrate with:

* **Representation layers** (e.g., protein embeddings)
* **Clustering frameworks**
* **Low-N ML pipelines**
* **Protein engineering workflows**

---

## 📄 License

GNU General Public License V3 License

---

## 👤 Author

KrenAI Lab
Computational Protein Engineering & Machine Learning

---

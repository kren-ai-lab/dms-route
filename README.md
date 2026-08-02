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
* MaveDB (API-based download)

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
ProteinGym and MaveDB orchestration, dataset output, and combined summaries.
The installed `dms-parser run` command is a command-line adapter over these
public APIs.

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

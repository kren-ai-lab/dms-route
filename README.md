# 📦 **dms-parser**

> A lightweight and modular Python library for downloading, parsing, standardizing, and transforming Deep Mutational Scanning (DMS) datasets from ProteinGym and MaveDB.

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

* ProteinGym (CSV / parquet)
* MaveDB (API-based download)

### ✅ Variant parsing

* Internal notation: `A23V`, `M1A;K2R`
* HGVS protein notation: `p.Met1Ala`, `p.[Met1Ala;Lys2Arg]`

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

### Library installation

```bash
git clone https://github.com/kren-ai-lab/parsing_dms_data.git
cd parsing_dms_data
pip install -e .
```

### Packaged command

Install the package normally. YAML configuration support and ProteinGym
Parquet support are included as runtime dependencies:

```bash
pip install -e .
```

Inspect the installed interface:

```bash
dms-parser --help
dms-parser run --help
```

Run both configured sources:

```bash
dms-parser run --config examples/yml_parser/config.yml
```

Restrict execution to one source:

```bash
dms-parser run \
    --config examples/yml_parser/config.yml \
    --only proteingym
```

Resolve metadata without downloading or processing score datasets:

```bash
dms-parser run \
    --config examples/yml_parser/config.yml \
    --dry-run
```

The command returns `0` when the pipeline completes without an `ERROR`
summary row, `1` for a configuration error or a pipeline result containing an
`ERROR`, and argparse's standard `2` for invalid command-line usage.

### Test installation

```bash
pip install -e ".[dev]"
python -m pytest
```

### YAML example runner

The installed `dms-parser run` command is the primary CLI. Installing the
project supplies the YAML and Parquet dependencies used by the pipeline; the
example wrapper does not require a separate dependency installation:

```bash
pip install -e .
python examples/yml_parser/run_dms_parser.py \
    --config examples/yml_parser/config.yml
```

`run_dms_parser.py` is only a compatibility/example wrapper around the
installed command. YAML loading and ProteinGym Parquet handling occur in the
package's pipeline, not in the wrapper.

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

### Cached downloads

`FilesystemCache` stores artifacts by source and dataset identifier with a
manifest containing the original URL, download time, file size, and SHA-256
checksum. `fetch_to_cache` returns valid cache hits without a network request
and uses atomic downloading and publication for misses or refreshes.

```python
from dms_parser import FilesystemCache, fetch_to_cache

cache = FilesystemCache("datasets/cache")
path = fetch_to_cache(
    "https://example.org/experiment.csv",
    source="proteingym",
    dataset_id="experiment-1",
    cache=cache,
)
```

Pass `refresh=True` to retrieve and safely publish a new copy.

---

### Logging

Importing `dms_parser` does not configure application logging. Applications
can enable library lifecycle messages with the standard library:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

To enable diagnostic output for only this package:

```python
logging.getLogger("dms_parser").setLevel(logging.DEBUG)
```

Logging configuration, handlers, and output destinations remain the
responsibility of the consuming application or CLI.

---

### Dataset catalog

Catalog operations return a common `DatasetRecord` dataclass with `source`,
`dataset_id`, optional `title`, `target_id`, `variant_type`, and `n_variants`
fields, plus the complete source object or CSV row in `raw_metadata`.

List public MaveDB score sets:

```python
from dms_parser import list_datasets

records = list_datasets("mavedb", query="BRCA1", limit=10)
```

List ProteinGym substitution assays using the lightweight reference-file cache:

```python
from dms_parser import FilesystemCache, list_datasets

cache = FilesystemCache("datasets/cache")
records = list_datasets(
    "proteingym",
    variant_type="substitutions",
    cache=cache,
    limit=10,
)
```

Retrieve one metadata record by source identifier:

```python
from dms_parser import get_dataset_metadata

record = get_dataset_metadata(
    "mavedb",
    "urn:mavedb:00000001-a-1",
)
```

Catalog operations retrieve metadata only. They never download score tables,
benchmark archives, raw assays, alignments, or model predictions. Listing
`source="all"` is intentionally unsupported because cross-source pagination
would be ambiguous; call each source separately.

---

### Source configuration

The YAML example selects the logical ProteinGym resource owned by the library
instead of repeating official URLs:

```yaml
proteingym:
  resource: dms_substitutions
  dir_base: datasets/proteingym
  datasets:
    - dataset_id: BLAT_ECOLX_Jacquier_2013

mavedb:
  dir_base: datasets/mavedb
  datasets:
    - dataset_id: "urn:mavedb:00000001-a-4"
```

ProteinGym registers `dms_substitutions`, `dms_indels`,
`clinical_substitutions`, and `clinical_indels`. Only `dms_substitutions` is
currently processable; selecting another registered resource for processing
raises a clear error before downloading. ProteinGym uses its canonical
`DMS_id` as `dataset_id`. MaveDB uses the score-set URN.

Gene or target searches are discovery operations and may return multiple
MaveDB score sets. They are not reproducible download identities. Advanced
Python callers may still override catalog or download endpoints for mirrors
and tests where those APIs support overrides.

Score transformations remain opt-in. The YAML runner's `drop_failed` option
controls traceability: `false` retains unsupported and error rows, while
`true` saves only rows whose status is `OK`.

Configuration loading and pipeline orchestration are also available directly
from the installed package:

```python
from dms_parser import load_pipeline_config, run_pipeline

config = load_pipeline_config("examples/yml_parser/config.yml")
result = run_pipeline(config)

print(result.summary)
raise SystemExit(result.exit_code)
```

`config.py` owns YAML loading and structural validation. `pipeline.py` owns
ProteinGym and MaveDB orchestration, dataset output, and combined summaries.
The installed `dms-parser run` command is a command-line adapter over these
public APIs. The script under `examples/yml_parser/` remains a compatibility
wrapper; essential implementation does not live under `examples/`.

---

## 🧬 Standardized Dataset Schema

All datasets are transformed into a common structure:

| Column             | Description                                   |
| ------------------ | --------------------------------------------- |
| `variant`          | Internal mutation notation                    |
| `mutated_sequence` | Reconstructed protein sequence                |
| `score_raw`        | Original score                                |
| `is_wildtype`      | WT flag                                       |
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
examples/yml_parser/run_dms_parser.py
```

---

## 🧪 Testing

Run the full test suite:

```bash
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

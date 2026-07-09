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

### From source

```bash
git clone https://github.com/kren-ai-lab/parsing_dms_data.git
cd parsing_dms_data
pip install -e .
```

### Minimal dependencies

```bash
pip install pandas numpy requests pyarrow
```

---

## ⚙️ Requirements

* Python ≥ 3.10
* pandas
* numpy
* requests
* pyarrow (for parquet support)

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
├── parsing.py         # Variant parsing logic
├── transforms.py      # Score transformations
├── validation.py      # Dataset and sequence validation
├── io.py              # File I/O and downloads
├── constants.py       # Shared constants
├── exceptions.py      # Custom exceptions
├── types.py           # Type definitions
└── sources/
    ├── mavedb.py
    └── proteingym.py
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

See the `notebooks/` directory:

```text
01_quickstart_proteingym_download.ipynb
02_quickstart_mavedb_download.ipynb
03_variant_parsing_and_reconstruction.ipynb
04_transforms_and_pseudo_labels.ipynb
```

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -q
```

---

## 🔮 Roadmap

* [ ] Support for indels (partial HGVS)
* [ ] Automatic WT extraction improvements
* [ ] Integration with representation libraries (e.g., Sylphy)
* [ ] Dataset versioning utilities
* [ ] CLI interface

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
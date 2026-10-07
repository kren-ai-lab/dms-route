# DMSRoute

DMSRoute obtains and standardizes Deep Mutational Scanning (DMS) datasets
from ProteinGym and MaveDB. It provides a command-line interface, a Python API,
and a YAML pipeline for producing consistent datasets from source-specific
tables and metadata.

The project name is DMSRoute. The distribution, Python package, and console
command are named `dmsroute`.

Supported sources:

- ProteinGym substitution and indel datasets, identified by canonical
  `DMS_id` values.
- MaveDB score sets, identified by permanent score-set URNs and acquired from
  the public API or a fixed bulk snapshot.

## Installation

DMSRoute requires Python 3.10 or newer.

```bash
git clone https://github.com/kren-ai-lab/dms-route.git
cd dms-route
python -m pip install -e .
```

Check the installation with:

```bash
dmsroute --help
```

Development setup and validation commands are documented in
[`DEVELOPMENT.md`](DEVELOPMENT.md).

## Quick start

Inspect a MaveDB score set:

```bash
dmsroute metadata \
  --source mavedb \
  --dataset-id urn:mavedb:00000001-a-4
```

Download and standardize it:

```bash
dmsroute download \
  --source mavedb \
  --dataset-id urn:mavedb:00000001-a-4 \
  --output-dir datasets/ube2i
```

ProteinGym uses canonical `DMS_id` values:

```bash
dmsroute list --source proteingym --query BRCA1 --limit 10

dmsroute download \
  --source proteingym \
  --dataset-id BRCA1_HUMAN_Findlay_2018 \
  --output-dir datasets/brca1
```

ProteinGym substitutions are selected by default. Use `--variant-type indels`
for the indel resource.

## CLI

The main commands are:

| Command | Purpose |
| --- | --- |
| `dmsroute list` | Search a ProteinGym or MaveDB catalog |
| `dmsroute metadata` | Retrieve metadata for one dataset |
| `dmsroute download` | Download and standardize one dataset |
| `dmsroute download-many` | Process several datasets from one source |
| `dmsroute run` | Run a YAML pipeline |
| `dmsroute discover` | Search a fixed local MaveDB snapshot |
| `dmsroute snapshot` | Fetch snapshots or extract archival tables |
| `dmsroute cache` | Inspect the local cache without modifying it |

Each command provides its complete options through `--help`.

For example, a batch can be specified with repeated identifiers:

```bash
dmsroute download-many \
  --source proteingym \
  --dataset-id BRCA1_HUMAN_Findlay_2018 \
  --dataset-id PTEN_HUMAN_Mighell_2018 \
  --output-dir datasets
```

Source artifacts are cached under `~/.cache/dmsroute` by default. Use
`--cache-dir` to select another location.

## MaveDB acquisition

### API

API acquisition is the default for MaveDB catalog, metadata, and download
commands. It retrieves the current score-set metadata and scores for a
permanent URN.

```bash
dmsroute download \
  --source mavedb \
  --dataset-id urn:mavedb:00000001-a-4 \
  --output-dir datasets/ube2i
```

### Snapshot

Bulk snapshots are versioned Zenodo records. Snapshot acquisition is explicit
and requires the selected record to be present in the local cache.

```bash
dmsroute snapshot fetch --record 20840937

dmsroute discover \
  --snapshot 20840937 \
  --query BRCA1

dmsroute download \
  --source mavedb \
  --dataset-id urn:mavedb:00000001-a-4 \
  --acquisition snapshot \
  --snapshot-record 20840937 \
  --output-dir datasets/ube2i-snapshot
```

`dmsroute snapshot extract` extracts raw score and count tables. Snapshot
standardization is available through `download` and `download-many`; the YAML
pipeline uses the MaveDB API.

## Configuration

Run a configured pipeline with:

```bash
dmsroute run --config examples/pipeline.yml
```

A configuration may contain ProteinGym, MaveDB, or both:

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

output:
  summary_dir: datasets/summaries
```

Use `--only proteingym` or `--only mavedb` to select one configured source.
`--dry-run` resolves configuration and source metadata without writing
processed datasets. The full configuration contract is documented in
[`examples/config.reference.yml`](examples/config.reference.yml).

## Output

A successful `download` writes one dataset bundle:

```text
output-directory/
|-- standardized.csv
|-- summary.csv
`-- summary.json
```

The standardized table preserves source columns and adds common dataset,
variant, sequence, score, and row-status fields. Source scores are copied to
`score_raw`; score transformations are opt-in.

`download-many` also writes `download-summary.csv` and
`download-summary.json` at the batch root. YAML pipelines write processed CSV
files under each source directory and combined summaries under
`output.summary_dir`.

## Python API

The same download workflow is available from Python:

```python
from pathlib import Path

from dmsroute import FilesystemCache, download_and_standardize_dataset

cache = FilesystemCache(Path.home() / ".cache" / "dmsroute")
result = download_and_standardize_dataset(
    "mavedb",
    "urn:mavedb:00000001-a-4",
    output_dir="datasets/ube2i",
    cache=cache,
)

print(result.dataset_path)
```

Local tables can be standardized with `build_proteingym_dataset`,
`build_proteingym_indel_dataset`, and `build_mavedb_dataset`.

## Documentation

- [`examples/`](examples/) contains notebooks for acquisition, parsing, and
  score transformations.
- [`examples/config.reference.yml`](examples/config.reference.yml) documents
  every YAML option.
- [`DEVELOPMENT.md`](DEVELOPMENT.md) covers development setup, architecture,
  tests, and packaging checks.
- `dmsroute <command> --help` is the reference for CLI options.

## Citation

If you use DMSRoute in your research, please cite:

> Fernández-Villegas, D.; Escobedo, S.; Alarcón, T.; Medina-Ortiz, D.
> *DMSRoute*.
> Version 0.1.1, 2026. Zenodo.
> https://doi.org/10.5281/zenodo.21924522

```bibtex
@software{dmsroute2026,
  author    = {Fernández-Villegas, Diego and Escobedo, Sebastián and Alarcón, Tomás and Medina-Ortiz, David},
  title     = {DMSRoute},
  year      = {2026},
  version   = {0.1.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21924522},
  url       = {https://doi.org/10.5281/zenodo.21924522}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

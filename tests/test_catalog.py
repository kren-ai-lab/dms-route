from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

import dms_parser.acquisition.fetch as fetch_module
from dms_parser import (
    DatasetRecord,
    FilesystemCache,
    MaveDBCatalog,
    ProteinGymCatalog,
    get_dataset_metadata,
    list_datasets,
)
from dms_parser.catalog import normalize_raw_metadata
from dms_parser.core.exceptions import (
    CatalogError,
    DatasetNotFoundError,
    InvalidCatalogQueryError,
)
from dms_parser.sources.proteingym_catalog import (
    PROTEINGYM_INDELS_CACHE_ID,
    PROTEINGYM_SUBSTITUTIONS_CACHE_ID,
)


class FakeResponse:
    """Offline HTTP response that records status and JSON handling order."""

    def __init__(
        self,
        payload: object,
        *,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.raise_called = False
        self.json_called = False

    def raise_for_status(self) -> None:
        """Raise the configured HTTP failure, if any."""
        self.raise_called = True
        if self.error is not None:
            raise self.error

    def json(self) -> object:
        """Return the configured JSON only after status validation."""
        assert self.raise_called
        self.json_called = True
        return self.payload


class FakeSession:
    """Offline session with queued responses and recorded requests."""

    def __init__(
        self,
        *,
        post_responses: list[FakeResponse] | None = None,
        get_responses: list[FakeResponse] | None = None,
    ) -> None:
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.post_calls: list[tuple[str, dict[str, Any], int]] = []
        self.get_calls: list[tuple[str, int]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        """Return the next queued POST response."""
        self.post_calls.append((url, json, timeout))
        return self.post_responses.pop(0)

    def get(self, url: str, *, timeout: int) -> FakeResponse:
        """Return the next queued GET response."""
        self.get_calls.append((url, timeout))
        return self.get_responses.pop(0)


@pytest.fixture
def proteingym_reference_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create lightweight offline ProteinGym substitutions and indels catalogs."""
    substitutions_path = tmp_path / "DMS_substitutions.csv"
    indels_path = tmp_path / "DMS_indels.csv"
    pd.DataFrame(
        [
            {
                "DMS_id": "ZETA_SUB",
                "UniProt_ID": "P99999",
                "DMS_total_number_mutants": 20,
                "extra_field": "kept",
            },
            {
                "DMS_id": "ALPHA_SUB",
                "UniProt_ID": pd.NA,
                "DMS_total_number_mutants": float("nan"),
                "extra_field": pd.NA,
            },
        ]
    ).to_csv(substitutions_path, index=False)
    pd.DataFrame(
        [
            {
                "DMS_id": "BETA_INDEL",
                "UniProt_ID": "Q12345",
                "DMS_total_number_mutants": 7,
                "extra_field": "indel metadata",
            }
        ]
    ).to_csv(indels_path, index=False)
    return substitutions_path, indels_path


def test_mavedb_mapping_preserves_source_metadata() -> None:
    response = FakeResponse(
        {
            "scoreSets": [
                {
                    "urn": "urn:mavedb:00000001-a-1",
                    "title": "Example assay",
                    "numVariants": 42,
                    "targetGenes": [{"name": "GENE1"}],
                    "unknownField": {"nested": True},
                }
            ],
            "numScoreSets": 1,
        }
    )
    catalog = MaveDBCatalog(session=FakeSession(post_responses=[response]))

    records = catalog.list_datasets(query="gene", limit=5, offset=10)

    assert records == [
        DatasetRecord(
            source="mavedb",
            dataset_id="urn:mavedb:00000001-a-1",
            title="Example assay",
            target_id="GENE1",
            variant_type=None,
            n_variants=42,
            raw_metadata={
                "urn": "urn:mavedb:00000001-a-1",
                "title": "Example assay",
                "numVariants": 42,
                "targetGenes": [{"name": "GENE1"}],
                "unknownField": {"nested": True},
            },
        )
    ]


def test_mavedb_accepts_documented_bare_list_response() -> None:
    response = FakeResponse(
        [{"urn": "urn:mavedb:00000002-a-1", "title": "Documented format"}]
    )
    catalog = MaveDBCatalog(session=FakeSession(post_responses=[response]))

    records = catalog.list_datasets()

    assert [record.dataset_id for record in records] == [
        "urn:mavedb:00000002-a-1"
    ]
    assert records[0].title == "Documented format"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"numScoreSets": 0},
        {"unexpected": []},
        {"scoreSets": None},
        {"scoreSets": {}},
        "not an envelope",
    ],
)
def test_mavedb_rejects_malformed_search_envelopes(payload: object) -> None:
    response = FakeResponse(payload)
    catalog = MaveDBCatalog(session=FakeSession(post_responses=[response]))

    with pytest.raises(CatalogError, match="scoreSets|JSON object"):
        catalog.list_datasets()

    assert response.raise_called and response.json_called


def test_mavedb_search_payload_and_no_query_behavior() -> None:
    query_response = FakeResponse({"scoreSets": [], "numScoreSets": 0})
    unfiltered_response = FakeResponse([])
    session = FakeSession(post_responses=[query_response, unfiltered_response])
    catalog = MaveDBCatalog(
        session=session,
        base_url="https://api.example.test/api/v1/",
    )

    assert catalog.list_datasets(query="BRCA1", limit=25, offset=50) == []
    assert catalog.list_datasets(limit=10, offset=0) == []

    assert session.post_calls == [
        (
            "https://api.example.test/api/v1/score-sets/search",
            {"limit": 25, "offset": 50, "text": "BRCA1"},
            60,
        ),
        (
            "https://api.example.test/api/v1/score-sets/search",
            {"limit": 10, "offset": 0},
            60,
        ),
    ]
    assert all("/scores" not in url for url, _, _ in session.post_calls)
    assert query_response.raise_called and query_response.json_called
    assert unfiltered_response.raise_called and unfiltered_response.json_called


def test_mavedb_missing_and_ambiguous_metadata_becomes_none() -> None:
    response = FakeResponse(
        [
            {
                "urn": "urn:mavedb:00000001-a-1",
                "targetGenes": [{"name": "GENE1"}, {"name": "GENE2"}],
            }
        ]
    )
    catalog = MaveDBCatalog(session=FakeSession(post_responses=[response]))

    record = catalog.list_datasets()[0]

    assert record.title is None
    assert record.target_id is None
    assert record.variant_type is None
    assert record.n_variants is None


def test_mavedb_metadata_lookup_uses_urn_endpoint_without_scores() -> None:
    urn = "urn:mavedb:00000001-a-1"
    response = FakeResponse({"urn": urn, "title": "One assay"})
    session = FakeSession(get_responses=[response])
    catalog = MaveDBCatalog(
        session=session,
        base_url="https://api.example.test/api/v1",
    )

    record = catalog.get_metadata(urn)

    assert record.dataset_id == urn
    assert session.get_calls == [
        (f"https://api.example.test/api/v1/score-sets/{urn}", 60)
    ]
    assert all("/scores" not in url for url, _ in session.get_calls)
    assert response.raise_called and response.json_called


@pytest.mark.parametrize("method", ["list", "get"])
def test_mavedb_http_failure_precedes_json_parsing(method: str) -> None:
    response = FakeResponse(
        {"must": "not be parsed"},
        error=requests.HTTPError("503 Server Error"),
    )
    session = FakeSession(
        post_responses=[response] if method == "list" else None,
        get_responses=[response] if method == "get" else None,
    )
    catalog = MaveDBCatalog(session=session)

    with pytest.raises(requests.HTTPError, match="503"):
        if method == "list":
            catalog.list_datasets()
        else:
            catalog.get_metadata("urn:mavedb:00000001-a-1")

    assert response.raise_called
    assert response.json_called is False


def test_proteingym_parses_both_reference_files(
    proteingym_reference_files: tuple[Path, Path],
) -> None:
    substitutions_path, indels_path = proteingym_reference_files
    catalog = ProteinGymCatalog(
        substitutions_path=substitutions_path,
        indels_path=indels_path,
    )

    records = catalog.list_datasets()

    assert [record.dataset_id for record in records] == [
        "ALPHA_SUB",
        "BETA_INDEL",
        "ZETA_SUB",
    ]
    assert [record.variant_type for record in records] == [
        "substitutions",
        "indels",
        "substitutions",
    ]
    zeta = records[2]
    assert zeta.title is None
    assert zeta.target_id == "P99999"
    assert zeta.n_variants == 20
    assert zeta.raw_metadata["extra_field"] == "kept"


def test_proteingym_raw_metadata_is_json_safe(
    proteingym_reference_files: tuple[Path, Path],
) -> None:
    substitutions_path, indels_path = proteingym_reference_files
    catalog = ProteinGymCatalog(
        substitutions_path=substitutions_path,
        indels_path=indels_path,
    )

    record = catalog.get_metadata("ALPHA_SUB", variant_type="substitutions")

    assert record.target_id is None
    assert record.n_variants is None
    assert record.raw_metadata["UniProt_ID"] is None
    assert record.raw_metadata["DMS_total_number_mutants"] is None
    assert record.raw_metadata["extra_field"] is None
    json.dumps(record.raw_metadata, allow_nan=False)
    assert not _contains_missing_value(record.raw_metadata)


def test_recursive_missing_values_and_dates_are_json_safe() -> None:
    valid_datetime = datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc)
    valid_date = date(2024, 1, 2)

    metadata = normalize_raw_metadata(
        {
            "nested": {
                "values": [pd.NaT, pd.NA, float("nan")],
                "datetime": valid_datetime,
                "date": valid_date,
            }
        }
    )

    assert metadata == {
        "nested": {
            "values": [None, None, None],
            "datetime": valid_datetime.isoformat(),
            "date": valid_date.isoformat(),
        }
    }
    json.dumps(metadata, allow_nan=False)


def test_proteingym_query_filter_is_case_insensitive(
    proteingym_reference_files: tuple[Path, Path],
) -> None:
    substitutions_path, indels_path = proteingym_reference_files
    catalog = ProteinGymCatalog(
        substitutions_path=substitutions_path,
        indels_path=indels_path,
    )

    by_dataset = catalog.list_datasets(query="zeta")
    by_target = catalog.list_datasets(query="q123")

    assert [record.dataset_id for record in by_dataset] == ["ZETA_SUB"]
    assert [record.dataset_id for record in by_target] == ["BETA_INDEL"]


def test_proteingym_variant_filter_limit_and_offset(
    proteingym_reference_files: tuple[Path, Path],
) -> None:
    substitutions_path, indels_path = proteingym_reference_files
    catalog = ProteinGymCatalog(
        substitutions_path=substitutions_path,
        indels_path=indels_path,
    )

    substitutions = catalog.list_datasets(variant_type="substitutions")
    indels = catalog.list_datasets(variant_type="indels")
    page = catalog.list_datasets(limit=1, offset=1)

    assert [record.dataset_id for record in substitutions] == [
        "ALPHA_SUB",
        "ZETA_SUB",
    ]
    assert [record.dataset_id for record in indels] == ["BETA_INDEL"]
    assert [record.dataset_id for record in page] == ["BETA_INDEL"]


def test_proteingym_metadata_lookup_and_unknown_id(
    proteingym_reference_files: tuple[Path, Path],
) -> None:
    substitutions_path, indels_path = proteingym_reference_files
    catalog = ProteinGymCatalog(
        substitutions_path=substitutions_path,
        indels_path=indels_path,
    )

    record = catalog.get_metadata("BETA_INDEL")

    assert record.dataset_id == "BETA_INDEL"
    assert record.variant_type == "indels"
    with pytest.raises(DatasetNotFoundError, match="UNKNOWN"):
        catalog.get_metadata("UNKNOWN")


def test_proteingym_reference_files_reuse_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substitutions_url = "https://example.test/reference_files/DMS_substitutions.csv"
    indels_url = "https://example.test/reference_files/DMS_indels.csv"
    content_by_url = {
        substitutions_url: (
            b"DMS_id,UniProt_ID,DMS_total_number_mutants\nSUB_ONE,P11111,10\n"
        ),
        indels_url: (
            b"DMS_id,UniProt_ID,DMS_total_number_mutants\nINDEL_ONE,Q22222,5\n"
        ),
    }
    download_calls: list[str] = []

    def offline_download(
        url: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 8192,
        timeout: int = 60,
    ) -> Path:
        del overwrite, chunk_size, timeout
        download_calls.append(url)
        path = Path(output_path)
        path.write_bytes(content_by_url[url])
        return path

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")

    first = ProteinGymCatalog(
        cache=cache,
        substitutions_url=substitutions_url,
        indels_url=indels_url,
    ).list_datasets()
    second = ProteinGymCatalog(
        cache=cache,
        substitutions_url=substitutions_url,
        indels_url=indels_url,
    ).list_datasets()

    assert [record.dataset_id for record in first] == ["INDEL_ONE", "SUB_ONE"]
    assert second == first
    assert download_calls == [substitutions_url, indels_url]
    substitutions_manifest = cache.load_manifest(
        "proteingym",
        PROTEINGYM_SUBSTITUTIONS_CACHE_ID,
    )
    indels_manifest = cache.load_manifest(
        "proteingym",
        PROTEINGYM_INDELS_CACHE_ID,
    )
    assert substitutions_manifest.original_url == substitutions_url
    assert indels_manifest.original_url == indels_url
    assert all("scores" not in url for url in download_calls)


def test_public_catalog_api_dispatches_both_sources(
    proteingym_reference_files: tuple[Path, Path],
) -> None:
    substitutions_path, indels_path = proteingym_reference_files
    mavedb_catalog = MaveDBCatalog(
        session=FakeSession(
            post_responses=[
                FakeResponse([{"urn": "urn:mavedb:00000001-a-1"}])
            ],
            get_responses=[
                FakeResponse({"urn": "urn:mavedb:00000001-a-1"})
            ],
        )
    )
    proteingym_catalog = ProteinGymCatalog(
        substitutions_path=substitutions_path,
        indels_path=indels_path,
    )

    mavedb_records = list_datasets(
        "mavedb",
        query="example",
        catalog=mavedb_catalog,
    )
    proteingym_records = list_datasets(
        "proteingym",
        variant_type="indels",
        catalog=proteingym_catalog,
    )
    mavedb_record = get_dataset_metadata(
        "mavedb",
        "urn:mavedb:00000001-a-1",
        catalog=mavedb_catalog,
    )
    proteingym_record = get_dataset_metadata(
        "proteingym",
        "ZETA_SUB",
        catalog=proteingym_catalog,
    )

    assert all(isinstance(record, DatasetRecord) for record in mavedb_records)
    assert all(isinstance(record, DatasetRecord) for record in proteingym_records)
    assert isinstance(mavedb_record, DatasetRecord)
    assert isinstance(proteingym_record, DatasetRecord)
    assert mavedb_record.source == "mavedb"
    assert proteingym_record.source == "proteingym"


@pytest.mark.parametrize("source", ["all", "unknown", "", 123])
def test_public_catalog_rejects_invalid_source(source: object) -> None:
    with pytest.raises(InvalidCatalogQueryError):
        list_datasets(source)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("mavedb", {"limit": 0}),
        ("mavedb", {"offset": -1}),
        ("proteingym", {"variant_type": "deletions"}),
        ("proteingym", {"limit": 0}),
        ("proteingym", {"offset": -1}),
    ],
)
def test_catalogs_reject_invalid_filters_before_loading(
    method: str,
    kwargs: dict[str, Any],
) -> None:
    if method == "mavedb":
        catalog: Any = MaveDBCatalog(session=FakeSession())
    else:
        catalog = ProteinGymCatalog()

    with pytest.raises(InvalidCatalogQueryError):
        catalog.list_datasets(**kwargs)


def test_source_specific_arguments_are_not_silently_ignored(
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidCatalogQueryError, match="variant_type"):
        list_datasets("mavedb", variant_type="substitutions")
    with pytest.raises(InvalidCatalogQueryError, match="cache"):
        list_datasets("mavedb", cache=FilesystemCache(tmp_path / "cache"))
    with pytest.raises(InvalidCatalogQueryError, match="FilesystemCache"):
        list_datasets("proteingym", variant_type="substitutions")


def _contains_missing_value(value: object) -> bool:
    """Return whether nested metadata contains a pandas or floating NaN value."""
    if isinstance(value, dict):
        return any(_contains_missing_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_missing_value(item) for item in value)
    if value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)

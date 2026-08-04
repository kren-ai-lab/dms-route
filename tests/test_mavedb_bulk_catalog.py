from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import dms_parser
import dms_parser.sources as sources_module
import dms_parser.sources.mavedb_bulk_catalog as bulk_module
from dms_parser import (
    MaveDBBulkCatalog,
    MaveDBDiscoveredExperiment,
    MaveDBDiscoveredScoreSet,
    MaveDBDiscoveryResult,
)
from dms_parser.exceptions import CatalogError, InvalidCatalogQueryError


def _target(
    *,
    name: str | None = None,
    mapped_name: str | None = None,
    gene: str | None = None,
    accession: str | None = None,
    uniprot: str | None = None,
    external: str | None = None,
) -> dict[str, Any]:
    """Return one target containing the requested searchable fields."""
    value: dict[str, Any] = {}
    if name is not None:
        value["name"] = name
    if mapped_name is not None:
        value["mappedHgncName"] = mapped_name
    if gene is not None or accession is not None:
        value["targetAccession"] = {}
        if gene is not None:
            value["targetAccession"]["gene"] = gene
        if accession is not None:
            value["targetAccession"]["accession"] = accession
    if uniprot is not None:
        value["uniprotIdFromMappedMetadata"] = uniprot
    if external is not None:
        value["externalIdentifiers"] = [
            {"identifier": {"identifier": external}}
        ]
    return value


def _valid_snapshot() -> dict[str, Any]:
    """Return a complete compact bulk hierarchy for discovery tests."""
    return {
        "title": "MaveDB public data dump v5",
        "asOf": "2026-06-24T18:13:01Z",
        "experimentSets": [
            {
                "urn": "urn:mavedb:00000002",
                "experiments": [
                    {
                        "urn": "urn:mavedb:00000002-a",
                        "title": "Zeta experiment",
                        "scoreSetUrns": [
                            "urn:mavedb:00000002-a-2",
                            "urn:mavedb:00000002-a-3",
                        ],
                        "scoreSets": [
                            {
                                "urn": "urn:mavedb:00000002-a-3",
                                "title": "Unrelated score set",
                                "numVariants": 5,
                                "targetGenes": [_target(name="OTHER")],
                            },
                            {
                                "urn": "urn:mavedb:00000002-a-2",
                                "title": "Current BRCA1 score set",
                                "numVariants": 12,
                                "targetGenes": [
                                    _target(
                                        name="BRCA1",
                                        mapped_name="BRCA One",
                                        gene="Breast cancer 1",
                                        accession="HGNC:1100",
                                        uniprot="P38398",
                                        external="NCBI:672",
                                    ),
                                    _target(name="Partner"),
                                    _target(name="brca1"),
                                ],
                            },
                            {
                                "urn": "urn:mavedb:00000002-a-1",
                                "title": "Legacy BRCA1 score set",
                                "numVariants": 10,
                                "targetGenes": [_target(name="BRCA1 legacy")],
                            },
                        ],
                    }
                ],
            },
            {
                "urn": "urn:mavedb:00000001",
                "experiments": [
                    {
                        "urn": "urn:mavedb:00000001-a",
                        "title": "Alpha experiment",
                        "scoreSetUrns": [
                            "urn:mavedb:00000001-a-5",
                            "urn:mavedb:00000001-a-4",
                            "urn:mavedb:00000001-a-3",
                            "urn:mavedb:00000001-a-2",
                            "urn:mavedb:00000001-a-1",
                        ],
                        "scoreSets": [
                            {
                                "urn": "urn:mavedb:00000001-a-5",
                                "title": "Mapped name",
                                "numVariants": 20,
                                "targetGenes": [
                                    _target(mapped_name="Mapped BRCA2")
                                ],
                            },
                            {
                                "urn": "urn:mavedb:00000001-a-4",
                                "title": "Accession target",
                                "numVariants": 21,
                                "targetGenes": [
                                    _target(gene="BRCA3", accession="HGNC:3430")
                                ],
                            },
                            {
                                "urn": "urn:mavedb:00000001-a-3",
                                "title": "UniProt target",
                                "numVariants": 22,
                                "targetGenes": [_target(uniprot="P51587")],
                            },
                            {
                                "urn": "urn:mavedb:00000001-a-2",
                                "title": "External target",
                                "numVariants": 23,
                                "targetGenes": [
                                    _target(external="ENSEMBL:ENSG000001")
                                ],
                            },
                            {
                                "urn": "urn:mavedb:00000001-a-1",
                                "title": "Unicode target",
                                "numVariants": 24,
                                "targetGenes": [_target(name="Straße kinase")],
                            },
                        ],
                    }
                ],
            },
        ],
    }


def _write_snapshot(tmp_path: Path, payload: object | None = None) -> Path:
    """Write one UTF-8 fixture and return its path."""
    path = tmp_path / "main.json"
    path.write_text(
        json.dumps(_valid_snapshot() if payload is None else payload),
        encoding="utf-8",
    )
    return path


def _catalog(tmp_path: Path) -> MaveDBBulkCatalog:
    """Load the standard valid fixture."""
    return MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path))


def test_valid_snapshot_loads_metadata_and_immutable_results(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    result = catalog.search_by_gene("BRCA")

    assert catalog.snapshot_title == "MaveDB public data dump v5"
    assert catalog.as_of == "2026-06-24T18:13:01Z"
    assert result.snapshot_title == catalog.snapshot_title
    assert result.as_of == catalog.as_of
    assert result.experiment_count == 2
    assert result.score_set_count == 3
    with pytest.raises(FrozenInstanceError):
        result.query = "changed"  # type: ignore[misc]


def test_repeated_searches_do_not_reopen_or_reparse_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(tmp_path)

    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Search reopened main.json")
        ),
    )
    monkeypatch.setattr(
        bulk_module.json,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Search reparsed main.json")
        ),
    )

    assert catalog.search_by_gene("BRCA1").score_set_count == 1
    assert catalog.search_by_gene("P51587").score_set_count == 1


def test_score_set_lookup_returns_current_superseded_and_missing_records(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)

    current = catalog.get_score_set("urn:mavedb:00000002-a-2")
    superseded = catalog.get_score_set("urn:mavedb:00000002-a-1")

    assert current is not None and current.is_superseded is False
    assert superseded is not None and superseded.is_superseded is True
    assert catalog.get_score_set("urn:mavedb:99999999-a-1") is None


@pytest.mark.parametrize(
    "dataset_id",
    ["", "invalid", "urn:mavedb:00000002-A-1", True, None],
)
def test_score_set_lookup_rejects_noncanonical_urns(
    dataset_id: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidCatalogQueryError, match="score-set URN|non-empty"):
        _catalog(tmp_path).get_score_set(dataset_id)  # type: ignore[arg-type]


def test_repeated_score_set_lookups_use_detached_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(tmp_path)
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Lookup reopened main.json")
        ),
    )
    catalog._score_sets = ()

    assert catalog.get_score_set("urn:mavedb:00000002-a-2") is not None
    assert catalog.get_score_set("urn:mavedb:99999999-a-1") is None


def test_name_search_groups_sorts_and_does_not_leak_siblings(tmp_path: Path) -> None:
    result = _catalog(tmp_path).search_by_gene("  bRcA  ")

    assert result.query == "bRcA"
    assert [item.experiment_id for item in result.experiments] == [
        "urn:mavedb:00000001-a",
        "urn:mavedb:00000002-a",
    ]
    assert [
        score_set.dataset_id
        for experiment in result.experiments
        for score_set in experiment.score_sets
    ] == [
        "urn:mavedb:00000001-a-4",
        "urn:mavedb:00000001-a-5",
        "urn:mavedb:00000002-a-2",
    ]
    assert all(
        score_set.dataset_id != "urn:mavedb:00000002-a-3"
        for experiment in result.experiments
        for score_set in experiment.score_sets
    )


def test_unicode_casefold_name_matching(tmp_path: Path) -> None:
    result = _catalog(tmp_path).search_by_gene("STRASSE")

    assert result.score_set_count == 1
    assert result.experiments[0].score_sets[0].dataset_id.endswith("-a-1")


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("BRCA1", "urn:mavedb:00000002-a-2"),
        ("mapped brca2", "urn:mavedb:00000001-a-5"),
        ("BRCA3", "urn:mavedb:00000001-a-4"),
    ],
)
def test_all_name_fields_use_case_insensitive_substrings(
    query: str,
    expected_id: str,
    tmp_path: Path,
) -> None:
    result = _catalog(tmp_path).search_by_gene(query)

    assert [
        score_set.dataset_id
        for experiment in result.experiments
        for score_set in experiment.score_sets
    ] == [expected_id]


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("hgnc:1100", "urn:mavedb:00000002-a-2"),
        ("p38398", "urn:mavedb:00000002-a-2"),
        ("ncbi:672", "urn:mavedb:00000002-a-2"),
        ("HGNC:3430", "urn:mavedb:00000001-a-4"),
        ("P51587", "urn:mavedb:00000001-a-3"),
        ("ensembl:ensg000001", "urn:mavedb:00000001-a-2"),
    ],
)
def test_identifier_fields_use_case_insensitive_exact_matching(
    query: str,
    expected_id: str,
    tmp_path: Path,
) -> None:
    result = _catalog(tmp_path).search_by_gene(query)

    assert result.score_set_count == 1
    assert result.experiments[0].score_sets[0].dataset_id == expected_id


@pytest.mark.parametrize("query", ["P515", "HGNC:343", "ENSG000001"])
def test_partial_identifiers_do_not_match(query: str, tmp_path: Path) -> None:
    assert _catalog(tmp_path).search_by_gene(query).score_set_count == 0


def test_duplicate_field_matches_return_one_score_set(tmp_path: Path) -> None:
    result = _catalog(tmp_path).search_by_gene("BRCA1")

    assert result.score_set_count == 1
    assert result.experiments[0].score_sets[0].dataset_id == (
        "urn:mavedb:00000002-a-2"
    )


def test_target_labels_are_deduplicated_sorted_and_include_all_targets(
    tmp_path: Path,
) -> None:
    score_set = _catalog(tmp_path).search_by_gene("HGNC:1100").experiments[0].score_sets[0]

    assert score_set.targets == ("BRCA1", "Partner")


@pytest.mark.parametrize(
    ("query", "expected_label"),
    [
        ("Mapped BRCA2", "Mapped BRCA2"),
        ("HGNC:3430", "BRCA3"),
        ("P51587", "P51587"),
        ("ENSEMBL:ENSG000001", "ENSEMBL:ENSG000001"),
    ],
)
def test_target_label_priority_and_fallbacks(
    query: str,
    expected_label: str,
    tmp_path: Path,
) -> None:
    result = _catalog(tmp_path).search_by_gene(query)

    assert result.experiments[0].score_sets[0].targets == (expected_label,)


def test_empty_search_result_is_successful(tmp_path: Path) -> None:
    result = _catalog(tmp_path).search_by_gene("NOT_PRESENT")

    assert result.experiments == ()
    assert result.experiment_count == 0
    assert result.score_set_count == 0


def test_superseded_records_are_opt_in_and_correctly_marked(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    current_only = catalog.search_by_gene("BRCA1")
    with_history = catalog.search_by_gene(
        "BRCA1",
        include_superseded=True,
    )

    assert current_only.score_set_count == 1
    assert current_only.experiments[0].score_sets[0].is_superseded is False
    assert [record.is_superseded for record in with_history.experiments[0].score_sets] == [
        True,
        False,
    ]


def test_unmatched_superseded_record_is_not_included(tmp_path: Path) -> None:
    result = _catalog(tmp_path).search_by_gene(
        "OTHER",
        include_superseded=True,
    )

    assert result.score_set_count == 1
    assert result.experiments[0].score_sets[0].dataset_id.endswith("-a-3")


@pytest.mark.parametrize("query", ["", "   ", None, 7, True])
def test_invalid_queries_are_rejected(query: object, tmp_path: Path) -> None:
    with pytest.raises(InvalidCatalogQueryError, match="query"):
        _catalog(tmp_path).search_by_gene(query)  # type: ignore[arg-type]


def test_non_boolean_include_superseded_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidCatalogQueryError, match="include_superseded"):
        _catalog(tmp_path).search_by_gene(
            "BRCA1",
            include_superseded=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("path", ["", "   ", True, False, b"main.json", 7, None])
def test_invalid_path_arguments_are_rejected(path: object) -> None:
    with pytest.raises(InvalidCatalogQueryError, match="path"):
        MaveDBBulkCatalog.from_file(path)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("experimentSets",), None, "experimentSets"),
        (("experimentSets", 0), None, "experimentSets\\[0\\]"),
        (("experimentSets", 0, "urn"), "", "urn"),
        (("experimentSets", 0, "experiments"), None, "experiments"),
        (("experimentSets", 0, "experiments", 0), None, "experiments\\[0\\]"),
        (("experimentSets", 0, "experiments", 0, "urn"), "", "urn"),
        (("experimentSets", 0, "experiments", 0, "scoreSetUrns"), None, "scoreSetUrns"),
        (("experimentSets", 0, "experiments", 0, "scoreSetUrns", 0), 3, "scoreSetUrns\\[0\\]"),
        (("experimentSets", 0, "experiments", 0, "scoreSets"), None, "scoreSets"),
        (("experimentSets", 0, "experiments", 0, "scoreSets", 0), None, "scoreSets\\[0\\]"),
        (("experimentSets", 0, "experiments", 0, "scoreSets", 0, "urn"), "", "urn"),
        (("experimentSets", 0, "experiments", 0, "scoreSets", 0, "targetGenes"), None, "targetGenes"),
        (("experimentSets", 0, "experiments", 0, "scoreSets", 0, "targetGenes", 0), None, "targetGenes\\[0\\]"),
    ],
)
def test_malformed_required_hierarchy_is_rejected_with_context(
    path: tuple[object, ...],
    value: object,
    message: str,
    tmp_path: Path,
) -> None:
    payload = _valid_snapshot()
    _set_nested(payload, path, value)

    with pytest.raises(CatalogError, match=message):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


def test_root_must_be_an_object(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="root"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, []))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("targetAccession", []),
        ("externalIdentifiers", {}),
    ],
)
def test_malformed_optional_target_structures_are_rejected(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    payload = _valid_snapshot()
    target = payload["experimentSets"][0]["experiments"][0]["scoreSets"][0][
        "targetGenes"
    ][0]
    target[field] = value

    with pytest.raises(CatalogError, match=field):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


@pytest.mark.parametrize(
    "value",
    [None, [], {}, {"identifier": None}, {"identifier": []}],
)
def test_malformed_external_identifier_is_rejected(
    value: object,
    tmp_path: Path,
) -> None:
    payload = _valid_snapshot()
    target = payload["experimentSets"][0]["experiments"][0]["scoreSets"][0][
        "targetGenes"
    ][0]
    target["externalIdentifiers"] = [value]

    with pytest.raises(CatalogError, match="externalIdentifiers\\[0\\]"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "2"])
def test_invalid_num_variants_is_rejected(value: object, tmp_path: Path) -> None:
    payload = _valid_snapshot()
    payload["experimentSets"][0]["experiments"][0]["scoreSets"][0][
        "numVariants"
    ] = value

    with pytest.raises(CatalogError, match="numVariants"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


def test_duplicate_experiment_urn_is_rejected(tmp_path: Path) -> None:
    payload = _valid_snapshot()
    duplicate = copy.deepcopy(payload["experimentSets"][0]["experiments"][0])
    duplicate["scoreSetUrns"] = []
    duplicate["scoreSets"] = []
    payload["experimentSets"][1]["experiments"].append(duplicate)

    with pytest.raises(CatalogError, match="Duplicate experiment URN"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


def test_duplicate_score_set_urn_is_rejected(tmp_path: Path) -> None:
    payload = _valid_snapshot()
    duplicate = copy.deepcopy(
        payload["experimentSets"][0]["experiments"][0]["scoreSets"][0]
    )
    payload["experimentSets"][1]["experiments"][0]["scoreSets"].append(duplicate)

    with pytest.raises(CatalogError, match="Duplicate score-set URN"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


def test_duplicate_current_score_set_urn_is_rejected(tmp_path: Path) -> None:
    payload = _valid_snapshot()
    current = payload["experimentSets"][0]["experiments"][0]["scoreSetUrns"]
    current.append(current[0])

    with pytest.raises(CatalogError, match="Duplicate score-set URN"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


def test_current_urn_missing_from_nested_score_sets_is_rejected(tmp_path: Path) -> None:
    payload = _valid_snapshot()
    payload["experimentSets"][0]["experiments"][0]["scoreSetUrns"].append(
        "urn:mavedb:missing-a-1"
    )

    with pytest.raises(CatalogError, match="missing from scoreSets"):
        MaveDBBulkCatalog.from_file(_write_snapshot(tmp_path, payload))


def test_nested_superseded_score_set_absent_from_current_list_is_valid(
    tmp_path: Path,
) -> None:
    result = _catalog(tmp_path).search_by_gene(
        "legacy",
        include_superseded=True,
    )

    assert result.score_set_count == 1
    assert result.experiments[0].score_sets[0].is_superseded is True


def test_invalid_utf8_becomes_catalog_error(tmp_path: Path) -> None:
    path = tmp_path / "main.json"
    path.write_bytes(b'{"experimentSets": []}\xff')

    with pytest.raises(CatalogError, match="UTF-8"):
        MaveDBBulkCatalog.from_file(path)


def test_invalid_json_becomes_catalog_error(tmp_path: Path) -> None:
    path = tmp_path / "main.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(CatalogError, match="valid JSON"):
        MaveDBBulkCatalog.from_file(path)


def test_public_bulk_catalog_exports_are_identical() -> None:
    names = (
        "MaveDBBulkCatalog",
        "MaveDBDiscoveredExperiment",
        "MaveDBDiscoveredScoreSet",
        "MaveDBDiscoveryResult",
    )
    for name in names:
        assert getattr(dms_parser, name) is getattr(sources_module, name)
        assert getattr(dms_parser, name) is getattr(bulk_module, name)

    assert MaveDBDiscoveredExperiment is bulk_module.MaveDBDiscoveredExperiment
    assert MaveDBDiscoveredScoreSet is bulk_module.MaveDBDiscoveredScoreSet
    assert MaveDBDiscoveryResult is bulk_module.MaveDBDiscoveryResult


def _set_nested(
    root: dict[str, Any],
    path: tuple[object, ...],
    value: object,
) -> None:
    """Set one nested test-fixture value by keys and list indices."""
    current: Any = root
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value

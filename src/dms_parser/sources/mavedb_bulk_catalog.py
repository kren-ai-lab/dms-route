"""Read-only discovery over a local MaveDB bulk ``main.json``."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dms_parser.catalog import normalize_raw_metadata
from dms_parser.exceptions import (
    CatalogError,
    InvalidCatalogQueryError,
    SourceConfigurationError,
)


@dataclass(frozen=True)
class MaveDBDiscoveredScoreSet:
    """One matching MaveDB score set from a bulk snapshot."""

    dataset_id: str
    title: str | None
    n_variants: int | None
    targets: tuple[str, ...]
    is_superseded: bool


@dataclass(frozen=True)
class MaveDBDiscoveredExperiment:
    """Matching score sets grouped under their containing experiment."""

    experiment_set_id: str
    experiment_id: str
    title: str | None
    score_sets: tuple[MaveDBDiscoveredScoreSet, ...]


@dataclass(frozen=True)
class MaveDBDiscoveryResult:
    """Deterministic result of one local MaveDB bulk search."""

    query: str
    snapshot_title: str | None
    as_of: str | None
    experiments: tuple[MaveDBDiscoveredExperiment, ...]

    @property
    def experiment_count(self) -> int:
        """Return the number of experiments containing matches."""
        return len(self.experiments)

    @property
    def score_set_count(self) -> int:
        """Return the number of matching score sets across experiments."""
        return sum(len(experiment.score_sets) for experiment in self.experiments)


@dataclass(frozen=True)
class _IndexedScoreSet:
    """Search-ready score set detached from the decoded JSON hierarchy."""

    experiment_set_id: str
    experiment_id: str
    experiment_title: str | None
    record: MaveDBDiscoveredScoreSet
    metadata: dict[str, Any]


class MaveDBBulkCatalog:
    """Load one MaveDB bulk catalog once and search its in-memory index."""

    def __init__(
        self,
        *,
        snapshot_title: str | None,
        as_of: str | None,
        score_sets: tuple[_IndexedScoreSet, ...],
        name_index: tuple[tuple[str, int], ...],
        identifier_index: Mapping[str, frozenset[int]],
    ) -> None:
        self.snapshot_title = snapshot_title
        self.as_of = as_of
        self._score_sets = score_sets
        self._score_set_index = MappingProxyType(
            {
                indexed.record.dataset_id: indexed.record
                for indexed in score_sets
            }
        )
        self._score_set_metadata_index = MappingProxyType(
            {
                indexed.record.dataset_id: indexed.metadata
                for indexed in score_sets
            }
        )
        self._name_index = name_index
        self._identifier_index = MappingProxyType(dict(identifier_index))

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
    ) -> MaveDBBulkCatalog:
        """Load, validate, and index one UTF-8 MaveDB ``main.json``."""
        resolved_path = _validate_path(path)
        try:
            with resolved_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except UnicodeError as exc:
            raise CatalogError(
                f"MaveDB bulk catalog is not valid UTF-8: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise CatalogError(
                f"MaveDB bulk catalog is not valid JSON: {exc.msg}."
            ) from exc

        (
            snapshot_title,
            as_of,
            score_sets,
            name_index,
            identifier_index,
        ) = _validate_and_index(payload)
        return cls(
            snapshot_title=snapshot_title,
            as_of=as_of,
            score_sets=score_sets,
            name_index=name_index,
            identifier_index=identifier_index,
        )

    def search_by_gene(
        self,
        query: str,
        *,
        include_superseded: bool = False,
    ) -> MaveDBDiscoveryResult:
        """Search target names by substring and identifiers by exact match."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidCatalogQueryError(
                "query must be a non-empty string."
            )
        if not isinstance(include_superseded, bool):
            raise InvalidCatalogQueryError(
                "include_superseded must be a boolean."
            )
        displayed_query = query.strip()
        normalized_query = displayed_query.casefold()
        matching_indices = set(
            self._identifier_index.get(normalized_query, frozenset())
        )
        matching_indices.update(
            score_set_index
            for name, score_set_index in self._name_index
            if normalized_query in name
        )

        grouped: dict[
            str,
            tuple[str, str | None, list[MaveDBDiscoveredScoreSet]],
        ] = {}
        for score_set_index in matching_indices:
            indexed = self._score_sets[score_set_index]
            if indexed.record.is_superseded and not include_superseded:
                continue
            group = grouped.setdefault(
                indexed.experiment_id,
                (
                    indexed.experiment_set_id,
                    indexed.experiment_title,
                    [],
                ),
            )
            group[2].append(indexed.record)

        experiments = tuple(
            MaveDBDiscoveredExperiment(
                experiment_set_id=grouped[experiment_id][0],
                experiment_id=experiment_id,
                title=grouped[experiment_id][1],
                score_sets=tuple(
                    sorted(
                        grouped[experiment_id][2],
                        key=lambda record: record.dataset_id,
                    )
                ),
            )
            for experiment_id in sorted(grouped)
        )
        return MaveDBDiscoveryResult(
            query=displayed_query,
            snapshot_title=self.snapshot_title,
            as_of=self.as_of,
            experiments=experiments,
        )

    def get_score_set(
        self,
        dataset_id: str,
    ) -> MaveDBDiscoveredScoreSet | None:
        """Return one indexed current or superseded score set by canonical URN."""
        from dms_parser.config import validate_source_dataset_id

        try:
            validate_source_dataset_id("mavedb", dataset_id)
        except SourceConfigurationError as exc:
            raise InvalidCatalogQueryError(str(exc)) from exc
        return self._score_set_index.get(dataset_id)

    def get_score_set_metadata(
        self,
        dataset_id: str,
    ) -> dict[str, Any] | None:
        """Return a detached normalized score-set metadata object by URN."""
        self.get_score_set(dataset_id)
        metadata = self._score_set_metadata_index.get(dataset_id)
        return copy.deepcopy(metadata) if metadata is not None else None


def _validate_path(path: object) -> Path:
    """Return a supported non-empty filesystem path argument."""
    if isinstance(path, (bool, bytes)) or not isinstance(path, (str, os.PathLike)):
        raise InvalidCatalogQueryError(
            "path must be a non-empty string or os.PathLike[str]."
        )
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise InvalidCatalogQueryError(
            "path must be a non-empty string or os.PathLike[str]."
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise InvalidCatalogQueryError(
            "path must be a non-empty string or os.PathLike[str]."
        )
    return Path(value)


def _validate_and_index(
    payload: object,
) -> tuple[
    str | None,
    str | None,
    tuple[_IndexedScoreSet, ...],
    tuple[tuple[str, int], ...],
    Mapping[str, frozenset[int]],
]:
    """Validate the complete hierarchy and build detached search indices."""
    root = _mapping(payload, "root")
    snapshot_title = _optional_string(root, "title", "root")
    as_of = _optional_string(root, "asOf", "root")
    experiment_sets = _list_field(root, "experimentSets", "root")

    indexed_score_sets: list[_IndexedScoreSet] = []
    name_index: list[tuple[str, int]] = []
    identifier_index: dict[str, set[int]] = {}
    seen_experiments: set[str] = set()
    seen_score_sets: set[str] = set()

    for experiment_set_index, raw_experiment_set in enumerate(experiment_sets):
        experiment_set_context = f"experimentSets[{experiment_set_index}]"
        experiment_set = _mapping(raw_experiment_set, experiment_set_context)
        experiment_set_id = _required_string(
            experiment_set,
            "urn",
            experiment_set_context,
        )
        experiments = _list_field(
            experiment_set,
            "experiments",
            experiment_set_context,
        )

        for experiment_index, raw_experiment in enumerate(experiments):
            experiment_context = (
                f"{experiment_set_context}.experiments[{experiment_index}]"
            )
            experiment = _mapping(raw_experiment, experiment_context)
            experiment_id = _required_string(
                experiment,
                "urn",
                experiment_context,
            )
            if experiment_id in seen_experiments:
                raise CatalogError(
                    f"Duplicate experiment URN {experiment_id!r} at "
                    f"{experiment_context}.urn."
                )
            seen_experiments.add(experiment_id)
            experiment_title = _optional_string(
                experiment,
                "title",
                experiment_context,
            )
            current_values = _list_field(
                experiment,
                "scoreSetUrns",
                experiment_context,
            )
            current_ids: list[str] = []
            current_seen: set[str] = set()
            for current_index, current_value in enumerate(current_values):
                current_context = (
                    f"{experiment_context}.scoreSetUrns[{current_index}]"
                )
                current_id = _standalone_required_string(
                    current_value,
                    current_context,
                )
                if current_id in current_seen:
                    raise CatalogError(
                        f"Duplicate score-set URN {current_id!r} in "
                        f"{experiment_context}.scoreSetUrns."
                    )
                current_seen.add(current_id)
                current_ids.append(current_id)

            raw_score_sets = _list_field(
                experiment,
                "scoreSets",
                experiment_context,
            )
            nested_ids: set[str] = set()
            pending: list[
                tuple[
                    MaveDBDiscoveredScoreSet,
                    tuple[str, ...],
                    tuple[str, ...],
                    dict[str, Any],
                ]
            ] = []
            for score_set_index, raw_score_set in enumerate(raw_score_sets):
                score_set_context = (
                    f"{experiment_context}.scoreSets[{score_set_index}]"
                )
                score_set = _mapping(raw_score_set, score_set_context)
                dataset_id = _required_string(
                    score_set,
                    "urn",
                    score_set_context,
                )
                if dataset_id in seen_score_sets:
                    raise CatalogError(
                        f"Duplicate score-set URN {dataset_id!r} at "
                        f"{score_set_context}.urn."
                    )
                seen_score_sets.add(dataset_id)
                nested_ids.add(dataset_id)
                title = _optional_string(score_set, "title", score_set_context)
                n_variants = _optional_non_negative_integer(
                    score_set,
                    "numVariants",
                    score_set_context,
                )
                target_values = _list_field(
                    score_set,
                    "targetGenes",
                    score_set_context,
                )
                names: set[str] = set()
                identifiers: set[str] = set()
                labels: dict[str, str] = {}
                for target_index, raw_target in enumerate(target_values):
                    target_context = (
                        f"{score_set_context}.targetGenes[{target_index}]"
                    )
                    target = _mapping(raw_target, target_context)
                    target_names, target_identifiers, label = _target_values(
                        target,
                        target_context,
                    )
                    names.update(value.casefold() for value in target_names)
                    identifiers.update(
                        value.casefold() for value in target_identifiers
                    )
                    if label is not None:
                        labels.setdefault(label.casefold(), label)

                displayed_targets = tuple(
                    sorted(labels.values(), key=lambda value: (value.casefold(), value))
                )
                pending.append(
                    (
                        MaveDBDiscoveredScoreSet(
                            dataset_id=dataset_id,
                            title=title,
                            n_variants=n_variants,
                            targets=displayed_targets,
                            is_superseded=dataset_id not in current_seen,
                        ),
                        tuple(sorted(names)),
                        tuple(sorted(identifiers)),
                        normalize_raw_metadata(score_set),
                    )
                )

            missing_current = set(current_ids).difference(nested_ids)
            if missing_current:
                missing = ", ".join(sorted(missing_current))
                raise CatalogError(
                    f"{experiment_context}.scoreSetUrns references score sets "
                    f"missing from scoreSets: {missing}."
                )

            for record, names, identifiers, metadata in pending:
                index = len(indexed_score_sets)
                indexed_score_sets.append(
                    _IndexedScoreSet(
                        experiment_set_id=experiment_set_id,
                        experiment_id=experiment_id,
                        experiment_title=experiment_title,
                        record=record,
                        metadata=metadata,
                    )
                )
                name_index.extend((name, index) for name in names)
                for identifier in identifiers:
                    identifier_index.setdefault(identifier, set()).add(index)

    immutable_identifiers = {
        key: frozenset(indices)
        for key, indices in identifier_index.items()
    }
    return (
        snapshot_title,
        as_of,
        tuple(indexed_score_sets),
        tuple(name_index),
        immutable_identifiers,
    )


def _target_values(
    target: Mapping[str, Any],
    context: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Validate one target and return names, identifiers, and display label."""
    name = _optional_string(target, "name", context)
    mapped_name = _optional_string(target, "mappedHgncName", context)
    uniprot = _optional_string(
        target,
        "uniprotIdFromMappedMetadata",
        context,
    )
    accession_value = target.get("targetAccession")
    accession_gene: str | None = None
    accession: str | None = None
    if accession_value is not None:
        accession_context = f"{context}.targetAccession"
        accession_object = _mapping(accession_value, accession_context)
        accession_gene = _optional_string(
            accession_object,
            "gene",
            accession_context,
        )
        accession = _optional_string(
            accession_object,
            "accession",
            accession_context,
        )

    external_values = target.get("externalIdentifiers")
    external_identifiers: list[str] = []
    if external_values is not None:
        if not isinstance(external_values, list):
            raise CatalogError(
                f"{context}.externalIdentifiers must be a list."
            )
        for external_index, raw_external in enumerate(external_values):
            external_context = (
                f"{context}.externalIdentifiers[{external_index}]"
            )
            external = _mapping(raw_external, external_context)
            identifier_object = _mapping(
                external.get("identifier"),
                f"{external_context}.identifier",
            )
            external_identifiers.append(
                _required_string(
                    identifier_object,
                    "identifier",
                    f"{external_context}.identifier",
                )
            )

    names = tuple(
        value
        for value in (name, mapped_name, accession_gene)
        if value is not None
    )
    identifiers = tuple(
        value
        for value in (accession, uniprot, *external_identifiers)
        if value is not None
    )
    external_label = (
        min(
            external_identifiers,
            key=lambda value: (value.casefold(), value),
        )
        if external_identifiers
        else None
    )
    label = next(
        (
            value
            for value in (
                name,
                mapped_name,
                accession_gene,
                accession,
                uniprot,
                external_label,
            )
            if value is not None
        ),
        None,
    )
    return names, identifiers, label


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    """Return a JSON object or raise a contextual catalog error."""
    if not isinstance(value, Mapping):
        raise CatalogError(f"{context} must be a JSON object.")
    return value


def _list_field(
    value: Mapping[str, Any],
    field: str,
    context: str,
) -> list[Any]:
    """Return one required JSON list field."""
    field_value = value.get(field)
    if not isinstance(field_value, list):
        raise CatalogError(f"{context}.{field} must be a list.")
    return field_value


def _required_string(
    value: Mapping[str, Any],
    field: str,
    context: str,
) -> str:
    """Return one required non-empty string field."""
    return _standalone_required_string(value.get(field), f"{context}.{field}")


def _standalone_required_string(value: object, context: str) -> str:
    """Validate a required string value with contextual errors."""
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context} must be a non-empty string.")
    if value != value.strip():
        raise CatalogError(
            f"{context} must not contain surrounding whitespace."
        )
    return value


def _optional_string(
    value: Mapping[str, Any],
    field: str,
    context: str,
) -> str | None:
    """Validate and normalize one optional string field."""
    if field not in value or value[field] is None:
        return None
    field_value = value[field]
    if not isinstance(field_value, str):
        raise CatalogError(f"{context}.{field} must be a string or null.")
    return field_value.strip() or None


def _optional_non_negative_integer(
    value: Mapping[str, Any],
    field: str,
    context: str,
) -> int | None:
    """Validate one optional non-negative non-Boolean integer."""
    if field not in value or value[field] is None:
        return None
    field_value = value[field]
    if (
        isinstance(field_value, bool)
        or not isinstance(field_value, int)
        or field_value < 0
    ):
        raise CatalogError(
            f"{context}.{field} must be a non-negative integer or null."
        )
    return field_value


__all__ = [
    "MaveDBBulkCatalog",
    "MaveDBDiscoveredExperiment",
    "MaveDBDiscoveredScoreSet",
    "MaveDBDiscoveryResult",
]

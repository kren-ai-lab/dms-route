"""Internal wild-type resolution and provenance helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import pandas as pd

from dms_parser.exceptions import (
    InvalidDatasetError,
    InvalidPipelineOptionError,
    MissingWildTypeError,
    SequenceValidationError,
    WildTypeConflictError,
)
from dms_parser.validation import validate_wt_sequence

_WT_RESOLUTION_ATTR = "dms_parser_wt_resolution"
_SCORE_REL_TOL = 1e-9
_SCORE_ABS_TOL = 1e-12
_RELATIVE_METHODS = ("ratio", "log_ratio", "log2_ratio", "difference")
_PROTECTED_STANDARDIZED_COLUMNS = frozenset(
    {
        "dataset_id",
        "source",
        "protein_id",
        "gene",
        "uniprot_id",
        "wt_sequence",
        "variant",
        "parsed_variant",
        "parsed_position",
        "parsed_wt_aa",
        "parsed_mut_aa",
        "parsed_is_wildtype",
        "parsed_n_mutations",
        "parsed_mutations",
        "mutated_sequence",
        "is_wildtype",
        "is_synthetic",
        "n_mutations",
        "score_raw",
        "status",
        "error",
    }
)


@dataclass(frozen=True)
class WildTypeResolution:
    """Resolved WT values and compact provenance for one built dataset."""

    sequence: str
    score: float | None
    sequence_provenance: str
    score_provenance: str | None
    observed_wildtype_row: bool
    synthetic_wildtype_inserted: bool
    score_unavailable_reason: str | None


def resolve_wt_sequence(
    automatic: Iterable[tuple[str, str]],
    fallback: str | None,
    *,
    dataset_id: str,
) -> tuple[str, str]:
    """Resolve a WT sequence without allowing a fallback to replace evidence."""
    evidence = [
        (origin, _validated_sequence(value, origin=origin, option=False))
        for origin, value in automatic
        if isinstance(value, str) and value.strip()
    ]
    fallback_sequence = (
        _validated_sequence(fallback, origin="user_fallback", option=True)
        if fallback is not None
        else None
    )

    if evidence:
        origin, sequence = evidence[0]
        conflicting_origins = [
            other_origin
            for other_origin, other_sequence in evidence[1:]
            if other_sequence != sequence
        ]
        if conflicting_origins:
            origins = ", ".join([origin, *conflicting_origins])
            raise WildTypeConflictError(
                f"Conflicting WT sequences for {dataset_id!r} from {origins}."
            )
        if fallback_sequence is not None and fallback_sequence != sequence:
            raise WildTypeConflictError(
                f"WT sequence fallback conflicts with {origin} for "
                f"{dataset_id!r}."
            )
        return sequence, origin

    if fallback_sequence is not None:
        return fallback_sequence, "user_fallback"
    raise MissingWildTypeError(
        f"No reliable WT sequence is available for {dataset_id!r}; provide "
        "wt_sequence (CLI: --wt-sequence)."
    )


def resolve_wt_score(
    table: pd.DataFrame,
    fallback: float | None,
    *,
    dataset_id: str,
    automatic: Iterable[tuple[str, float]] = (),
    allow_ambiguous_observed_scores: bool = False,
) -> tuple[float | None, str | None, bool, str | None]:
    """Resolve an observed, metadata, or explicitly supplied WT score.

    An internal raw-build policy may mark ambiguous observed-only scores as
    unavailable while keeping strict resolution as the default.
    """
    observed = table["is_wildtype"].eq(True)
    if "status" in table.columns:
        observed &= table["status"].eq("OK")
    if "is_synthetic" in table.columns:
        observed &= ~table["is_synthetic"].eq(True)
    observed_exists = bool(observed.any())

    candidates: list[tuple[str, float]] = []
    observed_values = pd.to_numeric(
        table.loc[observed, "score_raw"],
        errors="coerce",
    ).dropna()
    for value in observed_values:
        candidates.append(("observed_wildtype_row", _finite_score(value, "observed WT score")))
    automatic_exists = False
    for origin, value in automatic:
        automatic_exists = True
        candidates.append((origin, _finite_score(value, f"WT score from {origin}")))

    resolved_score: float | None = None
    resolved_origin: str | None = None
    if candidates:
        resolved_origin, resolved_score = candidates[0]
        conflicts = [
            origin
            for origin, score in candidates[1:]
            if not math.isclose(
                score,
                resolved_score,
                rel_tol=_SCORE_REL_TOL,
                abs_tol=_SCORE_ABS_TOL,
            )
        ]
        if conflicts:
            if (
                allow_ambiguous_observed_scores
                and fallback is None
                and not automatic_exists
            ):
                return (
                    None,
                    None,
                    observed_exists,
                    "conflicting_observed_wildtype_scores",
                )
            origins = ", ".join(
                dict.fromkeys([resolved_origin, *conflicts])
            )
            raise WildTypeConflictError(
                f"Conflicting WT scores for {dataset_id!r} from {origins}."
            )

    fallback_score = (
        _finite_score(fallback, "wt_score", option=True)
        if fallback is not None
        else None
    )
    if resolved_score is not None:
        if fallback_score is not None and not math.isclose(
            fallback_score,
            resolved_score,
            rel_tol=_SCORE_REL_TOL,
            abs_tol=_SCORE_ABS_TOL,
        ):
            raise WildTypeConflictError(
                "WT score fallback conflicts with "
                f"{resolved_origin} for {dataset_id!r}."
            )
        return resolved_score, resolved_origin, observed_exists, None
    if fallback_score is not None:
        return fallback_score, "user_fallback", observed_exists, None
    return (
        None,
        None,
        observed_exists,
        "no_observed_or_metadata_wt_score_and_no_fallback",
    )


def validate_standardization_options(
    *,
    wt_sequence: str | None,
    wt_score: float | None,
    add_relative_score: bool,
    relative_method: str,
    relative_output_col: str,
    add_binary_label: bool,
    delta: float,
    higher_is_better: bool,
    binary_output_col: str,
) -> None:
    """Validate public standardization options before acquisition side effects."""
    if wt_sequence is not None:
        _validated_sequence(wt_sequence, origin="user_fallback", option=True)
    if wt_score is not None:
        _finite_score(wt_score, "wt_score", option=True)
    for name, value in (
        ("add_relative_score", add_relative_score),
        ("add_binary_label", add_binary_label),
        ("higher_is_better", higher_is_better),
    ):
        if not isinstance(value, bool):
            raise InvalidPipelineOptionError(f"{name} must be a boolean.")
    if relative_method not in _RELATIVE_METHODS:
        choices = ", ".join(_RELATIVE_METHODS)
        raise InvalidPipelineOptionError(
            f"relative_method must be one of: {choices}."
        )
    for name, value in (
        ("relative_output_col", relative_output_col),
        ("binary_output_col", binary_output_col),
    ):
        if not isinstance(value, str) or not value.strip():
            raise InvalidPipelineOptionError(f"{name} must be a non-empty string.")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise InvalidPipelineOptionError("delta must be a finite non-negative number.")
    if not math.isfinite(float(delta)) or float(delta) < 0:
        raise InvalidPipelineOptionError("delta must be a finite non-negative number.")
    if add_binary_label and not add_relative_score:
        raise InvalidPipelineOptionError(
            "add_binary_label=True requires add_relative_score=True."
        )
    _validate_generated_output_columns(
        add_relative_score=add_relative_score,
        relative_output_col=relative_output_col,
        add_binary_label=add_binary_label,
        binary_output_col=binary_output_col,
    )


def set_wt_resolution(table: pd.DataFrame, resolution: WildTypeResolution) -> None:
    """Attach one internal immutable WT resolution to a built table."""
    table.attrs[_WT_RESOLUTION_ATTR] = asdict(resolution)


def get_wt_resolution(table: pd.DataFrame) -> WildTypeResolution:
    """Return the internal WT resolution attached by a builder."""
    values = table.attrs.get(_WT_RESOLUTION_ATTR)
    if not isinstance(values, dict):
        raise InvalidDatasetError("Built dataset is missing WT resolution metadata.")
    return WildTypeResolution(**values)


def replace_wt_sequence_provenance(
    table: pd.DataFrame,
    provenance: str,
) -> WildTypeResolution:
    """Replace builder-local sequence provenance with source provenance."""
    resolution = replace(
        get_wt_resolution(table),
        sequence_provenance=provenance,
    )
    set_wt_resolution(table, resolution)
    return resolution


def _validate_generated_output_columns(
    *,
    add_relative_score: bool,
    relative_output_col: str,
    add_binary_label: bool,
    binary_output_col: str,
) -> None:
    """Reject generated columns that would overwrite standardized structure."""
    active_columns: list[tuple[str, str]] = []
    if add_relative_score:
        active_columns.append(("relative_output_col", relative_output_col))
    if add_binary_label:
        active_columns.append(("binary_output_col", binary_output_col))

    for option_name, column in active_columns:
        if column in _PROTECTED_STANDARDIZED_COLUMNS:
            raise InvalidPipelineOptionError(
                f"{option_name} cannot overwrite protected standardized "
                f"column {column!r}."
            )
    if (
        add_relative_score
        and add_binary_label
        and relative_output_col == binary_output_col
    ):
        raise InvalidPipelineOptionError(
            "relative_output_col and binary_output_col must be different "
            "when both outputs are enabled."
        )


def _validated_sequence(
    value: str,
    *,
    origin: str,
    option: bool,
) -> str:
    error_type = InvalidPipelineOptionError if option else InvalidDatasetError
    if not isinstance(value, str) or not value.strip():
        raise error_type(
            f"WT sequence from {origin} must be a non-empty protein sequence."
        )
    sequence = value.strip().upper()
    try:
        validate_wt_sequence(sequence)
    except SequenceValidationError as exc:
        raise error_type(
            f"WT sequence from {origin} is invalid: {exc}"
        ) from exc
    return sequence


def _finite_score(value: Any, field: str, *, option: bool = False) -> float:
    error_type = InvalidPipelineOptionError if option else InvalidDatasetError
    if isinstance(value, bool):
        raise error_type(f"{field} must be a finite number.")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{field} must be a finite number."
        ) from exc
    if not math.isfinite(score):
        raise error_type(f"{field} must be a finite number.")
    return score

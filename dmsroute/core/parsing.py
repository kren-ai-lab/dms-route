"""Parsing utilities for DMS datasets."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from dmsroute.core.constants import (
    AA3_TO_AA1,
    DNA_CODON_TABLE,
    HGVS_BRACKET_RE,
    HGVS_MUT_RE,
    HGVS_SINGLE_RE,
    INDEL_TOKENS_PRO,
    MULTI_VARIANT_SPLIT_PATTERN,
    SINGLE_VARIANT_PATTERN,
    VALID_RESIDUES,
    WILDTYPE_TOKENS,
)
from dmsroute.core.exceptions import (
    InvalidHGVSVariantError,
    InvalidVariantError,
    MutationApplicationError,
    SequenceValidationError,
    UnsupportedVariantError,
)
from dmsroute.core.types import (
    MutationTuple,
    ParsedMaveDBHGVSRecord,
    ParsedProteinEdit,
    ParsedVariant,
    ParsedVariantToken,
    ProteinDeletionEdit,
    ProteinEdit,
    ProteinInsertionEdit,
    ProteinSubstitutionEdit,
)

_HGVS_SINGLE_DELETION_RE = re.compile(
    r"^p\.(?P<wt>[A-Z][a-z]{2})(?P<pos>\d+)del$"
)
_HGVS_SINGLE_INSERTION_RE = re.compile(
    r"^p\.(?P<left>[A-Z][a-z]{2})(?P<left_pos>\d+)_"
    r"(?P<right>[A-Z][a-z]{2})(?P<right_pos>\d+)"
    r"ins(?P<inserted>[A-Z][a-z]{2})$"
)


def _normalize_variant_string(variant: Any) -> str:
    """Normalize a raw variant value into a clean string."""
    if variant is None:
        return ""

    if pd.isna(variant):
        return ""

    normalized = str(variant).strip()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _normalize_wildtype_token(token: str) -> str:
    """Normalize a token for wild-type comparisons."""
    return token.strip().lower().replace("-", "_")


def read_fasta_one(path: str) -> tuple[str, str]:
    """Read a FASTA file containing a single sequence."""
    header = None
    seq_chunks: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise SequenceValidationError(
                        "FASTA has more than one record. Provide a single WT sequence FASTA."
                    )
                header = line[1:].strip()
            else:
                seq_chunks.append(line)

    if header is None:
        raise SequenceValidationError("No FASTA header found.")

    seq = "".join(seq_chunks).replace(" ", "").upper()
    if not seq:
        raise SequenceValidationError("Empty FASTA sequence.")

    return header, seq


def translate_dna(
    dna_sequence: str,
    frame: int = 1,
    stop_at_stop: bool = True,
) -> str:
    """Translate a DNA sequence into protein."""
    if frame not in {1, 2, 3}:
        raise SequenceValidationError("frame must be one of {1, 2, 3}")

    seq = re.sub(r"\s+", "", str(dna_sequence).upper())
    seq = seq.replace("U", "T")

    if not seq:
        raise SequenceValidationError("DNA sequence is empty.")

    protein: list[str] = []
    start = frame - 1

    for i in range(start, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        aa = DNA_CODON_TABLE.get(codon, "X")
        if aa == "*" and stop_at_stop:
            break
        protein.append(aa)

    return "".join(protein)


def is_wildtype_variant(variant: Any) -> bool:
    """Return whether a variant string represents the wild type."""
    normalized = _normalize_variant_string(variant)
    if normalized == "":
        return True

    normalized_reference = {_normalize_wildtype_token(token) for token in WILDTYPE_TOKENS}
    return _normalize_wildtype_token(normalized) in normalized_reference


def _split_variant_tokens(variant: str) -> list[str]:
    """Split a variant string into mutation tokens."""
    if not variant or is_wildtype_variant(variant):
        return []

    return [token for token in MULTI_VARIANT_SPLIT_PATTERN.split(variant) if token]


def parse_variant_token(token: str, *, strict: bool = True) -> ParsedVariantToken:
    """Parse a single substitution token such as A23V."""
    raw_token = _normalize_variant_string(token)

    if raw_token == "":
        message = "Empty mutation token cannot be parsed."
        if strict:
            raise InvalidVariantError(message)
        return {"token": raw_token, "wt_aa": None, "position": None, "mut_aa": None}

    match = SINGLE_VARIANT_PATTERN.match(raw_token)
    if match is None:
        message = f"Invalid substitution token: {token!r}"
        if strict:
            raise InvalidVariantError(message)
        return {"token": raw_token, "wt_aa": None, "position": None, "mut_aa": None}

    wt_aa = match.group("wt").upper()
    mut_aa = match.group("mut").upper()
    position = int(match.group("position"))

    if wt_aa not in VALID_RESIDUES or mut_aa not in VALID_RESIDUES:
        message = f"Unsupported residue token in variant: {token!r}"
        if strict:
            raise InvalidVariantError(message)
        return {"token": raw_token, "wt_aa": None, "position": None, "mut_aa": None}

    return {
        "token": raw_token,
        "wt_aa": wt_aa,
        "position": position,
        "mut_aa": mut_aa,
    }


def parse_variant(variant: Any, *, strict: bool = True) -> ParsedVariant:
    """Parse an internal protein variant string."""
    normalized = _normalize_variant_string(variant)

    if is_wildtype_variant(normalized):
        return {
            "variant": normalized,
            "is_wildtype": True,
            "n_mutations": 0,
            "mutations": [],
            "position": None,
            "wt_aa": None,
            "mut_aa": None,
        }

    tokens = _split_variant_tokens(normalized)
    mutations = [parse_variant_token(token, strict=strict) for token in tokens]

    if len(mutations) == 1:
        single = mutations[0]
        position = single["position"]
        wt_aa = single["wt_aa"]
        mut_aa = single["mut_aa"]
    else:
        position = None
        wt_aa = None
        mut_aa = None

    return {
        "variant": normalized,
        "is_wildtype": False,
        "n_mutations": len(tokens),
        "mutations": mutations,
        "position": position,
        "wt_aa": wt_aa,
        "mut_aa": mut_aa,
    }


def parse_variant_series(
    series: pd.Series,
    *,
    strict: bool = True,
    prefix: str = "",
) -> pd.DataFrame:
    """Parse a full pandas Series of internal variant strings."""
    parsed_records = series.apply(lambda value: parse_variant(value, strict=strict))
    df = pd.DataFrame(parsed_records.tolist(), index=series.index)

    rename_map = {
        "variant": f"{prefix}variant",
        "is_wildtype": f"{prefix}is_wildtype",
        "n_mutations": f"{prefix}n_mutations",
        "position": f"{prefix}position",
        "wt_aa": f"{prefix}wt_aa",
        "mut_aa": f"{prefix}mut_aa",
        "mutations": f"{prefix}mutations",
    }
    return df.rename(columns=rename_map)


def count_mutations(variant: Any) -> int:
    """Count the number of mutations encoded in a variant string."""
    normalized = _normalize_variant_string(variant)
    if is_wildtype_variant(normalized):
        return 0
    return len(_split_variant_tokens(normalized))


def explode_mutations(
    df: pd.DataFrame,
    mutations_col: str = "mutations",
) -> pd.DataFrame:
    """Explode a parsed mutations column into one row per mutation token."""
    if mutations_col not in df.columns:
        raise KeyError(f"Column {mutations_col!r} not found in DataFrame.")

    exploded = df.copy().explode(mutations_col, ignore_index=False)

    for field in ["token", "wt_aa", "position", "mut_aa"]:
        exploded[field] = exploded[mutations_col].apply(
            lambda value: value.get(field) if isinstance(value, dict) else None
        )

    return exploded


def extract_positions(variant: Any, *, strict: bool = True) -> list[int]:
    """Extract all mutated positions from a variant string."""
    parsed = parse_variant(variant, strict=strict)
    return [
        mutation["position"]
        for mutation in parsed["mutations"]
        if mutation.get("position") is not None
    ]


def extract_mutation_tokens(variant: Any) -> list[str]:
    """Extract raw mutation tokens from a variant string."""
    normalized = _normalize_variant_string(variant)
    return _split_variant_tokens(normalized)


def hgvs_pro_is_indel(hgvs_pro: str) -> bool:
    """Return whether HGVS suggests an indel or related syntax."""
    value = str(hgvs_pro).strip()
    return any(token in value for token in INDEL_TOKENS_PRO)


def _parse_hgvs_mut_token(token: str, hgvs_pro: str) -> MutationTuple:
    """Parse a single HGVS protein mutation token into one-letter notation."""
    token = token.strip()
    match = HGVS_MUT_RE.match(token)
    if not match:
        raise InvalidHGVSVariantError(f"Unsupported mutation token: {token} in {hgvs_pro}")

    wt3 = match.group("wt")
    pos = int(match.group("pos"))
    mut3 = match.group("mut")

    if wt3 not in AA3_TO_AA1:
        raise InvalidHGVSVariantError(f"Unknown WT amino acid code: {wt3} in {token}")
    wt1 = AA3_TO_AA1[wt3]

    if mut3 == "=":
        mut1 = wt1
    elif mut3 in {"Ter", "*"}:
        mut1 = "*"
    else:
        if mut3 not in AA3_TO_AA1:
            raise InvalidHGVSVariantError(f"Unknown MUT amino acid code: {mut3} in {token}")
        mut1 = AA3_TO_AA1[mut3]

    return wt1, pos, mut1


def _parse_indel_residue(code: str, hgvs_pro: str) -> str:
    """Return one supported amino-acid code for a bounded protein indel."""
    residue = AA3_TO_AA1.get(code)
    if residue is None or residue not in VALID_RESIDUES:
        raise InvalidHGVSVariantError(
            f"Unknown amino acid code: {code} in {hgvs_pro}"
        )
    return residue


def _parse_hgvs_pro_edits(hgvs_pro: str) -> list[ProteinEdit]:
    """Parse supported protein HGVS into structured internal edits."""
    hgvs_pro = str(hgvs_pro).strip()

    if hgvs_pro == "p.=":
        return []

    deletion_match = _HGVS_SINGLE_DELETION_RE.fullmatch(hgvs_pro)
    if deletion_match:
        return [
            ProteinDeletionEdit(
                wt_aa=_parse_indel_residue(
                    deletion_match.group("wt"),
                    hgvs_pro,
                ),
                position=int(deletion_match.group("pos")),
            )
        ]

    insertion_match = _HGVS_SINGLE_INSERTION_RE.fullmatch(hgvs_pro)
    if insertion_match:
        return [
            ProteinInsertionEdit(
                left_aa=_parse_indel_residue(
                    insertion_match.group("left"),
                    hgvs_pro,
                ),
                left_position=int(insertion_match.group("left_pos")),
                right_aa=_parse_indel_residue(
                    insertion_match.group("right"),
                    hgvs_pro,
                ),
                right_position=int(insertion_match.group("right_pos")),
                inserted_aa=_parse_indel_residue(
                    insertion_match.group("inserted"),
                    hgvs_pro,
                ),
            )
        ]

    if hgvs_pro_is_indel(hgvs_pro):
        raise UnsupportedVariantError(
            "Unsupported hgvs_pro (complex protein indel/frameshift/etc.)."
        )

    bracket_match = HGVS_BRACKET_RE.match(hgvs_pro)
    if bracket_match:
        body = bracket_match.group("body").strip()
        parts = [part.strip() for part in body.split(";") if part.strip()]
        if not parts:
            raise InvalidHGVSVariantError(f"Empty hgvs_pro body: {hgvs_pro}")
        return [
            ProteinSubstitutionEdit(*_parse_hgvs_mut_token(part, hgvs_pro))
            for part in parts
            if part != "="
        ]

    single_match = HGVS_SINGLE_RE.match(hgvs_pro)
    if single_match:
        token = single_match.group("single").strip()
        return [ProteinSubstitutionEdit(*_parse_hgvs_mut_token(token, hgvs_pro))]

    raise InvalidHGVSVariantError(f"Unsupported hgvs_pro format: {hgvs_pro}")


def parse_hgvs_pro(hgvs_pro: str) -> list[ParsedProteinEdit]:
    """Parse protein HGVS while preserving substitution tuple results."""
    edits = _parse_hgvs_pro_edits(hgvs_pro)
    return [
        (edit.wt_aa, edit.position, edit.mut_aa)
        if isinstance(edit, ProteinSubstitutionEdit)
        else edit
        for edit in edits
    ]


def apply_mutations(
    wt_seq: str,
    muts: list[MutationTuple],
) -> tuple[str, str]:
    """Apply a list of substitutions to a wild-type protein sequence."""
    seq_list = list(str(wt_seq).upper())
    n = len(seq_list)

    seen_positions: set[int] = set()
    for wt_aa, pos, _ in muts:
        if pos in seen_positions:
            raise MutationApplicationError(
                f"Duplicate position {pos} in mutation list (conflicting substitutions)."
            )
        seen_positions.add(pos)

        if pos < 1 or pos > n:
            raise MutationApplicationError(
                f"Position {pos} out of range for WT length {n}."
            )

        wt_at_pos = seq_list[pos - 1]
        if wt_at_pos != wt_aa:
            raise MutationApplicationError(
                f"WT mismatch at pos {pos}: mutation expects {wt_aa}, WT has {wt_at_pos}."
            )

    variant_tokens: list[str] = []
    for wt_aa, pos, mut_aa in muts:
        seq_list[pos - 1] = mut_aa
        variant_tokens.append(f"{wt_aa}{pos}{mut_aa}")

    return "".join(seq_list), ";".join(variant_tokens)


def variant_to_sequence(
    wt_seq: str,
    variant: Any,
    *,
    strict: bool = True,
) -> tuple[str, str]:
    """Apply an internal variant string to a WT sequence."""
    parsed = parse_variant(variant, strict=strict)

    if parsed["is_wildtype"]:
        return str(wt_seq).upper(), ""

    muts: list[MutationTuple] = [
        (mutation["wt_aa"], mutation["position"], mutation["mut_aa"])
        for mutation in parsed["mutations"]
    ]
    return apply_mutations(str(wt_seq).upper(), muts)


def hgvs_to_sequence(
    wt_seq: str,
    hgvs_pro: str,
) -> tuple[str, str]:
    """Convert a protein HGVS variant into mutated sequence and internal notation."""
    edits = _parse_hgvs_pro_edits(hgvs_pro)
    substitutions = [
        (edit.wt_aa, edit.position, edit.mut_aa)
        for edit in edits
        if isinstance(edit, ProteinSubstitutionEdit)
    ]
    if len(substitutions) == len(edits):
        return apply_mutations(wt_seq, substitutions)
    if len(edits) != 1:
        raise UnsupportedVariantError(
            "Protein indels cannot be combined with other edits."
        )

    sequence = str(wt_seq).upper()
    edit = edits[0]
    if isinstance(edit, ProteinDeletionEdit):
        _validate_edit_reference(
            sequence,
            edit.wt_aa,
            edit.position,
            label="deletion",
        )
        return (
            sequence[: edit.position - 1] + sequence[edit.position :],
            f"{edit.wt_aa}{edit.position}del",
        )
    if isinstance(edit, ProteinInsertionEdit):
        if edit.right_position != edit.left_position + 1:
            raise MutationApplicationError(
                "Insertion anchors must identify adjacent WT positions."
            )
        _validate_edit_reference(
            sequence,
            edit.left_aa,
            edit.left_position,
            label="left insertion anchor",
        )
        _validate_edit_reference(
            sequence,
            edit.right_aa,
            edit.right_position,
            label="right insertion anchor",
        )
        return (
            sequence[: edit.left_position]
            + edit.inserted_aa
            + sequence[edit.left_position :],
            f"{edit.left_aa}{edit.left_position}_"
            f"{edit.right_aa}{edit.right_position}ins{edit.inserted_aa}",
        )
    raise UnsupportedVariantError("Unsupported protein edit.")


def _validate_edit_reference(
    wt_seq: str,
    expected_aa: str,
    position: int,
    *,
    label: str,
) -> None:
    """Validate one referenced residue before applying a protein edit."""
    if position < 1 or position > len(wt_seq):
        raise MutationApplicationError(
            f"{label.capitalize()} position {position} out of range for "
            f"WT length {len(wt_seq)}."
        )
    observed_aa = wt_seq[position - 1]
    if observed_aa != expected_aa:
        raise MutationApplicationError(
            f"WT mismatch at pos {position}: {label} expects {expected_aa}, "
            f"WT has {observed_aa}."
        )


def parse_mavedb_hgvs_series(
    series: pd.Series,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Parse a Series of MaveDB hgvs_pro strings."""
    records: list[ParsedMaveDBHGVSRecord] = []

    for value in series:
        hgvs_pro = str(value)
        out: ParsedMaveDBHGVSRecord = {
            "hgvs_pro": hgvs_pro,
            "status": "OK",
            "error": "",
            "mutations": None,
        }

        try:
            out["mutations"] = parse_hgvs_pro(hgvs_pro)
        except UnsupportedVariantError as exc:
            out["status"] = "Unsupported"
            out["error"] = str(exc)
        except Exception as exc:
            if strict:
                raise
            out["status"] = "Error"
            out["error"] = str(exc)

        records.append(out)

    return pd.DataFrame.from_records(records, index=series.index)

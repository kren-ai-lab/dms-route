"""Shared type definitions for dmsroute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, TypedDict


MutationTuple = tuple[str, int, str]


@dataclass(frozen=True)
class ProteinSubstitutionEdit:
    """Internal structured representation of one protein substitution."""

    wt_aa: str
    position: int
    mut_aa: str
    kind: Literal["substitution"] = "substitution"


@dataclass(frozen=True)
class ProteinDeletionEdit:
    """Internal structured representation of one residue deletion."""

    wt_aa: str
    position: int
    kind: Literal["deletion"] = "deletion"


@dataclass(frozen=True)
class ProteinInsertionEdit:
    """Internal structured representation of one residue insertion."""

    left_aa: str
    left_position: int
    right_aa: str
    right_position: int
    inserted_aa: str
    kind: Literal["insertion"] = "insertion"


ProteinEdit: TypeAlias = (
    ProteinSubstitutionEdit | ProteinDeletionEdit | ProteinInsertionEdit
)
ParsedProteinEdit: TypeAlias = MutationTuple | ProteinDeletionEdit | ProteinInsertionEdit


class ParsedVariantToken(TypedDict):
    """Parsed representation of a single substitution token."""
    token: str
    wt_aa: Optional[str]
    position: Optional[int]
    mut_aa: Optional[str]


class ParsedVariant(TypedDict):
    """Parsed representation of an internal variant string."""
    variant: str
    is_wildtype: bool
    n_mutations: int
    mutations: list[ParsedVariantToken]
    position: Optional[int]
    wt_aa: Optional[str]
    mut_aa: Optional[str]


class SequenceBuildResult(TypedDict):
    """Result of reconstructing a mutated sequence from WT + variant."""
    variant: Optional[str]
    mutated_sequence: Optional[str]
    status: Literal["OK", "Unsupported", "Error"]
    error: str
    is_wildtype: bool
    n_mutations: Optional[int]


class ParsedMaveDBHGVSRecord(TypedDict):
    """Parsed representation of a MaveDB hgvs_pro entry."""
    hgvs_pro: str
    status: Literal["OK", "Unsupported", "Error"]
    error: str
    mutations: Optional[list[ParsedProteinEdit]]

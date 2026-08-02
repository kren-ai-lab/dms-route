"""Shared type definitions for dms_parser."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict


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
    mutations: Optional[list[tuple[str, int, str]]]


MutationTuple = tuple[str, int, str]

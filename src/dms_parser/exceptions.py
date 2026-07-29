"""Custom exceptions for the dms_parser package."""

from __future__ import annotations


class DMSParserError(Exception):
    """Base exception for the dms_parser package."""


class InvalidDatasetError(DMSParserError):
    """Raised when a dataset does not meet the expected schema or content requirements."""


class MissingWildTypeError(DMSParserError):
    """Raised when no wild-type reference can be identified."""


class InvalidVariantError(DMSParserError):
    """Raised when a variant string cannot be parsed."""


class InvalidHGVSVariantError(InvalidVariantError):
    """Raised when an HGVS protein variant cannot be parsed."""


class UnsupportedVariantError(InvalidVariantError):
    """Raised when a variant is syntactically valid but unsupported by the current workflow."""


class SequenceValidationError(DMSParserError):
    """Raised when a protein or nucleotide sequence is invalid."""


class MutationApplicationError(DMSParserError):
    """Raised when mutations cannot be applied to the wild-type sequence."""


class DownloadError(DMSParserError):
    """Raised when a remote dataset cannot be downloaded."""


class FileFormatError(DMSParserError):
    """Raised when an input file format is unsupported or malformed."""


class CacheError(DMSParserError):
    """Base exception for filesystem cache failures."""


class InvalidCacheEntryError(CacheError):
    """Raised when a cached artifact is missing, invalid, or corrupted."""


class CorruptCacheManifestError(InvalidCacheEntryError):
    """Raised when a cache manifest is malformed or inconsistent."""

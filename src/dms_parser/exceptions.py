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


class CatalogError(DMSParserError):
    """Base exception for dataset catalog failures."""


class InvalidCatalogQueryError(CatalogError, ValueError):
    """Raised when catalog arguments are invalid or unsupported."""


class DatasetNotFoundError(CatalogError, LookupError):
    """Raised when a requested dataset is absent from a source catalog."""


class SourceConfigurationError(DMSParserError, ValueError):
    """Raised when source selection or configuration is invalid."""


class UnknownSourceResourceError(SourceConfigurationError, LookupError):
    """Raised when a requested source resource is not registered."""


class UnsupportedSourceResourceError(SourceConfigurationError):
    """Raised when a known resource cannot be processed by the current package."""


class PipelineError(DMSParserError):
    """Base exception for pipeline orchestration failures."""


class InvalidPipelineOptionError(PipelineError, ValueError):
    """Raised when a programmatic pipeline option is invalid."""

from typing import Any


def parse_variant(variant: str) -> dict[str, Any]:
    """Parse a single amino-acid substitution string such as A23V."""
    raise NotImplementedError


def parse_variant_series():
    """Parse a full pandas Series of variant strings."""
    raise NotImplementedError


def count_mutations(variant: str) -> int:
    """Count the number of mutations encoded in a variant string."""
    raise NotImplementedError


def is_wildtype_variant(variant: str) -> bool:
    """Return True if the provided variant represents the wild type."""
    raise NotImplementedError
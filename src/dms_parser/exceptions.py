class DMSParserError(Exception):
    """Base exception for the dms_parser package."""


class MissingWildTypeError(DMSParserError):
    """Raised when no wild-type reference can be identified."""


class InvalidVariantError(DMSParserError):
    """Raised when a variant string cannot be parsed."""


class InvalidDatasetError(DMSParserError):
    """Raised when the dataset does not meet the minimum schema requirements."""
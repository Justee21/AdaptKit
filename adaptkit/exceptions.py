class AdaptKitError(Exception):
    """Base exception for AdaptKit."""


class ConfigurationError(AdaptKitError):
    """Raised when an AdaptKit component is configured incorrectly."""


class ValidationError(AdaptKitError, ValueError):
    """Raised when public API input or evaluator output is invalid."""

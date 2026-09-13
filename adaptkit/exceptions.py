class AdaptKitError(Exception):
    """Base exception for AdaptKit."""


class ConfigurationError(AdaptKitError):
    """Raised when an AdaptKit component is configured incorrectly."""


class ValidationError(AdaptKitError, ValueError):
    """Raised when public API input or evaluator output is invalid."""


class StorageError(AdaptKitError):
    """Base exception for persistence failures."""


class StorageBusyError(StorageError):
    """Raised when a storage write lock cannot be acquired in time."""


class UnsupportedSchemaVersionError(StorageError):
    """Raised when persisted state uses an unsupported schema version."""


class DecisionNotFoundError(StorageError):
    """Raised when a referenced decision does not exist."""


class DecisionMismatchError(StorageError):
    """Raised when a decision does not match its persisted record."""

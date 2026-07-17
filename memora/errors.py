"""Project-specific exceptions for Memora."""


class MemoraError(Exception):
    """Base exception for Memora errors."""


class MemoryNotFoundError(MemoraError):
    """Raised when a memory cannot be found."""


class SessionNotFoundError(MemoraError):
    """Raised when a session cannot be found."""


class MemoryValidationError(MemoraError):
    """Raised when memory data is malformed or unsupported."""


class MemoryPolicyError(MemoraError):
    """Raised when a policy decision blocks an operation."""

"""Validation package for NeuroADHD_RawMRI."""
from src.validation.authorization_gate import (
    AuthorizationBypassAttemptError,
    AuthorizationDeniedError,
    AuthorizationError,
    AuthorizationSchemaError,
    ResearchAuthorizationGate,
)
from src.validation.bids_validator import BIDSInputValidator
from src.validation.memory_guard import (
    DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
    get_cgroup_memory_stats,
    get_process_rss_mb,
    verify_os_memory_isolation,
)

__all__ = [
    "BIDSInputValidator",
    "get_cgroup_memory_stats",
    "get_process_rss_mb",
    "verify_os_memory_isolation",
    "DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES",
    "ResearchAuthorizationGate",
    "AuthorizationError",
    "AuthorizationDeniedError",
    "AuthorizationSchemaError",
    "AuthorizationBypassAttemptError",
]


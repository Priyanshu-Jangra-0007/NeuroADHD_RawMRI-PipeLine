"""Operating system memory enforcement and cgroup monitoring module.

Provides dual-layer memory safety:
1. Operating-system-level cgroup memory limit verification (cgroup v2 & v1).
2. Process-level resident set size (RSS) tracking and peak recording.

Enforces a hard ceiling of 2560 MiB (2.5 GB) inside containerized deployment
environments and fails closed if limits are unconfigured or exceeded.
"""
from __future__ import annotations

import os
from pathlib import Path
import resource
import sys
from typing import Any, Dict, Optional, Tuple

CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
CGROUP_V2_PEAK = Path("/sys/fs/cgroup/memory.peak")

CGROUP_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
CGROUP_V1_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
CGROUP_V1_MAX_USAGE = Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")

# Default hard limit: 2560 MiB = 2.5 GiB = 2,684,354,560 bytes
DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES = 2560 * 1024 * 1024


def get_cgroup_memory_stats() -> Dict[str, Any]:
    """Read kernel-level memory parameters from cgroup filesystem.

    Supports cgroup v2 (unified) and cgroup v1 (legacy).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - "cgroup_version": Optional[int] (1, 2, or None)
        - "memory_max_bytes": Optional[int] (hard ceiling in bytes, None if unconstrained)
        - "memory_current_bytes": Optional[int] (current usage)
        - "memory_peak_bytes": Optional[int] (peak usage recorded by kernel)
        - "memory_max_mib": Optional[float]
        - "memory_peak_mib": Optional[float]
    """
    # 1. Try cgroup v2 (standard on modern Linux containers and Docker Desktop)
    if CGROUP_V2_MAX.is_file():
        try:
            val_str = CGROUP_V2_MAX.read_text().strip()
            max_bytes: Optional[int] = None if val_str == "max" else int(val_str)

            cur_bytes: Optional[int] = None
            if CGROUP_V2_CURRENT.is_file():
                try:
                    cur_bytes = int(CGROUP_V2_CURRENT.read_text().strip())
                except (ValueError, OSError):
                    pass

            peak_bytes: Optional[int] = None
            if CGROUP_V2_PEAK.is_file():
                try:
                    peak_bytes = int(CGROUP_V2_PEAK.read_text().strip())
                except (ValueError, OSError):
                    pass

            return {
                "cgroup_version": 2,
                "memory_max_bytes": max_bytes,
                "memory_current_bytes": cur_bytes,
                "memory_peak_bytes": peak_bytes,
                "memory_max_mib": round(max_bytes / (1024 * 1024), 2) if max_bytes is not None else None,
                "memory_peak_mib": round(peak_bytes / (1024 * 1024), 2) if peak_bytes is not None else None,
            }
        except (ValueError, OSError):
            pass

    # 2. Try cgroup v1 (legacy Linux distributions)
    if CGROUP_V1_LIMIT.is_file():
        try:
            val_bytes = int(CGROUP_V1_LIMIT.read_text().strip())
            # cgroup v1 reports ~2^63 - 1 or very large number when unlimited
            max_bytes = None if val_bytes > (1 << 60) else val_bytes

            cur_bytes = None
            if CGROUP_V1_USAGE.is_file():
                try:
                    cur_bytes = int(CGROUP_V1_USAGE.read_text().strip())
                except (ValueError, OSError):
                    pass

            peak_bytes = None
            if CGROUP_V1_MAX_USAGE.is_file():
                try:
                    peak_bytes = int(CGROUP_V1_MAX_USAGE.read_text().strip())
                except (ValueError, OSError):
                    pass

            return {
                "cgroup_version": 1,
                "memory_max_bytes": max_bytes,
                "memory_current_bytes": cur_bytes,
                "memory_peak_bytes": peak_bytes,
                "memory_max_mib": round(max_bytes / (1024 * 1024), 2) if max_bytes is not None else None,
                "memory_peak_mib": round(peak_bytes / (1024 * 1024), 2) if peak_bytes is not None else None,
            }
        except (ValueError, OSError):
            pass

    # 3. Running outside cgroup (e.g., bare metal macOS or unconstrained host)
    return {
        "cgroup_version": None,
        "memory_max_bytes": None,
        "memory_current_bytes": None,
        "memory_peak_bytes": None,
        "memory_max_mib": None,
        "memory_peak_mib": None,
    }


def get_process_rss_mb() -> float:
    """Return process peak resident set size (RSS) in megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes, Linux reports in kilobytes
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024.0 * 1024.0)
    return rusage.ru_maxrss / 1024.0


def verify_os_memory_isolation(
    max_allowed_bytes: int = DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
    require_enforcement: bool = False,
) -> Dict[str, Any]:
    """Verify kernel-enforced cgroup memory limit against the required 2560 MiB ceiling.

    Parameters
    ----------
    max_allowed_bytes : int
        Maximum permitted memory ceiling in bytes (default: 2560 MiB = 2.5 GiB).
    require_enforcement : bool
        If True, raises RuntimeError if kernel cgroup limit is absent, unlimited,
        or exceeds max_allowed_bytes (strict fail-closed container deployment mode).

    Returns
    -------
    Dict[str, Any]
        Verification status and metrics.

    Raises
    ------
    RuntimeError
        If require_enforcement is True and memory isolation is absent or misconfigured.
    """
    stats = get_cgroup_memory_stats()
    max_bytes = stats["memory_max_bytes"]

    if max_bytes is None:
        if require_enforcement:
            raise RuntimeError(
                "MEMORY ISOLATION FAILURE: Operating-system cgroup memory limit is not enforced. "
                "Container must be started with hard limit of 2560 MiB (--memory=2560m --memory-swap=2560m). "
                f"Observed cgroup stats: {stats}."
            )
        return {
            "enforced": False,
            "status": "UNCONSTRAINED_HOST",
            "reason": "No cgroup memory limit detected (running on host or unconstrained environment)",
            "stats": stats,
            "process_rss_mb": round(get_process_rss_mb(), 2),
        }

    max_mib = max_bytes / (1024 * 1024)
    allowed_mib = max_allowed_bytes / (1024 * 1024)

    # Permit a small 1 MiB alignment tolerance if kernel rounds to pages
    if max_bytes > (max_allowed_bytes + 1024 * 1024):
        raise RuntimeError(
            f"MEMORY ISOLATION FAILURE: Container memory ceiling ({max_mib:.1f} MiB) "
            f"exceeds the maximum permitted ceiling ({allowed_mib:.1f} MiB). "
            "Pipeline requires container isolation of at most 2560 MiB."
        )

    return {
        "enforced": True,
        "status": "ENFORCED_OK",
        "cgroup_version": stats["cgroup_version"],
        "memory_max_bytes": max_bytes,
        "memory_max_mib": max_mib,
        "memory_current_bytes": stats["memory_current_bytes"],
        "memory_peak_bytes": stats["memory_peak_bytes"],
        "stats": stats,
        "process_rss_mb": round(get_process_rss_mb(), 2),
    }


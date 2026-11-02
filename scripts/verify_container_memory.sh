#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NeuroADHD RawMRI PoC V2 — OS Memory Isolation Verification Script
#
# Validates:
# 1. Docker cgroup v2/v1 memory limit configuration (2560 MiB / 2684354560 bytes)
# 2. Linux kernel Out-Of-Memory (OOM) killer enforcement via constrained container
# 3. Fail-closed behavior of run_poc_v2.py when OS isolation is missing
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================================================="
echo "OS Cgroup Memory Isolation Verification"
echo "==========================================================================="

# Check docker availability
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker CLI not found in PATH." >&2
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running or socket is inaccessible." >&2
    exit 1
fi

echo "[Test 1/3] Verifying 2560 MiB container cgroup memory limit..."
ACTUAL_LIMIT=$(docker run --rm --memory=2560m --memory-swap=2560m alpine sh -c '
    if [ -f /sys/fs/cgroup/memory.max ]; then
        cat /sys/fs/cgroup/memory.max
    elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        cat /sys/fs/cgroup/memory/memory.limit_in_bytes
    else
        echo "NO_CGROUP"
    fi
')

EXPECTED_BYTES="2684354560"
if [[ "${ACTUAL_LIMIT}" != "${EXPECTED_BYTES}" ]]; then
    echo "FAILURE: Expected cgroup memory limit ${EXPECTED_BYTES} bytes, got '${ACTUAL_LIMIT}'." >&2
    exit 1
fi
echo "  -> Kernel cgroup memory limit confirmed: ${ACTUAL_LIMIT} bytes (2560 MiB / 2.5 GiB) [PASSED]"

echo "[Test 2/3] Verifying Linux Kernel OOM-Killer enforcement..."
set +e
docker run --rm --memory=50m --memory-swap=50m python:3.11-slim python -c '
# Deliberately allocate 120MB in a 50MB container to trigger kernel OOM kill
import sys
print("Allocating memory beyond 50MB ceiling...", flush=True)
chunk = bytearray(120 * 1024 * 1024)
print("Unexpected success: memory allocated without OOM!", flush=True)
' > /dev/null 2>&1
OOM_EXIT_CODE=$?
set -e

if [[ "${OOM_EXIT_CODE}" -eq 137 ]]; then
    echo "  -> Container was terminated by kernel OOM-killer (exit code 137 / SIGKILL) [PASSED]"
else
    echo "FAILURE: Expected OOM termination code 137, received exit code ${OOM_EXIT_CODE}." >&2
    exit 1
fi

echo "[Test 3/3] Verifying fail-closed policy when enforcement requested on unconstrained host..."
set +e
# Determine python executable
if [[ -f "${SCRIPT_DIR}/../.venv/bin/python" ]]; then
    PY_BIN="${SCRIPT_DIR}/../.venv/bin/python"
elif command -v python &> /dev/null; then
    PY_BIN="python"
else
    PY_BIN="python3"
fi

OUTPUT=$(export NEUROADHD_HMAC_KEY="test" && "${PY_BIN}" "${SCRIPT_DIR}/run_poc_v2.py" --mode synthetic --enforce-container-memory 2>&1)
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]] && echo "${OUTPUT}" | grep -q "MEMORY ISOLATION FAILURE"; then
    echo "  -> Pipeline correctly failed closed when run on unconstrained host [PASSED]"
else
    echo "FAILURE: Pipeline did not fail closed on unconstrained host! Exit code: ${EXIT_CODE}" >&2
    echo "Output: ${OUTPUT}" >&2
    exit 1
fi

echo "==========================================================================="
echo "ALL MEMORY ISOLATION TESTS PASSED (3/3)"
echo "OS kernel enforcement and fail-closed safety policy verified."
echo "==========================================================================="

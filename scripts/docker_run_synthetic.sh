#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NeuroADHD RawMRI PoC V2 — Container Deployment Execution Script
#
# Enforces OS-level cgroup memory limit of 2560 MiB (2.5 GiB) and swap isolation.
# Mounts authentic canonical CC200 atlas read-only into the container.
# Executes end-to-end synthetic phantom pipeline in canonical atlas mode.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-neuro_adhd_pipeline:v2}"

# Parse command line arguments
REQUIRE_MODEL_INFERENCE="${REQUIRE_MODEL_INFERENCE:-0}"
POSITIONAL_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--require-model-inference" ]]; then
        REQUIRE_MODEL_INFERENCE="1"
    else
        POSITIONAL_ARGS+=("$arg")
    fi
done

TARGET_OUT="${POSITIONAL_ARGS[0]:-${PROJECT_ROOT}/outputs/docker_deployment}"
mkdir -p "${TARGET_OUT}"
OUTPUT_DIR="$(cd "${TARGET_OUT}" && pwd)"

ATLAS_DIR="${PROJECT_ROOT}/data/atlases"
ATLAS_FILE="${ATLAS_DIR}/cc200_roi_atlas.nii.gz"

# Verify HMAC key is explicitly configured (fail-closed, no hardcoded fallbacks)
if [[ -z "${NEUROADHD_HMAC_KEY:-}" ]]; then
    echo "ERROR: Mandatory NEUROADHD_HMAC_KEY environment variable is missing or empty." >&2
    echo "An explicit HMAC integrity key must be configured in the environment." >&2
    exit 1
fi

# Optional / Mandatory frozen model directory mount (read-only)
MODEL_MOUNT_ARGS=()
RUNNER_MODEL_ARGS=()
if [[ "${REQUIRE_MODEL_INFERENCE}" == "1" ]]; then
    if [[ -z "${NEUROADHD_MODEL_DIR:-}" ]]; then
        echo "ERROR: Integrated model inference was explicitly requested, but NEUROADHD_MODEL_DIR is unset or empty." >&2
        exit 1
    fi
fi

if [[ -n "${NEUROADHD_MODEL_DIR:-}" ]]; then
    if [[ -L "${NEUROADHD_MODEL_DIR}" ]]; then
        echo "ERROR: Symlink substitution detected at NEUROADHD_MODEL_DIR: ${NEUROADHD_MODEL_DIR}" >&2
        exit 1
    fi
    if [[ ! -d "${NEUROADHD_MODEL_DIR}" ]]; then
        echo "ERROR: Configured NEUROADHD_MODEL_DIR does not exist or is not a directory: ${NEUROADHD_MODEL_DIR}" >&2
        exit 1
    fi
    MODEL_DIR_RESOLVED="$(cd "${NEUROADHD_MODEL_DIR}" && pwd)"
    if [[ -L "${MODEL_DIR_RESOLVED}/final_preprocessing_pipeline.joblib" ]] || [[ -L "${MODEL_DIR_RESOLVED}/final_model.joblib" ]]; then
        echo "ERROR: Symlink substitution detected at model artifact (.joblib)" >&2
        exit 1
    fi
    if [[ ! -f "${MODEL_DIR_RESOLVED}/final_preprocessing_pipeline.joblib" ]] || [[ ! -f "${MODEL_DIR_RESOLVED}/final_model.joblib" ]]; then
        echo "ERROR: NEUROADHD_MODEL_DIR missing required frozen artifacts (.joblib)" >&2
        exit 1
    fi
    MODEL_MOUNT_ARGS=(-v "${MODEL_DIR_RESOLVED}:/opt/neuro_adhd/models:ro" -e NEUROADHD_MODEL_DIR="/opt/neuro_adhd/models")
    RUNNER_MODEL_ARGS=(--model-dir /opt/neuro_adhd/models)
fi

if [[ "${REQUIRE_MODEL_INFERENCE}" == "1" ]]; then
    RUNNER_MODEL_ARGS+=(--require-model-inference)
fi

echo "==========================================================================="
echo "NeuroADHD RawMRI PoC V2 — Container Deployment Runner"
echo "==========================================================================="
echo "Image:                ${IMAGE_NAME}"
echo "Memory Constraint:    2560m (Hard Ceiling: 2.5 GiB)"
echo "Swap Constraint:      2560m (Swap Disabled/Contained)"
echo "Host Atlas Path:      ${ATLAS_FILE}"
echo "Host Output Path:     ${OUTPUT_DIR}"
if [[ -n "${NEUROADHD_MODEL_DIR:-}" ]]; then
    echo "Host Model Mount:     ${MODEL_DIR_RESOLVED} -> /opt/neuro_adhd/models:ro"
fi
echo "Require Model Infer:  ${REQUIRE_MODEL_INFERENCE}"
echo "==========================================================================="

# Validate canonical atlas file presence
if [[ ! -f "${ATLAS_FILE}" ]]; then
    echo "ERROR: Authentic CC200 atlas not found at '${ATLAS_FILE}'." >&2
    echo "Task 1 requires deploying the authentic CC200 atlas." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Execute container with hard memory limit and read-only atlas/model mounts
docker run --rm \
    --name neuro_adhd_synthetic_deploy_$$ \
    --memory=2560m \
    --memory-swap=2560m \
    -e NEUROADHD_HMAC_KEY="${NEUROADHD_HMAC_KEY}" \
    "${MODEL_MOUNT_ARGS[@]}" \
    -v "${ATLAS_DIR}:/opt/neuro_adhd/data/atlases:ro" \
    -v "${OUTPUT_DIR}:/opt/neuro_adhd/outputs:rw" \
    "${IMAGE_NAME}" \
    --mode synthetic \
    --atlas-mode canonical \
    --enforce-container-memory \
    --output-dir /opt/neuro_adhd/outputs \
    "${RUNNER_MODEL_ARGS[@]}"

echo "==========================================================================="
echo "Container execution completed successfully."
echo "Verified outputs saved to: ${OUTPUT_DIR}"
echo "==========================================================================="

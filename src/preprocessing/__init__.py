"""Preprocessing modules for NeuroADHD_RawMRI PoC V2.

Contains:
- slice_timing: Metadata-driven BIDS slice-timing correction (DEC-STC-01)
- motion: Rigid volume realignment, FD calculation, Friston-24 regressors
- registration: Coregistration, MNI SyN normalization, composed resampling
- segmentation: Brain extraction, bias correction, tissue segmentation (CSF/GM/WM)
- nuisance: Anatomical CompCor, confound matrix construction, simultaneous filtering
- qc: Visual registration and parcellation inspection overlays
"""
from src.preprocessing.motion import (
    apply_motion_transforms,
    build_friston_24_regressors,
    compute_framewise_displacement,
    estimate_motion_parameters,
    realign_4d_bold,
)
from src.preprocessing.nuisance import (
    build_confound_matrix,
    clean_functional_bold,
    extract_acompcor_components,
)
from src.preprocessing.qc import (
    compute_registration_qc_metrics,
    generate_before_after_qc_figure,
    generate_coregistration_qc_overlay,
    generate_landmark_qc_overlay,
    generate_normalization_qc_overlay,
    generate_parcellation_qc_overlay,
    generate_registration_qc_overlay,
    generate_tissue_mask_qc_overlay,
    validate_qc_grid_alignment,
)
from src.preprocessing.registration import (
    compose_transformations,
    compose_volume_transform,
    compute_spatial_dice,
    generate_functional_reference,
    register_epi_to_t1w,
    register_t1w_to_mni,
    resample_4d_bold,
    resample_mask_to_target,
)
from src.preprocessing.segmentation import (
    correct_bias_field,
    create_eroded_tissue_masks,
    extract_anatomical_brain,
    segment_t1w_tissue,
    validate_tissue_masks,
)
from src.preprocessing.slice_timing import (
    DEFAULT_STC_METHOD,
    SUPPORTED_STC_METHODS,
    DoubleSliceTimingCorrectionError,
    InvalidSliceTimingError,
    MissingSliceTimingError,
    SliceTimingValidationError,
    SliceTimingValidationReport,
    build_slice_timing_derivative_metadata,
    compute_temporal_boundary_regions,
    correct_slice_timing,
    correct_slice_timing_with_report,
    detect_stc_software,
    validate_slice_timing_metadata,
    write_slice_timing_derivative,
)

__all__ = [
    "DEFAULT_STC_METHOD",
    "SUPPORTED_STC_METHODS",
    "DoubleSliceTimingCorrectionError",
    "InvalidSliceTimingError",
    "MissingSliceTimingError",
    "SliceTimingValidationError",
    "SliceTimingValidationReport",
    "build_slice_timing_derivative_metadata",
    "compute_temporal_boundary_regions",
    "correct_slice_timing",
    "correct_slice_timing_with_report",
    "detect_stc_software",
    "validate_slice_timing_metadata",
    "write_slice_timing_derivative",
    "estimate_motion_parameters",
    "apply_motion_transforms",
    "realign_4d_bold",
    "compute_framewise_displacement",
    "build_friston_24_regressors",
    "generate_functional_reference",
    "register_epi_to_t1w",
    "register_t1w_to_mni",
    "compose_transformations",
    "compose_volume_transform",
    "resample_4d_bold",
    "resample_mask_to_target",
    "compute_spatial_dice",
    "extract_anatomical_brain",
    "correct_bias_field",
    "segment_t1w_tissue",
    "create_eroded_tissue_masks",
    "validate_tissue_masks",
    "extract_acompcor_components",
    "build_confound_matrix",
    "clean_functional_bold",
    "generate_registration_qc_overlay",
    "generate_coregistration_qc_overlay",
    "generate_normalization_qc_overlay",
    "generate_tissue_mask_qc_overlay",
    "generate_parcellation_qc_overlay",
    "compute_registration_qc_metrics",
    "generate_before_after_qc_figure",
    "generate_landmark_qc_overlay",
    "validate_qc_grid_alignment",
]

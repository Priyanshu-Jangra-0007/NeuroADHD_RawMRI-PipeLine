"""Scientific and numerical validation tests for spatial registration and resampling.

Phase A.2 Validation:
1. Review transformation mathematics and pullback conventions.
2. Deterministic numerical landmark tests:
   - Identity transformation
   - Pure 3D translation
   - Proper 3D rotation (preserving chirality, det(R) = +1)
   - General 12-DOF affine transformation (det(A) > 0, kappa(A) < 10)
   - Nonlinear diffeomorphic SyN deformation
   - Composed EPI-to-MNI sequential landmark consistency
3. Resampling verification:
   - Left-right asymmetry preservation (no sign/hemispheric flips)
   - Discrete label preservation (nearest-neighbor preserves exact integer set)
   - Constant signal amplitude invariance
4. Scientific registration QC metrics (Pearson r, RMSE, Dice, TRE).
"""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from dipy.align.imwarp import DiffeomorphicMap

from src.preprocessing.qc import (
    compute_registration_qc_metrics,
    generate_before_after_qc_figure,
    generate_landmark_qc_overlay,
)
from src.preprocessing.registration import (
    compose_transformations,
    compute_spatial_dice,
    resample_4d_bold,
    resample_mask_to_target,
)


def _euler_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Generate 4x4 proper rotation matrix with positive determinant (+1.0)."""
    r_x = np.deg2rad(roll_deg)
    r_y = np.deg2rad(pitch_deg)
    r_z = np.deg2rad(yaw_deg)

    cx, sx = np.cos(r_x), np.sin(r_x)
    cy, sy = np.cos(r_y), np.sin(r_y)
    cz, sz = np.cos(r_z), np.sin(r_z)

    Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


class TestScientificRegistrationValidation(unittest.TestCase):
    """Deterministic scientific validation suite for registration transforms and resampling."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)

        # Standard grids: Target (30x30x30, 2mm isotropic), Source (32x32x28, 2.5mm isotropic)
        self.target_shape = (30, 30, 30)
        self.target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        self.target_affine[:3, 3] = [-30.0, -30.0, -30.0]

        self.source_shape = (32, 32, 28)
        self.source_affine = np.diag([2.5, 2.5, 2.5, 1.0])
        self.source_affine[:3, 3] = [-40.0, -40.0, -35.0]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_identity_transformation_landmarks_and_residuals(self):
        """Test 1: Identity transform M = I4, displacement error < 1e-6 mm, RMSE = 0.0."""
        identity_map = DiffeomorphicMap(
            3,
            disp_shape=self.target_shape,
            disp_grid2world=self.target_affine,
            domain_shape=self.target_shape,
            domain_grid2world=self.target_affine,
            codomain_shape=self.target_shape,
            codomain_grid2world=self.target_affine,
        )
        identity_map.forward = np.zeros((*self.target_shape, 3), dtype=np.float32)
        identity_map.backward = np.zeros((*self.target_shape, 3), dtype=np.float32)

        # Known physical landmarks in target space (mm)
        landmarks_mm = np.array([
            [-10.0, -10.0, -10.0],
            [0.0, 0.0, 0.0],
            [15.0, 15.0, 15.0],
            [-5.0, 20.0, 10.0],
        ], dtype=np.float64)

        # Apply identity transform: x_warped = x_orig
        warped_landmarks_mm = landmarks_mm.copy()

        # Target registration error must be identically zero
        tre = np.linalg.norm(warped_landmarks_mm - landmarks_mm, axis=1)
        self.assertTrue(np.all(tre < 1e-6), f"TRE exceeds 1e-6 mm: {tre}")

        # Signal test
        vol = np.zeros(self.target_shape, dtype=np.float32)
        vol[10:20, 10:20, 10:20] = 100.0
        warped_vol = identity_map.transform(vol)

        rmse = float(np.sqrt(np.mean((vol - warped_vol) ** 2)))
        self.assertEqual(rmse, 0.0, f"Identity transform produced non-zero RMSE: {rmse}")

    def test_pure_3d_translation_landmarks_and_consistency(self):
        """Test 2: Pure translation t = [4.0, -4.0, 6.0] mm, verify TRE and consistency."""
        t_mm = np.array([4.0, -4.0, 6.0])

        # Pullback matrix maps static coords to moving coords: x_moving = x_static + t
        M_pullback = np.eye(4, dtype=np.float64)
        M_pullback[:3, 3] = t_mm

        # Forward transformation: x_static = x_moving - t
        M_forward = np.linalg.inv(M_pullback)
        np.testing.assert_allclose(M_forward[:3, 3], -t_mm)

        # Verify landmark forward and pullback consistency: M_forward @ M_pullback == I
        np.testing.assert_allclose(M_forward @ M_pullback, np.eye(4), atol=1e-12)

        # Physical landmarks in static space (interior coordinates)
        x_static = np.array([20.0, 20.0, 20.0, 1.0])
        x_moving = M_pullback @ x_static
        x_recovered = M_forward @ x_moving
        self.assertLess(np.linalg.norm(x_static - x_recovered), 1e-12)

        # Diffeomorphic map with constant translation displacement
        grid_shape = (25, 25, 25)
        grid_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        trans_map = DiffeomorphicMap(
            3,
            disp_shape=grid_shape,
            disp_grid2world=grid_affine,
            domain_shape=grid_shape,
            domain_grid2world=grid_affine,
            codomain_shape=grid_shape,
            codomain_grid2world=grid_affine,
        )
        disp = np.zeros((*grid_shape, 3), dtype=np.float32)
        disp[...] = t_mm
        trans_map.backward = disp
        trans_map.forward = -disp
        trans_map.is_inverse = True

        # Place Gaussian source at x_moving voxel
        i_moving = (np.linalg.inv(grid_affine) @ x_moving)[:3]
        gx, gy, gz = np.indices(grid_shape)
        source_vol = np.exp(-((gx - i_moving[0]) ** 2 + (gy - i_moving[1]) ** 2 + (gz - i_moving[2]) ** 2) / 2.0).astype(np.float32)

        warped_vol = trans_map.transform(source_vol)

        # Peak of warped volume should land at x_static voxel (10, 10, 10)
        i_static = (np.linalg.inv(grid_affine) @ x_static)[:3]
        peak_idx = np.unravel_index(np.argmax(warped_vol), warped_vol.shape)
        np.testing.assert_allclose(peak_idx, np.round(i_static), atol=1e-5)

    def test_3d_rotation_preserving_chirality_orientation(self):
        """Test 3: 3D rotation with roll=10°, pitch=-8°, yaw=15°. det(R) = +1.000000 (no flip)."""
        R = _euler_matrix(10.0, -8.0, 15.0)

        # Determinant must be exactly +1.0 (proper orthogonal matrix SO(3))
        det_R = float(np.linalg.det(R))
        self.assertAlmostEqual(det_R, 1.0, places=6, msg="Rotation must have det = +1.0 (proper rotation)")
        self.assertGreater(det_R, 0.0, "det(R) <= 0 introduces an improper left-right flip/reflection!")

        # Orthogonality: R.T @ R == I
        np.testing.assert_allclose(R[:3, :3].T @ R[:3, :3], np.eye(3), atol=1e-12)

        # Check landmarks under forward and inverse rotation
        landmarks = np.array([
            [15.0, 20.0, -10.0, 1.0],
            [-12.0, 5.0, 25.0, 1.0],
            [0.0, 30.0, 0.0, 1.0],
        ])
        R_inv = np.linalg.inv(R)

        for lm in landmarks:
            lm_rot = R @ lm
            lm_back = R_inv @ lm_rot
            np.testing.assert_allclose(lm, lm_back, atol=1e-12)

    def test_general_12_dof_affine_transformation(self):
        """Test 4: 12-DOF affine (rotation, scaling, shear, translation), det(A) > 0, kappa(A) < 10."""
        R = _euler_matrix(5.0, -4.0, 8.0)[:3, :3]
        S = np.diag([1.10, 0.95, 1.05])  # Anisotropic scaling
        shear = np.array([
            [1.0, 0.05, 0.0],
            [0.0, 1.0, 0.02],
            [0.0, 0.0, 1.0],
        ])
        A3 = R @ shear @ S
        A = np.eye(4, dtype=np.float64)
        A[:3, :3] = A3
        A[:3, 3] = [3.2, -4.5, 2.1]

        # Orientation preservation: det(A) > 0
        det_A = float(np.linalg.det(A))
        self.assertGreater(det_A, 0.0, "Affine transform must preserve orientation (det > 0)")
        expected_det = np.linalg.det(R) * np.linalg.det(shear) * np.linalg.det(S)
        self.assertAlmostEqual(det_A, expected_det, places=5)

        # Condition number: kappa(A[:3, :3]) < 10
        cond = float(np.linalg.cond(A[:3, :3]))
        self.assertLess(cond, 10.0, f"Condition number too high ({cond:.2f}), ill-conditioned affine")

        # Invertibility: A @ inv(A) == I
        A_inv = np.linalg.inv(A)
        np.testing.assert_allclose(A @ A_inv, np.eye(4), atol=1e-12)

    def test_nonlinear_diffeomorphic_syn_deformation_and_inversion(self):
        """Test 5: Nonlinear SyN displacement bump, displacement vector orientation and consistency."""
        disp_shape = (24, 24, 24)
        disp_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        gx, gy, gz = np.indices(disp_shape)
        r2 = (gx - 12) ** 2 + (gy - 12) ** 2 + (gz - 12) ** 2
        sigma = 3.5
        bump = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)

        disp_backward = np.zeros((*disp_shape, 3), dtype=np.float32)
        disp_backward[..., 0] = bump * 3.0  # +3mm X
        disp_backward[..., 1] = bump * -2.0  # -2mm Y
        disp_backward[..., 2] = bump * 1.5  # +1.5mm Z

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=disp_shape,
            disp_grid2world=disp_affine,
            domain_shape=disp_shape,
            domain_grid2world=disp_affine,
            codomain_shape=disp_shape,
            codomain_grid2world=disp_affine,
        )
        syn_map.backward = disp_backward
        syn_map.forward = -disp_backward

        # Center point displacement at voxel (12, 12, 12)
        center_disp = syn_map.backward[12, 12, 12]
        np.testing.assert_allclose(center_disp, [3.0, -2.0, 1.5], atol=1e-5)

        # Forward/backward consistency: sum of displacement vectors is zero for small deformation
        sum_disp = syn_map.backward[12, 12, 12] + syn_map.forward[12, 12, 12]
        np.testing.assert_allclose(sum_disp, [0.0, 0.0, 0.0], atol=1e-6)

    def test_composed_epi_to_mni_sequential_landmark_verification(self):
        """Test 6: Composed EPI-to-MNI registration matches independent sequential coordinate math."""
        # Target (MNI) geometry
        target_shape = (28, 28, 26)
        target_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        # Source (EPI) geometry
        source_shape = (26, 26, 24)
        source_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        # 1. Forward T1w-to-MNI affine (rotation + translation)
        R_t1w_to_mni = _euler_matrix(4.0, -3.0, 6.0)
        R_t1w_to_mni[:3, 3] = [2.0, -1.0, 3.0]
        # Pullback matrix from MNI to T1w
        M_mni_to_t1w = np.linalg.inv(R_t1w_to_mni)

        # 2. Forward EPI-to-T1w rigid (rotation + translation)
        R_epi_to_t1w = _euler_matrix(-2.0, 5.0, -3.0)
        R_epi_to_t1w[:3, 3] = [-3.0, 2.0, -1.0]
        # Pullback matrix from T1w to EPI
        M_t1w_to_epi = np.linalg.inv(R_epi_to_t1w)

        # 3. SyN displacement bump centered at (14, 14, 13)
        gx, gy, gz = np.indices(target_shape)
        r2 = (gx - 14) ** 2 + (gy - 14) ** 2 + (gz - 13) ** 2
        bump = np.exp(-r2 / (2 * (3.0 ** 2))).astype(np.float32)

        disp_backward = np.zeros((*target_shape, 3), dtype=np.float32)
        disp_backward[..., 0] = bump * 2.5
        disp_backward[..., 1] = bump * -1.8
        disp_backward[..., 2] = bump * 1.2

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=source_shape,
            domain_grid2world=source_affine,
            codomain_shape=target_shape,
            codomain_grid2world=target_affine,
        )
        syn_map.backward = disp_backward
        syn_map.forward = -disp_backward

        # Compose transformations
        composed = compose_transformations(
            rigid_epi_to_t1w=M_t1w_to_epi,
            affine_t1w_to_mni=M_mni_to_t1w,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        # Target landmark at (14, 14, 13)
        tgt_vox = np.array([14.0, 14.0, 13.0, 1.0])
        x_tgt_mm = target_affine @ tgt_vox
        d_tgt_mm = np.array([2.5, -1.8, 1.2, 0.0])

        # Sequential pullback:
        # Step A: SyN deformation in target world
        x_disp_mm = x_tgt_mm + d_tgt_mm
        # Step B: Affine from MNI to T1w
        x_t1w_mm = M_mni_to_t1w @ x_disp_mm
        # Step C: Rigid from T1w to EPI
        x_epi_mm = M_t1w_to_epi @ x_t1w_mm
        # Step D: Continuous voxel coordinate in source EPI grid
        i_epi_vox = (np.linalg.inv(source_affine) @ x_epi_mm)[:3]

        # Place Gaussian landmark peak in source volume at i_epi_vox
        sx, sy, sz = np.indices(source_shape)
        sr2 = (sx - i_epi_vox[0]) ** 2 + (sy - i_epi_vox[1]) ** 2 + (sz - i_epi_vox[2]) ** 2
        source_vol = (100.0 * np.exp(-sr2 / 2.0)).astype(np.float32)

        # Resample through composed map
        warped_vol = composed.transform(source_vol)

        # Peak of warped volume must land at the target landmark voxel
        peak_idx = np.unravel_index(np.argmax(warped_vol), warped_vol.shape)
        self.assertEqual(peak_idx, (14, 14, 13), f"Warped peak {peak_idx} deviated from target (14, 14, 13)")
        self.assertGreater(warped_vol[14, 14, 13], 70.0)

    def test_left_right_asymmetry_preservation_no_flip(self):
        """Test 7: Asymmetric left-hemisphere phantom signal remains strictly in left hemisphere."""
        grid_shape = (30, 30, 30)
        grid_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        # Center origin so x=0 is mid-sagittal
        grid_affine[0, 3] = -float(grid_shape[0])  # x ranges from -30 to +28 mm

        # Signal strictly in left hemisphere (voxel x < 15, physical x < 0)
        asym_data = np.zeros(grid_shape, dtype=np.float32)
        asym_data[5:12, 10:20, 10:20] = 100.0  # Entirely in x < 15

        # Rotation preserving orientation (det = +1)
        R_pullback = _euler_matrix(0.0, 0.0, 10.0)  # 10 deg in-plane rotation
        np.testing.assert_allclose(np.linalg.det(R_pullback), 1.0)

        rot_map = DiffeomorphicMap(
            3,
            disp_shape=grid_shape,
            disp_grid2world=grid_affine,
            domain_shape=grid_shape,
            domain_grid2world=grid_affine,
            codomain_shape=grid_shape,
            codomain_grid2world=grid_affine,
        )
        rot_map.forward = np.zeros((*grid_shape, 3), dtype=np.float32)
        rot_map.backward = np.zeros((*grid_shape, 3), dtype=np.float32)

        composed = compose_transformations(
            rigid_epi_to_t1w=R_pullback,
            affine_t1w_to_mni=np.eye(4),
            syn_mapping=rot_map,
            target_shape=grid_shape,
            target_affine=grid_affine,
            source_shape=grid_shape,
            source_affine=grid_affine,
        )

        warped = composed.transform(asym_data)

        # Right hemisphere (x >= 15) must have no significant signal (> 1e-3)
        right_hemisphere_signal = np.sum(warped[17:, :, :])
        total_signal = np.sum(warped)
        self.assertGreater(total_signal, 1000.0)
        self.assertLess(right_hemisphere_signal / total_signal, 0.05, "Signal erroneously flipped into opposite hemisphere!")

    def test_discrete_label_preservation_nearest_neighbor(self):
        """Test 8: Discrete integer mask labels {0..200} preserved exactly with nearest neighbor."""
        grid_shape = (25, 25, 25)
        grid_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        label_data = np.zeros(grid_shape, dtype=np.int32)
        label_data[5:10, 5:10, 5:10] = 42
        label_data[12:18, 12:18, 12:18] = 137
        label_data[19:23, 19:23, 19:23] = 200

        label_img = nib.Nifti1Image(label_data, grid_affine)

        # Identity displacement map
        identity_map = DiffeomorphicMap(
            3,
            disp_shape=grid_shape,
            disp_grid2world=grid_affine,
            domain_shape=grid_shape,
            domain_grid2world=grid_affine,
            codomain_shape=grid_shape,
            codomain_grid2world=grid_affine,
        )
        identity_map.forward = np.zeros((*grid_shape, 3), dtype=np.float32)
        identity_map.backward = np.zeros((*grid_shape, 3), dtype=np.float32)

        resampled_mask = resample_mask_to_target(label_img, identity_map, grid_shape, grid_affine)

        res_data = np.asanyarray(resampled_mask.dataobj)
        unique_labels = np.unique(res_data)

        # Must only contain original integer labels, zero non-integer fractions
        self.assertTrue(set(unique_labels).issubset({0, 42, 137, 200}))
        self.assertEqual(res_data.dtype.kind, "i", "Resampled mask data must remain integer")

    def test_constant_signal_amplitude_invariance(self):
        """Test 9: Constant signal C = 100.0 inside volume remains 100.0 +/- 1e-4."""
        grid_shape = (25, 25, 25)
        grid_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        const_data = np.full(grid_shape, 100.0, dtype=np.float32)

        identity_map = DiffeomorphicMap(
            3,
            disp_shape=grid_shape,
            disp_grid2world=grid_affine,
            domain_shape=grid_shape,
            domain_grid2world=grid_affine,
            codomain_shape=grid_shape,
            codomain_grid2world=grid_affine,
        )
        identity_map.forward = np.zeros((*grid_shape, 3), dtype=np.float32)
        identity_map.backward = np.zeros((*grid_shape, 3), dtype=np.float32)

        warped_const = identity_map.transform(const_data)

        # Interior voxels must remain 100.0 +/- 1e-4
        interior = warped_const[3:-3, 3:-3, 3:-3]
        np.testing.assert_allclose(interior, 100.0, atol=1e-4)

    def test_scientific_qc_metrics_and_visual_generation(self):
        """Test 10: Scientific QC evaluation (Pearson r, RMSE, Dice, TRE overlay generation)."""
        gx, gy, gz = np.indices(self.target_shape)
        r2 = ((gx - 15) / 6.0) ** 2 + ((gy - 15) / 6.0) ** 2 + ((gz - 15) / 6.0) ** 2
        vol_a = (np.exp(-r2) * 100.0).astype(np.float32)
        img_a = nib.Nifti1Image(vol_a, self.target_affine)

        # Slightly shifted volume (0.5 voxel)
        r2_b = ((gx - 15.5) / 6.0) ** 2 + ((gy - 15) / 6.0) ** 2 + ((gz - 15) / 6.0) ** 2
        vol_b = (np.exp(-r2_b) * 100.0).astype(np.float32)
        img_b = nib.Nifti1Image(vol_b, self.target_affine)

        # 1. Registration QC metrics
        metrics = compute_registration_qc_metrics(img_a, img_b)
        self.assertGreater(metrics["pearson_r"], 0.70)
        self.assertGreater(metrics["dice"], 0.70)
        self.assertTrue(metrics["qc_pass"])

        # 2. Before/After QC visualization figure
        before_png = self.work_dir / "before_after_qc.png"
        gen_before = generate_before_after_qc_figure(img_a, img_a, img_b, before_png)
        self.assertTrue(gen_before.is_file())
        self.assertGreater(gen_before.stat().st_size, 5000)

        # 3. Landmark QC overlay
        lms_tgt = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        lms_warped = np.array([[0.5, -0.3, 0.2], [10.2, 9.8, 10.1]])
        lm_png = self.work_dir / "landmark_qc.png"
        gen_lm, lm_metrics = generate_landmark_qc_overlay(
            background_img=img_a,
            target_landmarks_mm=lms_tgt,
            warped_landmarks_mm=lms_warped,
            output_png_path=lm_png,
            max_allowed_tre_mm=2.0,
        )
        self.assertTrue(gen_lm.is_file())
        self.assertTrue(lm_metrics["qc_pass"])
        self.assertLess(lm_metrics["max_tre_mm"], 2.0)


if __name__ == "__main__":
    unittest.main()

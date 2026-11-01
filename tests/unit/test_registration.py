"""Unit tests for spatial coregistration, normalization, and composed resampling."""
from __future__ import annotations

import unittest
import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, transform_centers_of_mass
from dipy.align.transforms import RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration, DiffeomorphicMap
from dipy.align.metrics import CCMetric

from src.preprocessing.registration import (
    check_process_memory_gb,
    compose_transformations,
    compute_spatial_dice,
    generate_functional_reference,
    register_epi_to_t1w,
    register_t1w_to_mni,
    resample_4d_bold,
    resample_mask_to_target,
)


class TestRegistration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create standard test grid affine (3mm isotropic)
        self.affine_3mm = np.diag([3.0, 3.0, 3.0, 1.0])
        # Create finer test grid affine (1.5mm isotropic)
        self.affine_1p5mm = np.diag([1.5, 1.5, 1.5, 1.0])

    def test_generate_functional_reference_valid(self):
        # 4D BOLD array (20, 20, 15, 60)
        data = np.ones((20, 20, 15, 60), dtype=np.float32)
        # Add temporal variation
        for t in range(60):
            data[5:15, 5:15, 3:12, t] = 100.0 + t * 0.1

        bold_img = nib.Nifti1Image(data, self.affine_3mm)
        ref_img, mask_img = generate_functional_reference(bold_img)

        self.assertEqual(ref_img.shape, (20, 20, 15))
        self.assertEqual(mask_img.shape, (20, 20, 15))
        np.testing.assert_allclose(ref_img.affine, self.affine_3mm)
        # Check mean intensity
        expected_mean = np.mean(data, axis=3)
        np.testing.assert_allclose(np.asanyarray(ref_img.dataobj), expected_mean, rtol=1e-5)

    def test_generate_functional_reference_rejects_3d(self):
        data_3d = np.ones((20, 20, 15), dtype=np.float32)
        img_3d = nib.Nifti1Image(data_3d, self.affine_3mm)
        with self.assertRaises(ValueError) as ctx:
            generate_functional_reference(img_3d)
        self.assertIn("Expected 4D functional scan", str(ctx.exception))

    def test_rigid_coregistration_direction(self):
        # Static: T1w with object at (10..20, 10..20, 10..20)
        static = np.zeros((30, 30, 30), dtype=np.float32)
        static[10:20, 10:20, 10:20] = 100.0
        static_img = nib.Nifti1Image(static, np.eye(4))

        # Moving: EPI shifted by +4 along X
        moving = np.zeros((30, 30, 30), dtype=np.float32)
        moving[14:24, 10:20, 10:20] = 100.0
        moving_img = nib.Nifti1Image(moving, np.eye(4))

        # Execute rigid coregistration
        matrix, affine_map = register_epi_to_t1w(moving_img, static_img, level_iters=[10])

        # Verify translation maps points in static space to moving space (+4 along X)
        self.assertAlmostEqual(matrix[0, 3], 4.0, delta=0.5)

        # Warp moving image into static space and verify high overlap
        warped = affine_map.transform(moving)
        dice = compute_spatial_dice(warped > 50, static > 50)
        self.assertGreater(dice, 0.85)

    def test_composed_transformation_and_single_step_resampling(self):
        # Static (MNI): object at [8..18]
        mni = np.zeros((30, 30, 30), dtype=np.float32)
        mni[8:18, 10:20, 10:20] = 100.0
        mni_img = nib.Nifti1Image(mni, np.eye(4))

        # Intermediate (T1w): shifted by +3 relative to MNI [11..21]
        t1w = np.zeros((30, 30, 30), dtype=np.float32)
        t1w[11:21, 10:20, 10:20] = 100.0
        t1w_img = nib.Nifti1Image(t1w, np.eye(4))

        # Moving (EPI 4D): shifted by +4 relative to T1w (+7 relative to MNI) [15..25], T=50
        epi_4d = np.zeros((30, 30, 30, 50), dtype=np.float32)
        for t in range(50):
            epi_4d[15:25, 10:20, 10:20, t] = 100.0 + (t % 5)
        epi_img = nib.Nifti1Image(epi_4d, np.eye(4))

        mean_epi_img = nib.Nifti1Image(np.mean(epi_4d, axis=3), np.eye(4))

        # Step 1: EPI to T1w
        m_epi_to_t1w, _ = register_epi_to_t1w(mean_epi_img, t1w_img, level_iters=[10])

        # Step 2: T1w to MNI
        m_t1w_to_mni, syn_map = register_t1w_to_mni(
            t1w_img, mni_img, affine_level_iters=[10], syn_level_iters=[5]
        )

        # Step 3: Compose
        composed_map = compose_transformations(
            m_epi_to_t1w,
            m_t1w_to_mni,
            syn_map,
            target_shape=(30, 30, 30),
            target_affine=np.eye(4),
            source_shape=(30, 30, 30),
            source_affine=np.eye(4),
        )

        # Step 4: Resample 4D series in single step
        resampled_img = resample_4d_bold(
            epi_img, composed_map, target_shape=(30, 30, 30), target_affine=np.eye(4)
        )

        self.assertEqual(resampled_img.shape, (30, 30, 30, 50))
        np.testing.assert_allclose(resampled_img.affine, np.eye(4))

        # Check spatial overlap of resampled mean with MNI
        resampled_mean = np.mean(np.asanyarray(resampled_img.dataobj), axis=3)
        dice = compute_spatial_dice(resampled_mean > 50, mni > 50)
        self.assertGreater(dice, 0.70)

    def test_resample_mask_nearest_interpolation(self):
        # Create discrete label image with values 0, 1, 2, 5
        labels_3d = np.zeros((20, 20, 20), dtype=np.int32)
        labels_3d[5:10, 5:10, 5:10] = 1
        labels_3d[10:15, 10:15, 10:15] = 2
        labels_3d[15:18, 15:18, 15:18] = 5
        mask_img = nib.Nifti1Image(labels_3d, np.eye(4))

        # Identity diffeomorphic map
        dmap = DiffeomorphicMap(3, disp_shape=(20, 20, 20))
        dmap.forward = np.zeros((20, 20, 20, 3), dtype=np.float32)
        dmap.backward = np.zeros((20, 20, 20, 3), dtype=np.float32)

        resampled_mask = resample_mask_to_target(
            mask_img, dmap, target_shape=(20, 20, 20), target_affine=np.eye(4)
        )

        resampled_data = np.asanyarray(resampled_mask.dataobj)
        # Nearest neighbor should strictly preserve unique discrete integer labels
        unique_vals = set(np.unique(resampled_data))
        self.assertTrue(unique_vals.issubset({0, 1, 2, 5}))
        self.assertEqual(resampled_data.dtype, np.int32)

    def test_resample_4d_preserves_temporal_dimension(self):
        data = np.random.uniform(50, 150, (15, 15, 10, 55)).astype(np.float32)
        bold_img = nib.Nifti1Image(data, self.affine_3mm)

        dmap = DiffeomorphicMap(3, disp_shape=(12, 12, 8))
        dmap.forward = np.zeros((12, 12, 8, 3), dtype=np.float32)
        dmap.backward = np.zeros((12, 12, 8, 3), dtype=np.float32)

        resampled = resample_4d_bold(
            bold_img, dmap, target_shape=(12, 12, 8), target_affine=self.affine_3mm
        )

        self.assertEqual(resampled.shape, (12, 12, 8, 55))
        self.assertEqual(resampled.shape[3], 55)

    def test_resample_4d_rejects_insufficient_timepoints(self):
        data = np.ones((15, 15, 10, 30), dtype=np.float32)  # T=30 < 50
        bold_img = nib.Nifti1Image(data, self.affine_3mm)
        dmap = DiffeomorphicMap(3, disp_shape=(15, 15, 10))

        with self.assertRaises(ValueError) as ctx:
            resample_4d_bold(bold_img, dmap, target_shape=(15, 15, 10), target_affine=self.affine_3mm)
        self.assertIn("Insufficient BOLD timepoints", str(ctx.exception))

    def test_peak_memory_below_ceiling(self):
        mem_gb = check_process_memory_gb()
        self.assertLess(mem_gb, 2.5, f"Peak memory {mem_gb:.2f} GB exceeds 2.5 GB ceiling")

    def test_compute_spatial_dice(self):
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)

        # Both empty -> 1.0
        self.assertEqual(compute_spatial_dice(a, b), 1.0)

        # One empty -> 0.0
        a[2:5, 2:5, 2:5] = True
        self.assertEqual(compute_spatial_dice(a, b), 0.0)

        # Identical -> 1.0
        b[2:5, 2:5, 2:5] = True
        self.assertEqual(compute_spatial_dice(a, b), 1.0)

        # Partial overlap (half)
        b = np.zeros((10, 10, 10), dtype=bool)
        b[2:5, 2:5, 2:4] = True  # 18 voxels out of 27
        dice = compute_spatial_dice(a, b)
        self.assertAlmostEqual(dice, (2 * 18) / (27 + 18), delta=1e-4)

    def test_compose_transformations_output_has_target_not_source_shape(self):
        """Composed map must output target (CC200) dimensions, not source (EPI) dimensions."""
        # Source (EPI): smaller grid, coarser resolution
        source_shape = (20, 18, 15)
        source_affine = np.diag([3.5, 3.5, 3.5, 1.0])

        # Target (CC200-like): larger grid, different resolution and orientation
        target_shape = (24, 22, 18)
        target_affine = np.array([
            [-3.0, 0, 0, 45.0],
            [0, 3.0, 0, -63.0],
            [0, 0, 3.0, -36.0],
            [0, 0, 0, 1.0],
        ])

        t1w_shape = (28, 26, 20)
        t1w_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        rigid_epi_to_t1w = np.eye(4)
        rigid_epi_to_t1w[0, 3] = 5.0  # +5mm X translation

        affine_t1w_to_mni = np.eye(4)
        affine_t1w_to_mni[0, 3] = 3.0  # +3mm X translation

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=t1w_shape,
            codomain_grid2world=t1w_affine,
        )
        syn_map.forward = np.zeros((*target_shape, 3), dtype=np.float32)
        syn_map.backward = np.zeros((*t1w_shape, 3), dtype=np.float32)

        composed_map = compose_transformations(
            rigid_epi_to_t1w=rigid_epi_to_t1w,
            affine_t1w_to_mni=affine_t1w_to_mni,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        # In DIPY is_inverse=True convention:
        # domain represents moving source input grid
        # codomain represents static target output grid
        self.assertTrue(composed_map.is_inverse)
        np.testing.assert_array_equal(composed_map.domain_shape, source_shape)
        np.testing.assert_allclose(composed_map.domain_grid2world, source_affine)
        np.testing.assert_array_equal(composed_map.codomain_shape, target_shape)
        np.testing.assert_allclose(composed_map.codomain_grid2world, target_affine)

        # Prealign_inv maps standard (MNI/CC200) coords -> native EPI coords: M_t1w_to_epi @ M_mni_to_t1w
        expected_prealign = rigid_epi_to_t1w @ affine_t1w_to_mni
        np.testing.assert_allclose(composed_map.prealign_inv, expected_prealign, atol=1e-6)

        # transform(source) -> target CC200 shape
        src_vol = np.zeros(source_shape, dtype=np.float32)
        warped = composed_map.transform(src_vol)
        self.assertEqual(warped.shape, target_shape)

        # transform_inverse(target) -> source EPI shape
        tgt_vol = np.zeros(target_shape, dtype=np.float32)
        inv_warped = composed_map.transform_inverse(tgt_vol)
        self.assertEqual(inv_warped.shape, source_shape)

    def test_resample_4d_bold_output_has_target_affine_not_source_affine(self):
        """Resampled BOLD must carry target CC200 affine, not source EPI affine."""
        source_shape = (18, 16, 14)
        source_affine = np.diag([3.0, 3.0, 3.5, 1.0])

        target_shape = (20, 18, 15)
        target_affine = np.array([
            [-3.0, 0, 0, 40.0],
            [0, 3.0, 0, -45.0],
            [0, 0, 3.0, -30.0],
            [0, 0, 0, 1.0],
        ])

        t_vol = 52
        bold_data = np.random.uniform(80, 120, (*source_shape, t_vol)).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, source_affine)

        dmap = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=source_shape,
            codomain_grid2world=source_affine,
        )
        dmap.forward = np.zeros((*target_shape, 3), dtype=np.float32)
        dmap.backward = np.zeros((*source_shape, 3), dtype=np.float32)

        resampled_img = resample_4d_bold(
            bold_img,
            dmap,
            target_shape=target_shape,
            target_affine=target_affine,
        )

        # Shape must be target shape + T
        self.assertEqual(resampled_img.shape, (*target_shape, t_vol))

        # Affine must be target affine (NOT source affine)
        np.testing.assert_allclose(resampled_img.affine, target_affine, atol=1e-6)

        # Temporal dimension preserved
        self.assertEqual(resampled_img.shape[3], t_vol)

    def test_compose_transformations_prealign_matrix_composition_order(self):
        """prealign must be affine_t1w_to_mni @ rigid_epi_to_t1w (NOT reversed)."""
        # Non-commutative matrices
        rigid_epi_to_t1w = np.eye(4)
        rigid_epi_to_t1w[0, 3] = 7.0

        affine_t1w_to_mni = np.eye(4)
        affine_t1w_to_mni[1, 3] = 4.0
        affine_t1w_to_mni[0, 0] = 0.98  # mild anisotropy scaling

        source_shape = (20, 18, 15)
        source_affine = np.diag([3.5, 3.5, 3.5, 1.0])
        target_shape = (22, 20, 16)
        target_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        t1w_shape = (28, 26, 20)
        t1w_affine = np.diag([1.5, 1.5, 1.5, 1.0])

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=t1w_shape,
            codomain_grid2world=t1w_affine,
        )
        syn_map.forward = np.zeros((*target_shape, 3), dtype=np.float32)
        syn_map.backward = np.zeros((*t1w_shape, 3), dtype=np.float32)

        composed_map = compose_transformations(
            rigid_epi_to_t1w=rigid_epi_to_t1w,
            affine_t1w_to_mni=affine_t1w_to_mni,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        # Pullback prealign_inv order: M_t1w_to_epi @ M_mni_to_t1w = rigid_epi_to_t1w @ affine_t1w_to_mni
        expected_prealign = rigid_epi_to_t1w @ affine_t1w_to_mni
        wrong_order = affine_t1w_to_mni @ rigid_epi_to_t1w

        np.testing.assert_allclose(
            composed_map.prealign_inv, expected_prealign, atol=1e-6,
            err_msg="prealign_inv composition order is wrong",
        )
        # Ensure the test was non-trivial
        self.assertFalse(np.allclose(expected_prealign, wrong_order, atol=1e-6))

    def test_resample_mask_different_source_and_target_resolution(self):
        """Nearest-neighbor mask resampling must preserve discrete labels across resolution change."""
        source_shape = (24, 22, 18)
        source_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        target_shape = (16, 15, 12)
        target_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        labels = np.zeros(source_shape, dtype=np.int32)
        labels[6:12, 5:10, 4:9] = 1   # WM label
        labels[2:5, 2:5, 2:5] = 2     # CSF label
        mask_img = nib.Nifti1Image(labels, source_affine)

        dmap = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=source_shape,
            codomain_grid2world=source_affine,
        )
        dmap.forward = np.zeros((*target_shape, 3), dtype=np.float32)
        dmap.backward = np.zeros((*source_shape, 3), dtype=np.float32)

        resampled = resample_mask_to_target(
            mask_img,
            dmap,
            target_shape=target_shape,
            target_affine=target_affine,
        )

        resampled_data = np.asanyarray(resampled.dataobj)
        self.assertEqual(resampled_data.shape, target_shape)

        unique_vals = set(np.unique(resampled_data))
        self.assertTrue(unique_vals.issubset({0, 1, 2}),
            f"Unexpected label values after nearest-neighbor resampling: {unique_vals}")
        self.assertTrue(np.issubdtype(resampled_data.dtype, np.integer))

    def test_compose_transformations_nonzero_displacement_field(self):
        """Composed map must carry forward the SyN displacement fields unchanged."""
        target_shape = (18, 16, 14)
        target_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        source_shape = (20, 18, 16)
        source_affine = np.diag([3.5, 3.5, 3.5, 1.0])
        t1w_shape = (24, 22, 20)
        t1w_affine = np.diag([1.5, 1.5, 1.5, 1.0])

        forward_field = np.zeros((*target_shape, 3), dtype=np.float32)
        forward_field[..., 0] = 2.0  # 2mm X displacement
        # In DIPY DiffeomorphicMap, both forward and backward deformation fields share disp_shape
        backward_field = np.zeros((*target_shape, 3), dtype=np.float32)
        backward_field[..., 0] = -2.0  # inverse

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=t1w_shape,
            codomain_grid2world=t1w_affine,
        )
        syn_map.forward = forward_field
        syn_map.backward = backward_field

        rigid_epi_to_t1w = np.eye(4)
        affine_t1w_to_mni = np.eye(4)

        composed_map = compose_transformations(
            rigid_epi_to_t1w=rigid_epi_to_t1w,
            affine_t1w_to_mni=affine_t1w_to_mni,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        np.testing.assert_array_equal(composed_map.forward, forward_field)
        np.testing.assert_array_equal(composed_map.backward, backward_field)
        np.testing.assert_array_equal(composed_map.disp_shape, target_shape)
        np.testing.assert_allclose(composed_map.disp_grid2world, target_affine)

    def test_landmark_epi_to_t1w_rigid_mapping(self):
        """Independently verify physical landmark mapping from EPI to T1w with rotation and translation."""
        from dipy.align.imaffine import AffineMap
        from scipy.spatial.transform import Rotation as R_scipy

        # EPI grid: (20, 20, 16) @ 3.0mm
        epi_shape = (20, 20, 16)
        epi_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        # T1w grid: (30, 30, 24) @ 1.5mm
        t1w_shape = (30, 30, 24)
        t1w_affine = np.diag([1.5, 1.5, 1.5, 1.0])

        # Known transformation from static T1w to moving EPI:
        # 15 degree rotation around Y + translation [+3.0, -2.0, 4.0]
        rot_y = R_scipy.from_euler("y", 15, degrees=True).as_matrix()
        M_t1w_to_epi = np.eye(4)
        M_t1w_to_epi[:3, :3] = rot_y
        M_t1w_to_epi[:3, 3] = [3.0, -2.0, 4.0]

        # Forward coordinate transform: x_T1w = inv(M_t1w_to_epi) * x_EPI
        F_epi_to_t1w = np.linalg.inv(M_t1w_to_epi)

        # Place landmark at EPI voxel [8, 10, 6]
        epi_voxel = np.array([8, 10, 6])
        epi_data = np.zeros(epi_shape, dtype=np.float32)
        epi_data[tuple(epi_voxel)] = 1000.0

        # Physical coordinate in EPI space
        x_epi_world = epi_affine @ np.array([*epi_voxel, 1.0])

        # Expected physical coordinate in T1w space
        x_t1w_world = F_epi_to_t1w @ x_epi_world

        # Expected T1w voxel
        expected_t1w_voxel = np.round(np.linalg.inv(t1w_affine) @ x_t1w_world)[:3].astype(int)

        # Apply DIPY AffineMap:
        # In DIPY, AffineMap(affine) takes M_static_to_moving
        affine_map = AffineMap(
            M_t1w_to_epi,
            domain_grid_shape=t1w_shape,
            domain_grid2world=t1w_affine,
            codomain_grid_shape=epi_shape,
            codomain_grid2world=epi_affine,
        )
        warped_t1w = affine_map.transform(epi_data, interpolation="nearest")
        peak_t1w = np.unravel_index(np.argmax(warped_t1w), warped_t1w.shape)

        # Expected voxel must receive the landmark intensity
        self.assertEqual(warped_t1w[tuple(expected_t1w_voxel)], 1000.0)
        # Due to resolution change (3.0mm EPI -> 1.5mm T1w, an EPI voxel covers ~8 T1w voxels),
        # peak coordinate must be within 1 voxel of the expected centroid
        self.assertLessEqual(np.max(np.abs(np.array(peak_t1w) - expected_t1w_voxel)), 1)

    def test_landmark_t1w_to_mni_affine_mapping(self):
        """Independently verify physical landmark mapping from T1w to MNI with scaling and rotation."""
        from dipy.align.imaffine import AffineMap
        from scipy.spatial.transform import Rotation as R_scipy

        # T1w grid: (30, 30, 24) @ 1.5mm
        t1w_shape = (30, 30, 24)
        t1w_affine = np.diag([1.5, 1.5, 1.5, 1.0])

        # MNI target grid: (25, 25, 20) @ 2.0mm
        mni_shape = (25, 25, 20)
        mni_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        # Transformation from static MNI to moving T1w:
        # 10 degree rotation around Z + translation [+2.0, +4.0, -1.0]
        rot_z = R_scipy.from_euler("z", 10, degrees=True).as_matrix()
        M_mni_to_t1w = np.eye(4)
        M_mni_to_t1w[:3, :3] = rot_z
        M_mni_to_t1w[:3, 3] = [2.0, 4.0, -1.0]

        F_t1w_to_mni = np.linalg.inv(M_mni_to_t1w)

        # Place landmark in T1w at voxel [12, 14, 10]
        t1w_voxel = np.array([12, 14, 10])
        t1w_data = np.zeros(t1w_shape, dtype=np.float32)
        t1w_data[tuple(t1w_voxel)] = 1000.0

        x_t1w_world = t1w_affine @ np.array([*t1w_voxel, 1.0])
        x_mni_world = F_t1w_to_mni @ x_t1w_world
        expected_mni_voxel = np.round(np.linalg.inv(mni_affine) @ x_mni_world)[:3].astype(int)

        affine_map = AffineMap(
            M_mni_to_t1w,
            domain_grid_shape=mni_shape,
            domain_grid2world=mni_affine,
            codomain_grid_shape=t1w_shape,
            codomain_grid2world=t1w_affine,
        )
        warped_mni = affine_map.transform(t1w_data, interpolation="nearest")
        peak_mni = np.unravel_index(np.argmax(warped_mni), warped_mni.shape)

        self.assertEqual(tuple(peak_mni), tuple(expected_mni_voxel))

    def test_landmark_complete_composed_epi_to_mni_pipeline(self):
        """Verify complete composed EPI->T1w->MNI single-step landmark propagation."""
        from scipy.spatial.transform import Rotation as R_scipy

        # Target MNI/CC200 grid: (30, 30, 25) @ 3.0mm
        target_shape = (30, 30, 25)
        target_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        # Intermediate T1w: (40, 40, 35) @ 1.5mm
        t1w_shape = (40, 40, 35)
        t1w_affine = np.diag([1.5, 1.5, 1.5, 1.0])

        # Source EPI: (35, 35, 25) @ 3.5mm
        source_shape = (35, 35, 25)
        source_affine = np.diag([3.5, 3.5, 3.5, 1.0])

        # Non-trivial rigid M_t1w_to_epi
        rot_epi = R_scipy.from_euler("y", -10, degrees=True).as_matrix()
        M_t1w_to_epi = np.eye(4)
        M_t1w_to_epi[:3, :3] = rot_epi
        M_t1w_to_epi[:3, 3] = [2.0, -3.0, 4.0]

        # Non-trivial affine M_mni_to_t1w
        rot_t1w = R_scipy.from_euler("z", 12, degrees=True).as_matrix()
        M_mni_to_t1w = np.eye(4)
        M_mni_to_t1w[:3, :3] = rot_t1w
        M_mni_to_t1w[:3, 3] = [3.0, 1.0, -2.0]

        # Forward coordinate transformation: x_MNI = F_t1w_to_mni @ F_epi_to_t1w @ x_EPI
        # Pullback sampling: x_EPI = M_t1w_to_epi @ M_mni_to_t1w @ x_MNI = P @ x_MNI
        P = M_t1w_to_epi @ M_mni_to_t1w
        F = np.linalg.inv(P)

        # Place landmark at EPI voxel [14, 15, 10]
        epi_voxel = np.array([14, 15, 10])
        # 4D BOLD with T=50
        bold_data = np.zeros((*source_shape, 50), dtype=np.float32)
        bold_data[14, 15, 10, :] = 1000.0
        bold_img = nib.Nifti1Image(bold_data, source_affine)

        x_epi_world = source_affine @ np.array([*epi_voxel, 1.0])
        x_target_world = F @ x_epi_world
        expected_target_voxel = np.round(np.linalg.inv(target_affine) @ x_target_world)[:3].astype(int)

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=t1w_shape,
            codomain_grid2world=t1w_affine,
        )
        syn_map.forward = np.zeros((*target_shape, 3), dtype=np.float32)
        syn_map.backward = np.zeros((*target_shape, 3), dtype=np.float32)

        composed_map = compose_transformations(
            rigid_epi_to_t1w=M_t1w_to_epi,
            affine_t1w_to_mni=M_mni_to_t1w,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        resampled_img = resample_4d_bold(
            bold_img,
            composed_map,
            target_shape=target_shape,
            target_affine=target_affine,
            interpolation="nearest",
        )
        resampled_data = np.asanyarray(resampled_img.dataobj)
        peak_voxel = np.unravel_index(np.argmax(resampled_data[..., 0]), target_shape)

        # The expected target voxel must receive the landmark intensity
        self.assertEqual(resampled_data[tuple(expected_target_voxel) + (0,)], 1000.0)
        # Peak voxel coordinate must be within 1 voxel of expected location
        self.assertLessEqual(np.max(np.abs(np.array(peak_voxel) - expected_target_voxel)), 1)

    def test_forward_and_inverse_affine_consistency(self):
        """Verify that forward and pullback transformations are exact algebraic inverses."""
        from scipy.spatial.transform import Rotation as R_scipy

        rot = R_scipy.from_euler("xyz", [12.0, -8.0, 15.0], degrees=True).as_matrix()
        M = np.eye(4)
        M[:3, :3] = rot
        M[:3, 3] = [5.0, -3.0, 7.0]

        F = np.linalg.inv(M)
        np.testing.assert_allclose(M @ F, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(F @ M, np.eye(4), atol=1e-12)

    def test_syn_prealignment_direction_and_alignment(self):
        """Empirically verify DIPY SyN prealignment contract: M_static_to_moving aligns without double-inversion."""
        from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import CCMetric
        from dipy.align.transforms import RigidTransform3D

        # Static reference image (MNI-like)
        static = np.zeros((24, 24, 24), dtype=np.float32)
        static[8:16, 8:16, 8:16] = 1.0

        # Moving image (T1w-like) shifted by +3 voxels in X
        moving = np.zeros((24, 24, 24), dtype=np.float32)
        moving[11:19, 8:16, 8:16] = 1.0

        affine = np.eye(4)
        affreg = AffineRegistration(metric=MutualInformationMetric(nbins=16), level_iters=[20, 10])
        res = affreg.optimize(static, moving, RigidTransform3D(), None, static_grid2world=affine, moving_grid2world=affine)

        # DIPY sdr.optimize takes prealign = res.affine (M_static_to_moving)
        sdr = SymmetricDiffeomorphicRegistration(metric=CCMetric(3), level_iters=[5, 2], step_length=0.25)
        syn = sdr.optimize(static, moving, static_grid2world=affine, moving_grid2world=affine, prealign=res.affine)
        warped = syn.transform(moving)

        dice_correct = compute_spatial_dice(warped > 0.5, static > 0.5)
        self.assertGreater(dice_correct, 0.90, f"Expected high Dice alignment, got {dice_correct:.4f}")

    def test_composed_transformations_actively_applies_syn_displacement(self):
        """Verify that composed_map actively deforms voxel intensities using SyN field under non-identity affine."""
        target_shape = (24, 24, 20)
        target_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        source_shape = (28, 28, 20)
        source_affine = np.diag([3.5, 3.5, 3.5, 1.0])

        # Localized non-zero displacement field in target space at voxel (10, 10, 10)
        disp = np.zeros((*target_shape, 3), dtype=np.float32)
        disp[10, 10, 10, 0] = 10.5  # +10.5mm displacement in X (3 voxels of 3.5mm)

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=target_shape,
            domain_grid2world=target_affine,
            codomain_shape=source_shape,
            codomain_grid2world=source_affine,
        )
        syn_map.forward = disp
        syn_map.backward = -disp

        # Non-identity affine translation (+14mm in Y)
        rigid_epi_to_t1w = np.eye(4)
        rigid_epi_to_t1w[1, 3] = 14.0

        composed_map = compose_transformations(
            rigid_epi_to_t1w=rigid_epi_to_t1w,
            affine_t1w_to_mni=np.eye(4),
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        # Continuous gradient in source image
        x, y, z = np.indices(source_shape)
        src_data = (x * 10.0 + y + z).astype(np.float32)

        warped_with_disp = composed_map.transform(src_data)

        # Now zero out displacement and compare
        syn_map_zero = DiffeomorphicMap(
            3,
            disp_shape=target_shape,
            disp_grid2world=target_affine,
            domain_shape=source_shape,
            domain_grid2world=source_affine,
            codomain_shape=target_shape,
            codomain_grid2world=target_affine,
        )
        syn_map_zero.forward = np.zeros_like(disp)
        syn_map_zero.backward = np.zeros_like(disp)

        composed_map_zero = compose_transformations(
            rigid_epi_to_t1w=rigid_epi_to_t1w,
            affine_t1w_to_mni=np.eye(4),
            syn_mapping=syn_map_zero,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )
        warped_zero = composed_map_zero.transform(src_data)

        diff = float(np.max(np.abs(warped_with_disp - warped_zero)))
        self.assertGreater(diff, 5.0, "Displacement field must produce demonstrable change in warped volume")

    def test_nonlinear_displacement_composition_against_independent_sequential_evaluation(self):
        """Validate composed nonlinear transform against independent sequential coordinate evaluation.

        Tests:
        - Non-identity T1w-to-MNI affine with rotation (8 deg around Z), translation (+3, -4, +2), scaling (1.02, 0.98, 1.01).
        - Non-identity EPI-to-T1w rigid transform with rotation (-6 deg around X), translation (+2, +3, -1).
        - Smooth, non-zero nonlinear displacement field (Gaussian bump).
        - Independent sequential coordinate evaluation:
            x_target -> x_displaced = x_target + d_target -> x_T1w -> x_EPI -> i_EPI
        - Verifies composed_map.transform(source_vol) maps source peak located at i_EPI to exact target voxel (16, 16, 14).
        - Verifies forward / inverse consistency with composed_map.transform_inverse().
        - Verifies field lookup coordinates, displacement-vector orientation, and units.
        """
        target_shape = (32, 32, 28)
        target_affine = np.array([
            [-3.0, 0, 0, 48.0],
            [0, 3.0, 0, -48.0],
            [0, 0, 3.0, -30.0],
            [0, 0, 0, 1.0],
        ], dtype=np.float64)

        source_shape = (28, 28, 24)
        source_affine = np.array([
            [3.5, 0, 0, -45.0],
            [0, 3.5, 0, -45.0],
            [0, 0, 3.5, -40.0],
            [0, 0, 0, 1.0],
        ], dtype=np.float64)

        # T1w-to-MNI affine: rotation 8 deg around Z, scaling, translation
        theta_z = np.deg2rad(8.0)
        R_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0, 0],
            [np.sin(theta_z),  np.cos(theta_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        S = np.diag([1.02, 0.98, 1.01, 1.0])
        T_mni = np.eye(4)
        T_mni[:3, 3] = [3.0, -4.0, 2.0]
        A_t1w_to_mni_fwd = T_mni @ R_z @ S
        M_mni_to_t1w = np.linalg.inv(A_t1w_to_mni_fwd)

        # EPI-to-T1w rigid: rotation -6 deg around X, translation
        theta_x = np.deg2rad(-6.0)
        R_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x), 0],
            [0, np.sin(theta_x),  np.cos(theta_x), 0],
            [0, 0, 0, 1],
        ])
        T_t1w = np.eye(4)
        T_t1w[:3, 3] = [2.0, 3.0, -1.0]
        R_epi_to_t1w_fwd = T_t1w @ R_x
        M_t1w_to_epi = np.linalg.inv(R_epi_to_t1w_fwd)

        # Smooth Gaussian SyN displacement bump centered at target voxel (16, 16, 14)
        disp_shape = target_shape
        gx, gy, gz = np.indices(disp_shape)
        r2 = (gx - 16)**2 + (gy - 16)**2 + (gz - 14)**2
        sigma = 4.0
        bump = np.exp(-r2 / (2 * sigma**2)).astype(np.float32)

        disp_backward = np.zeros((*disp_shape, 3), dtype=np.float32)
        disp_backward[..., 0] = bump * 4.5   # +4.5mm in X
        disp_backward[..., 1] = bump * -3.0  # -3.0mm in Y
        disp_backward[..., 2] = bump * 2.0   # +2.0mm in Z

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

        composed_map = compose_transformations(
            rigid_epi_to_t1w=M_t1w_to_epi,
            affine_t1w_to_mni=M_mni_to_t1w,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=source_shape,
            source_affine=source_affine,
        )

        # Independent sequential coordinate calculation for target landmark (16, 16, 14):
        target_voxel = np.array([16.0, 16.0, 14.0, 1.0])
        x_target = target_affine @ target_voxel
        d_target = np.array([4.5, -3.0, 2.0, 0.0])  # displacement at center of Gaussian bump
        x_displaced = x_target + d_target
        x_t1w = M_mni_to_t1w @ x_displaced
        x_epi = M_t1w_to_epi @ x_t1w
        i_epi = np.linalg.inv(source_affine) @ x_epi

        # Place landmark peak at independently computed continuous coordinates i_epi[:3]
        sx, sy, sz = np.indices(source_shape)
        sr2 = (sx - i_epi[0])**2 + (sy - i_epi[1])**2 + (sz - i_epi[2])**2
        source_vol = (100.0 * np.exp(-sr2 / 2.0)).astype(np.float32)

        warped = composed_map.transform(source_vol)
        peak_at_target = warped[16, 16, 14]
        max_idx = np.unravel_index(np.argmax(warped), warped.shape)

        # Maximum of warped volume must land at the exact target voxel
        self.assertEqual(max_idx, (16, 16, 14))
        self.assertGreater(peak_at_target, 80.0)

        # Forward/Inverse shape and consistency verification
        tgt_vol = np.ones(target_shape, dtype=np.float32)
        inv_warped = composed_map.transform_inverse(tgt_vol)
        self.assertEqual(inv_warped.shape, source_shape)
        self.assertEqual(warped.shape, target_shape)



if __name__ == "__main__":
    unittest.main()


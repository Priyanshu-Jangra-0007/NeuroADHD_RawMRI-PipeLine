"""Unit tests for spatial coregistration, normalization, and composed resampling."""
from __future__ import annotations

import unittest
import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap, AffineRegistration, MutualInformationMetric, transform_centers_of_mass
from dipy.align.transforms import RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration, DiffeomorphicMap
from dipy.align.metrics import CCMetric
from scipy.spatial.transform import Rotation as R_scipy

from src.preprocessing.motion import apply_motion_transforms, estimate_motion_parameters
from src.preprocessing.registration import (
    check_process_memory_gb,
    compose_transformations,
    compose_volume_transform,
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

    def test_composed_volume_transform_single_step_motion_resampling(self):
        """Composing per-volume motion_transforms with composed_map executes single-step pullback accurately."""
        shape = (24, 24, 24)
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        t_vol = 52
        # Build EPI with alternating sub-voxel / voxel shift across volumes and known EPI->T1w->MNI offset
        mni = np.zeros(shape, dtype=np.float32)
        mni[7:15, 7:15, 7:15] = 100.0

        epi_4d = np.zeros((*shape, t_vol), dtype=np.float32)
        for t in range(t_vol):
            shift = 1 if (t % 2 == 1) else 0
            epi_4d[8 + shift : 16 + shift, 7:15, 7:15, t] = 100.0 + 0.1 * (t % 4)
        epi_img = nib.Nifti1Image(epi_4d, affine)

        transforms, motion_params, fd = estimate_motion_parameters(epi_img)
        self.assertEqual(len(transforms), t_vol)
        self.assertEqual(motion_params.shape, (t_vol, 6))
        self.assertEqual(fd.shape, (t_vol,))

        # Single identity SyN + known 1-voxel pullback from EPI mean center (~x+1.5mm..3mm) to MNI
        mean_epi = nib.Nifti1Image(np.mean(epi_4d, axis=3), affine)
        mni_img = nib.Nifti1Image(mni, affine)
        m_epi_to_mni, _ = register_epi_to_t1w(mean_epi, mni_img, level_iters=[20])

        id_syn = DiffeomorphicMap(
            3,
            disp_shape=shape,
            disp_grid2world=affine,
            domain_shape=shape,
            domain_grid2world=affine,
            codomain_shape=shape,
            codomain_grid2world=affine,
        )
        id_syn.is_inverse = True
        id_syn.backward = np.zeros((*shape, 3), dtype=np.float32)
        id_syn.forward = np.zeros((*shape, 3), dtype=np.float32)

        composed = compose_transformations(
            rigid_epi_to_t1w=m_epi_to_mni,
            affine_t1w_to_mni=np.eye(4, dtype=np.float64),
            syn_mapping=id_syn,
            target_shape=shape,
            target_affine=affine,
            source_shape=shape,
            source_affine=affine,
        )

        # Single-step composed resampling (Motion + EPI-to-MNI)
        vol0_map = compose_volume_transform(composed, transforms[0])
        np.testing.assert_allclose(
            vol0_map.prealign_inv,
            np.asarray(transforms[0].affine) @ np.asarray(composed.prealign_inv),
            atol=1e-10,
        )
        single_step = resample_4d_bold(
            epi_img, composed, target_shape=shape, target_affine=affine, motion_transforms=transforms
        )
        single_data = np.asanyarray(single_step.dataobj)
        self.assertEqual(single_step.shape, (*shape, t_vol))
        np.testing.assert_allclose(single_step.affine, affine)

        # Two-step baseline for comparison: resampled once for motion, once for coreg
        realigned = apply_motion_transforms(epi_img, transforms)
        two_step = resample_4d_bold(realigned, composed, target_shape=shape, target_affine=affine)

        # Verify single-step realigns to MNI target accurately and preserves sharper interior variance (less blur)
        single_mean = np.mean(single_data, axis=3)
        self.assertGreater(compute_spatial_dice(single_mean > 50, mni > 50), 0.85)
        two_step_mean = np.mean(np.asanyarray(two_step.dataobj), axis=3)
        # Fewer interpolations retain higher image contrast gradient (std across intensities)
        self.assertGreaterEqual(float(np.std(single_mean)), float(np.std(two_step_mean)) - 1e-4)

        # Mismatched length must fail closed
        with self.assertRaises(ValueError):
            resample_4d_bold(epi_img, composed, target_shape=shape, target_affine=affine, motion_transforms=transforms[:-1])

    def test_resample_mask_nearest_interpolation(self):
        # Create discrete label image with values 0, 1, 2, 5
        labels_3d = np.zeros((20, 20, 20), dtype=np.int32)
        labels_3d[5:10, 5:10, 5:10] = 1
        labels_3d[10:15, 10:15, 10:15] = 2
        labels_3d[15:18, 15:18, 15:18] = 5
        mask_img = nib.Nifti1Image(labels_3d, np.eye(4))

        # Identity diffeomorphic map
        dmap = DiffeomorphicMap(
            3,
            disp_shape=(20, 20, 20),
            domain_shape=(20, 20, 20),
            domain_grid2world=np.eye(4),
            codomain_shape=(20, 20, 20),
            codomain_grid2world=np.eye(4),
        )
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
            domain_shape=source_shape,
            domain_grid2world=source_affine,
            codomain_shape=target_shape,
            codomain_grid2world=target_affine,
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
            domain_shape=source_shape,
            domain_grid2world=source_affine,
            codomain_shape=target_shape,
            codomain_grid2world=target_affine,
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



class TestTissueMaskSpatialGeometry(unittest.TestCase):
    """Rigorous unit tests for tissue mask spatial geometry enforcement."""

    def setUp(self):
        self.t1_shape = (40, 40, 40)
        self.t1_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        self.epi_shape = (30, 30, 20)
        self.epi_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        self.target_shape = (30, 30, 20)
        self.target_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        # Valid T1w tissue mask
        mask_data = np.zeros(self.t1_shape, dtype=np.int32)
        mask_data[10:30, 10:30, 10:30] = 1
        self.t1_mask = nib.Nifti1Image(mask_data, self.t1_affine)

        # Valid SyN map for T1w -> MNI
        self.syn_map = DiffeomorphicMap(
            3,
            disp_shape=self.target_shape,
            disp_grid2world=self.target_affine,
            domain_shape=self.t1_shape,
            domain_grid2world=self.t1_affine,
            codomain_shape=self.target_shape,
            codomain_grid2world=self.target_affine,
        )
        self.syn_map.forward = np.zeros((*self.target_shape, 3), dtype=np.float32)
        self.syn_map.backward = np.zeros((*self.t1_shape, 3), dtype=np.float32)

        # Composed map for EPI -> MNI (domain is EPI)
        self.composed_map = DiffeomorphicMap(
            3,
            disp_shape=self.target_shape,
            disp_grid2world=self.target_affine,
            domain_shape=self.epi_shape,
            domain_grid2world=self.epi_affine,
            codomain_shape=self.target_shape,
            codomain_grid2world=self.target_affine,
        )
        self.composed_map.forward = np.zeros((*self.target_shape, 3), dtype=np.float32)
        self.composed_map.backward = np.zeros((*self.epi_shape, 3), dtype=np.float32)

    def test_mask_resample_dimension_mismatch_raises_valueerror(self):
        """resample_mask_to_target must raise ValueError if mask shape does not match map domain shape."""
        mismatched_data = np.zeros((35, 40, 40), dtype=np.int32)
        mismatched_mask = nib.Nifti1Image(mismatched_data, self.t1_affine)

        with self.assertRaises(ValueError) as ctx:
            resample_mask_to_target(mismatched_mask, self.syn_map, self.target_shape, self.target_affine)
        self.assertIn("Spatial dimension mismatch", str(ctx.exception))

    def test_mask_resample_translation_offset_mismatch_raises_valueerror(self):
        """resample_mask_to_target must raise ValueError on origin translation offset discrepancy."""
        shifted_affine = self.t1_affine.copy()
        shifted_affine[0, 3] = 25.0  # Shift X origin by 25 mm
        shifted_mask = nib.Nifti1Image(np.zeros(self.t1_shape, dtype=np.int32), shifted_affine)

        with self.assertRaises(ValueError) as ctx:
            resample_mask_to_target(shifted_mask, self.syn_map, self.target_shape, self.target_affine)
        self.assertIn("Translation offset mismatch", str(ctx.exception))

    def test_mask_resample_voxel_size_scaling_mismatch_raises_valueerror(self):
        """resample_mask_to_target must raise ValueError on voxel-size scaling discrepancy."""
        scaled_affine = np.diag([2.5, 2.0, 2.0, 1.0])  # X voxel is 2.5 mm instead of 2.0 mm
        scaled_mask = nib.Nifti1Image(np.zeros(self.t1_shape, dtype=np.int32), scaled_affine)

        with self.assertRaises(ValueError) as ctx:
            resample_mask_to_target(scaled_mask, self.syn_map, self.target_shape, self.target_affine)
        self.assertIn("Voxel-size scaling mismatch", str(ctx.exception))

    def test_mask_resample_orientation_mismatch_raises_valueerror(self):
        """resample_mask_to_target must raise ValueError on oblique rotation discrepancy."""
        theta = np.deg2rad(15.0)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rotated_affine = rot @ self.t1_affine
        rotated_mask = nib.Nifti1Image(np.zeros(self.t1_shape, dtype=np.int32), rotated_affine)

        with self.assertRaises(ValueError) as ctx:
            resample_mask_to_target(rotated_mask, self.syn_map, self.target_shape, self.target_affine)
        self.assertIn("orientation", str(ctx.exception).lower())

    def test_mask_resample_composed_map_rejects_t1w_mask(self):
        """resample_mask_to_target must reject passing T1w mask to composed EPI-to-MNI map."""
        with self.assertRaises(ValueError) as ctx:
            resample_mask_to_target(self.t1_mask, self.composed_map, self.target_shape, self.target_affine)
        self.assertIn("Anatomical T1w masks must use T1w-to-MNI transform", str(ctx.exception))

    def test_mask_resample_missing_domain_geometry_raises_valueerror(self):
        """resample_mask_to_target must raise ValueError if map lacks domain geometry."""
        empty_map = DiffeomorphicMap(3, disp_shape=(20, 20, 20))
        with self.assertRaises(ValueError) as ctx:
            resample_mask_to_target(self.t1_mask, empty_map, self.target_shape, self.target_affine)
        self.assertIn("lacks source domain geometry", str(ctx.exception))

    def test_mask_resample_syn_map_accepts_matching_t1w_mask(self):
        """resample_mask_to_target succeeds and preserves nearest-neighbor labels when geometry matches."""
        resampled = resample_mask_to_target(self.t1_mask, self.syn_map, self.target_shape, self.target_affine)
        self.assertEqual(resampled.shape, self.target_shape)
        np.testing.assert_allclose(resampled.affine, self.target_affine, atol=1e-4)
        unique_vals = np.unique(np.asanyarray(resampled.dataobj))
        self.assertTrue(set(unique_vals).issubset({0, 1}))


class TestComposedTransformPhantomValidation(unittest.TestCase):
    """Independent geometric landmark phantom validation of composed transforms & DIPY conventions."""

    def setUp(self):
        self.epi_shape = (32, 32, 24)
        self.epi_aff = np.diag([3.0, 3.0, 4.0, 1.0])
        self.epi_aff[:3, 3] = [-48.0, -48.0, -44.0]
        self.t1_shape = (40, 40, 32)
        self.t1_aff = np.diag([2.0, 2.0, 2.0, 1.0])
        self.t1_aff[:3, 3] = [-40.0, -40.0, -32.0]
        self.mni_shape = (36, 36, 30)
        self.mni_aff = np.array(
            [
                [-2.5, 0.0, 0.0, 45.0],
                [0.0, 2.5, 0.0, -45.0],
                [0.0, 0.0, 2.5, -37.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # 5 well-separated asymmetric EPI voxel landmarks spanning all 3D quadrants
        self.epi_pts_vox = np.array(
            [[10, 12, 8], [20, 11, 14], [12, 22, 10], [21, 19, 13], [16, 16, 12]],
            dtype=np.float64,
        )
        self.epi_pts_mm = np.array(
            [(self.epi_aff @ np.array([*p, 1.0]))[:3] for p in self.epi_pts_vox],
            dtype=np.float64,
        )

    @staticmethod
    def _rigid(euler_deg, trans_mm) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = R_scipy.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        m[:3, 3] = trans_mm
        return m

    @staticmethod
    def _build_diffeo_pair(shape, affine, zero_disp: bool = True):
        if zero_disp:
            fwd = np.zeros((*shape, 3), dtype=np.float32)
            bwd = np.zeros((*shape, 3), dtype=np.float32)
            return fwd, bwd, (lambda pts: np.zeros_like(pts, dtype=np.float64))

        amp = np.array([3.5, -2.5, 2.0], dtype=np.float64)
        sigma_d = 18.0

        def u_fwd(pts: np.ndarray) -> np.ndarray:
            pts_arr = np.asarray(pts, dtype=np.float64)
            r2 = np.sum(pts_arr ** 2, axis=-1, keepdims=True)
            return amp * np.exp(-r2 / (2.0 * sigma_d ** 2))

        mx, my, mz = np.indices(shape)
        ijk = np.stack([mx, my, mz, np.ones_like(mx)], axis=-1)
        x_mni = np.einsum("ij,...j->...i", affine, ijk)[..., :3]
        fwd = u_fwd(x_mni).astype(np.float32)

        x_p = x_mni.copy()
        for _ in range(25):
            x_p = x_mni - u_fwd(x_p)
        bwd = (x_p - x_mni).astype(np.float32)
        return fwd, bwd, u_fwd

    def _evaluate_regime(
        self,
        f_mot: np.ndarray,
        f_epi_t1: np.ndarray,
        f_t1_mni: np.ndarray,
        zero_disp: bool = True,
    ):
        m_ref_to_vol = np.linalg.inv(f_mot)
        m_t1_to_epi = np.linalg.inv(f_epi_t1)
        m_mni_to_t1 = np.linalg.inv(f_t1_mni)

        fwd_f, bwd_f, u_fwd_fn = self._build_diffeo_pair(self.mni_shape, self.mni_aff, zero_disp=zero_disp)
        syn = DiffeomorphicMap(
            3,
            disp_shape=self.mni_shape,
            disp_grid2world=self.mni_aff,
            domain_shape=self.mni_shape,
            domain_grid2world=self.mni_aff,
            codomain_shape=self.t1_shape,
            codomain_grid2world=self.t1_aff,
        )
        syn.is_inverse = True
        syn.forward = fwd_f
        syn.backward = bwd_f

        cmap = compose_transformations(
            m_t1_to_epi,
            m_mni_to_t1,
            syn,
            target_shape=self.mni_shape,
            target_affine=self.mni_aff,
            source_shape=self.epi_shape,
            source_affine=self.epi_aff,
        )
        vmap = compose_volume_transform(cmap, m_ref_to_vol)

        # Synthetic 3D native EPI landmark volume
        gx, gy, gz = np.indices(self.epi_shape)
        epi_vol = np.zeros(self.epi_shape, dtype=np.float32)
        for k, p in enumerate(self.epi_pts_vox):
            r2 = ((gx - p[0]) * 3.0) ** 2 + ((gy - p[1]) * 3.0) ** 2 + ((gz - p[2]) * 4.0) ** 2
            epi_vol += (100.0 + 10.0 * k) * np.exp(-r2 / (2.0 * 4.5 ** 2))

        warped_mni = vmap.transform(
            epi_vol, interpolation="linear", out_shape=self.mni_shape, out_grid2world=self.mni_aff
        )

        # Analytical target coordinates in MNI mm + MNI inverse landmark phantom
        f_aff = f_t1_mni @ f_epi_t1 @ f_mot
        x_p = np.array([(f_aff @ np.array([*x, 1.0]))[:3] for x in self.epi_pts_mm])
        x_mni_true = x_p + u_fwd_fn(x_p)

        inv_mni = np.linalg.inv(self.mni_aff)
        mx, my, mz = np.indices(self.mni_shape)
        mni_vol = np.zeros(self.mni_shape, dtype=np.float32)
        for k, xm in enumerate(x_mni_true):
            iv = (inv_mni @ np.array([*xm, 1.0]))[:3]
            r2 = ((mx - iv[0]) * 2.5) ** 2 + ((my - iv[1]) * 2.5) ** 2 + ((mz - iv[2]) * 2.5) ** 2
            mni_vol += (100.0 + 10.0 * k) * np.exp(-r2 / (2.0 * 4.5 ** 2))

        recovered_epi = vmap.transform_inverse(
            mni_vol, interpolation="linear", out_shape=self.epi_shape, out_grid2world=self.epi_aff
        )

        errs_fwd_mm, errs_inv_mm = [], []
        mni_zooms = np.linalg.norm(self.mni_aff[:3, :3], axis=0)
        epi_zooms = np.linalg.norm(self.epi_aff[:3, :3], axis=0)
        errs_fwd_vox, errs_inv_vox = [], []
        for k in range(len(self.epi_pts_vox)):
            xm = x_mni_true[k]
            iv_exp = (inv_mni @ np.array([*xm, 1.0]))[:3]
            mask_m = ((mx - iv_exp[0]) ** 2 + (my - iv_exp[1]) ** 2 + (mz - iv_exp[2]) ** 2) <= 9.0
            wm = np.clip(warped_mni * mask_m, 0.0, None)
            com_m_vox = np.array([np.sum(mx * wm), np.sum(my * wm), np.sum(mz * wm)]) / np.sum(wm)
            com_m_mm = (self.mni_aff @ np.array([*com_m_vox, 1.0]))[:3]
            errs_fwd_mm.append(float(np.linalg.norm(com_m_mm - xm)))
            errs_fwd_vox.append(float(np.linalg.norm((com_m_mm - xm) / mni_zooms)))

            p = self.epi_pts_vox[k]
            mask_e = ((gx - p[0]) ** 2 + (gy - p[1]) ** 2 + (gz - p[2]) ** 2) <= 9.0
            we = np.clip(recovered_epi * mask_e, 0.0, None)
            com_e_vox = np.array([np.sum(gx * we), np.sum(gy * we), np.sum(gz * we)]) / np.sum(we)
            com_e_mm = (self.epi_aff @ np.array([*com_e_vox, 1.0]))[:3]
            errs_inv_mm.append(float(np.linalg.norm(com_e_mm - self.epi_pts_mm[k])))
            errs_inv_vox.append(float(np.linalg.norm((com_e_mm - self.epi_pts_mm[k]) / epi_zooms)))

        pt_fwd_err = float(
            np.max(np.linalg.norm(vmap.transform_points_inverse(self.epi_pts_mm) - x_mni_true, axis=1))
        )
        pt_inv_err = float(
            np.max(np.linalg.norm(vmap.transform_points(x_mni_true) - self.epi_pts_mm, axis=1))
        )
        return {
            "vmap": vmap,
            "cmap": cmap,
            "epi_vol": epi_vol,
            "warped_mni": warped_mni,
            "x_mni_true": x_mni_true,
            "mean_fwd_err_mm": float(np.mean(errs_fwd_mm)),
            "max_fwd_err_mm": float(np.max(errs_fwd_mm)),
            "max_fwd_err_vox": float(np.max(errs_fwd_vox)),
            "mean_inv_err_mm": float(np.mean(errs_inv_mm)),
            "max_inv_err_mm": float(np.max(errs_inv_mm)),
            "max_inv_err_vox": float(np.max(errs_inv_vox)),
            "point_fwd_max_err_mm": pt_fwd_err,
            "point_inv_max_err_mm": pt_inv_err,
        }

    def test_identity_translation_rotation_and_combined_rigid_landmarks(self):
        """Verify identity, pure translation, pure rotation, and combined rigid composed transforms."""
        i4 = np.eye(4, dtype=np.float64)
        regimes = {
            "identity": (i4, i4, i4),
            "pure_translation": (
                self._rigid([0.0, 0.0, 0.0], [3.5, -2.5, 4.0]),
                self._rigid([0.0, 0.0, 0.0], [-2.0, 4.0, -1.5]),
                i4,
            ),
            "pure_rotation": (
                self._rigid([8.0, -10.0, 12.0], [0.0, 0.0, 0.0]),
                self._rigid([-5.0, 7.0, -9.0], [0.0, 0.0, 0.0]),
                i4,
            ),
            "combined_rigid": (
                self._rigid([9.0, -7.0, 11.0], [3.0, -2.0, 3.5]),
                self._rigid([-11.0, 8.0, -6.0], [-3.5, 4.0, -2.0]),
                i4,
            ),
        }
        for name, (fm, fe, ft) in regimes.items():
            metrics = self._evaluate_regime(fm, fe, ft, zero_disp=True)
            self.assertLess(metrics["max_fwd_err_mm"], 0.25, f"{name} fwd mm err too high")
            self.assertLess(metrics["max_inv_err_mm"], 0.35, f"{name} inv mm err too high")
            self.assertLess(metrics["max_fwd_err_vox"], 0.15, f"{name} fwd vox err too high")
            self.assertLess(metrics["point_fwd_max_err_mm"], 1e-6, f"{name} point fwd mismatch")
            self.assertLess(metrics["point_inv_max_err_mm"], 1e-6, f"{name} point inv mismatch")

    def test_non_commuting_composed_syn_forward_inverse_and_swapped_order_fails(self):
        """Non-commuting motion+coreg+norm+SyN must agree analytically & with sequential resampling; swapped order fails."""
        f_mot = self._rigid([11.0, 0.0, 0.0], [3.5, -2.0, 4.0])
        f_epi_t1 = self._rigid([0.0, -13.0, 0.0], [-4.0, 3.0, -2.5])
        f_t1_mni = self._rigid([0.0, 0.0, 17.0], [2.0, -5.0, 3.0]) @ np.diag([1.04, 0.96, 1.02, 1.0])

        res = self._evaluate_regime(f_mot, f_epi_t1, f_t1_mni, zero_disp=False)
        self.assertLess(res["mean_fwd_err_mm"], 0.25)
        self.assertLess(res["max_fwd_err_mm"], 0.30)
        self.assertLess(res["max_fwd_err_vox"], 0.15)
        self.assertLess(res["mean_inv_err_mm"], 0.30)
        self.assertLess(res["max_inv_err_mm"], 0.45)
        self.assertLess(res["max_inv_err_vox"], 0.20)
        self.assertLess(res["point_fwd_max_err_mm"], 0.05)
        self.assertLess(res["point_inv_max_err_mm"], 0.05)

        # Compare single-step composed resampling vs sequential (motion AffineMap -> composed coreg+norm+SyN)
        mot_map = AffineMap(
            np.linalg.inv(f_mot),
            domain_grid_shape=self.epi_shape,
            domain_grid2world=self.epi_aff,
            codomain_grid_shape=self.epi_shape,
            codomain_grid2world=self.epi_aff,
        )
        seq_step1 = mot_map.transform(res["epi_vol"], interpolation="linear")
        seq_mni = res["cmap"].transform(
            seq_step1, interpolation="linear", out_shape=self.mni_shape, out_grid2world=self.mni_aff
        )
        inv_mni = np.linalg.inv(self.mni_aff)
        mx, my, mz = np.indices(self.mni_shape)
        seq_vs_single_diffs = []
        for k, xm in enumerate(res["x_mni_true"]):
            iv = (inv_mni @ np.array([*xm, 1.0]))[:3]
            mask = ((mx - iv[0]) ** 2 + (my - iv[1]) ** 2 + (mz - iv[2]) ** 2) <= 9.0
            w_s = np.clip(res["warped_mni"] * mask, 0.0, None)
            w_q = np.clip(seq_mni * mask, 0.0, None)
            c_s = (self.mni_aff @ np.array([np.sum(mx * w_s), np.sum(my * w_s), np.sum(mz * w_s), np.sum(w_s)]) / np.sum(w_s))[:3]
            c_q = (self.mni_aff @ np.array([np.sum(mx * w_q), np.sum(my * w_q), np.sum(mz * w_q), np.sum(w_q)]) / np.sum(w_q))[:3]
            seq_vs_single_diffs.append(float(np.linalg.norm(c_s - c_q)))
        self.assertLess(float(np.max(seq_vs_single_diffs)), 0.15)

        # Negative controls: swapping EITHER motion order OR coreg/norm order must produce large (> 1.5 mm) errors
        m_ref_to_vol = np.linalg.inv(f_mot)
        wrong_motion_map = DiffeomorphicMap(
            3,
            disp_shape=res["vmap"].disp_shape,
            disp_grid2world=res["vmap"].disp_grid2world,
            domain_shape=res["vmap"].domain_shape,
            domain_grid2world=res["vmap"].domain_grid2world,
            codomain_shape=res["vmap"].codomain_shape,
            codomain_grid2world=res["vmap"].codomain_grid2world,
            prealign=np.linalg.inv(res["cmap"].prealign_inv @ m_ref_to_vol),  # SWAPPED order
        )
        wrong_motion_map.is_inverse = True
        wrong_motion_map.forward = res["vmap"].forward
        wrong_motion_map.backward = res["vmap"].backward
        wrong_mot_pts = wrong_motion_map.transform_points_inverse(self.epi_pts_mm)
        self.assertGreater(float(np.max(np.linalg.norm(wrong_mot_pts - res["x_mni_true"], axis=1))), 1.5)

        wrong_coreg_cmap = compose_transformations(
            np.linalg.inv(f_t1_mni),  # SWAPPED order
            np.linalg.inv(f_epi_t1),
            res["vmap"],
            target_shape=self.mni_shape,
            target_affine=self.mni_aff,
            source_shape=self.epi_shape,
            source_affine=self.epi_aff,
        )
        wrong_coreg_vmap = compose_volume_transform(wrong_coreg_cmap, m_ref_to_vol)
        wrong_coreg_pts = wrong_coreg_vmap.transform_points_inverse(self.epi_pts_mm)
        self.assertGreater(float(np.max(np.linalg.norm(wrong_coreg_pts - res["x_mni_true"], axis=1))), 1.5)

        # Singular motion transform fails closed
        with self.assertRaises(ValueError):
            compose_volume_transform(res["cmap"], np.zeros((4, 4), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()


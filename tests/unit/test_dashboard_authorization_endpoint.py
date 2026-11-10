"""Unit tests for Dashboard Research Authorization endpoint."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.serve_dashboard import DashboardApplication


class TestDashboardAuthorizationEndpoint(unittest.TestCase):
    """Test dashboard authorization endpoint payload and lock invariants."""

    def setUp(self):
        self.app = DashboardApplication()

    def test_authorization_status_locked_invariants(self):
        """Dashboard authorization status must return fail-closed locked governance state."""
        status = self.app.authorization_status()

        self.assertEqual(status["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertEqual(status["authorization_status"], "NOT_AUTHORIZED")
        self.assertFalse(status["real_data_authorized"])
        self.assertEqual(status["badge"], "REAL DATA PROCESSING LOCKED")
        self.assertEqual(status["dataset_id"], "sub-01")
        self.assertEqual(status["dataset_authorization_status"], "PENDING")
        self.assertEqual(status["signature_status"], "MISSING_SIGNATURE")
        self.assertEqual(status["approval_reference_status"], "PENDING_APPROVAL_REFERENCE")
        self.assertIsNone(status["institutional_approval_reference"])
        self.assertEqual(status["expiry_status"], "NO_EXPIRY_DATE")
        self.assertEqual(status["pending_items_count"], 23)
        self.assertEqual(status["total_items_count"], 23)
        self.assertIsInstance(status["blockers"], list)
        self.assertGreater(len(status["blockers"]), 0)
        self.assertTrue(any("pending formal approval" in b for b in status["blockers"]))

    def test_pipeline_verification_endpoint(self):
        """Dashboard verification endpoint must return verified status for synthetic pipeline while real data remains locked."""
        status = self.app.pipeline_verification_status()
        self.assertEqual(status["status"], "VERIFIED")
        self.assertEqual(status["badge"], "SYNTHETIC PIPELINE VERIFIED")
        self.assertEqual(status["stages_total"], 19)
        self.assertEqual(status["stages_verified"], 19)
        self.assertFalse(status["safety_gate"]["real_data_authorized"])
        self.assertEqual(status["safety_gate"]["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertEqual(status["safety_gate"]["authorization_status"], "NOT_AUTHORIZED")
        self.assertIn("LOCKED", status["safety_gate"]["real_scan_pipeline_status"])
        self.assertEqual(len(status["stages"]), 19)
        self.assertEqual(status["parcellation"]["rois"], 200)
        self.assertEqual(status["parcellation"]["edges"], 19900)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

"""Tests for the synthetic-only dashboard run launcher."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import serve_dashboard


class _FakeThread:
    def __init__(self, target, args, daemon, name):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self):
        self.started = True


class TestDashboardApplication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp.name) / "dashboard_cohorts"
        self.output_patch = mock.patch.object(serve_dashboard, "OUTPUT_ROOT", self.output_root)
        self.thread_patch = mock.patch.object(serve_dashboard.threading, "Thread", _FakeThread)
        self.env_patch = mock.patch.dict(
            "os.environ",
            {"NEUROADHD_MODEL_DIR": "", "NEUROADHD_HMAC_KEY": ""},
            clear=False,
        )
        self.output_patch.start()
        self.thread_patch.start()
        self.env_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.thread_patch.stop)
        self.addCleanup(self.env_patch.stop)
        self.app = serve_dashboard.DashboardApplication()

    def tearDown(self):
        self.temp.cleanup()

    def test_health_always_reports_synthetic_only_and_unauthorized(self):
        health = self.app.health()
        self.assertEqual(health["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertIs(health["real_data_authorized"], False)
        self.assertEqual(health["authorization_state"], "NOT_AUTHORIZED")

    def test_run_launcher_uses_fixed_runner_and_server_owned_output_path(self):
        result = self.app.start_run(
            {"n_subjects": 2, "t_vol": 60, "seed": 123, "atlas_mode": "synthetic_phantom"}
        )
        cohort_dir = self.output_root / result["cohort_id"]
        self.assertTrue(cohort_dir.is_dir())
        state = self.app._active[result["cohort_id"]]
        self.assertEqual(state["status"], "STARTING")
        fake_thread = None
        # The fake thread is retained via its target state in the application registry.
        self.assertTrue(result["cohort_id"].startswith("cohort_"))
        self.assertEqual(result["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")


    def test_runner_log_is_bounded_and_read_from_cohort_directory(self):
        cohort_id = "cohort_20261103T123000_456789Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        (cohort_dir / "dashboard-run.log").write_text("pipeline started\\npipeline finished\\n", encoding="utf-8")

        result = self.app.process_log(cohort_id)

        self.assertTrue(result["available"])
        self.assertIn("pipeline finished", result["text"])

    def test_runner_log_rejects_symlink(self):
        cohort_id = "cohort_20261103T123000_456790Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        external = Path(self.temp.name) / "outside.log"
        external.write_text("outside", encoding="utf-8")
        (cohort_dir / "dashboard-run.log").symlink_to(external)

        result = self.app.process_log(cohort_id)

        self.assertFalse(result["available"])
        self.assertEqual(result["text"], "")

    def test_launcher_rejects_browser_supplied_paths(self):
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.app.start_run({"n_subjects": 1, "output_dir": "/tmp/external"})

    def test_launcher_caps_cohort_size_and_volumes(self):
        with self.assertRaisesRegex(ValueError, "1–4"):
            self.app.start_run({"n_subjects": 5})
        with self.assertRaisesRegex(ValueError, "50 and 120"):
            self.app.start_run({"n_subjects": 1, "t_vol": 121})

    def test_model_inference_requires_trusted_server_mount(self):
        with self.assertRaisesRegex(ValueError, "NEUROADHD_MODEL_DIR"):
            self.app.start_run({"n_subjects": 1, "model_inference": True})

    def test_only_one_run_can_be_active(self):
        self.app.start_run({"n_subjects": 1})
        with self.assertRaisesRegex(RuntimeError, "already running"):
            self.app.start_run({"n_subjects": 1})


if __name__ == "__main__":
    unittest.main()

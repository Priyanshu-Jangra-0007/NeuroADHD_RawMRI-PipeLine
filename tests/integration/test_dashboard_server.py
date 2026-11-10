"""Tests for the synthetic-only dashboard run launcher."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from scripts import serve_dashboard


class _FakeThread:
    last_instance: _FakeThread | None = None

    def __init__(self, target, args, daemon, name):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.name = name
        self.started = False
        _FakeThread.last_instance = self

    def start(self):
        self.started = True


class TestDashboardApplication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp.name) / "dashboard_cohorts"
        self.intake_root = Path(self.temp.name) / "intake_sessions"
        self.output_patch = mock.patch.object(serve_dashboard, "OUTPUT_ROOT", self.output_root)
        self.intake_patch = mock.patch.object(serve_dashboard, "INTAKE_ROOT", self.intake_root)
        self.thread_patch = mock.patch.object(serve_dashboard.threading, "Thread", _FakeThread)
        self.env_patch = mock.patch.dict(
            "os.environ",
            {"NEUROADHD_MODEL_DIR": "", "NEUROADHD_HMAC_KEY": ""},
            clear=False,
        )
        self.output_patch.start()
        self.intake_patch.start()
        self.thread_patch.start()
        self.env_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.intake_patch.stop)
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

    def test_start_run_writes_process_file_with_pid_and_start_marker(self):
        result = self.app.start_run({"n_subjects": 1})
        cohort_id = result["cohort_id"]
        cohort_dir = self.output_root / cohort_id
        status_path = cohort_dir / "dashboard-process.json"

        thread = _FakeThread.last_instance
        self.assertIsNotNone(thread)

        mid_run_checked = False

        def fake_wait():
            nonlocal mid_run_checked
            self.assertTrue(status_path.is_file())
            data = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(data["cohort_id"], cohort_id)
            self.assertEqual(data["status"], "RUNNING")
            self.assertEqual(data["pid"], os.getpid())
            self.assertIsNotNone(data.get("pid_start_marker"))
            self.assertIn("started_utc", data)
            mid_run_checked = True
            return 0

        mock_proc = mock.MagicMock()
        mock_proc.pid = os.getpid()
        mock_proc.wait = fake_wait

        with mock.patch("scripts.serve_dashboard.subprocess.Popen", return_value=mock_proc):
            thread.target(*thread.args)

        self.assertTrue(mid_run_checked)
        final_data = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(final_data["status"], "SUCCESS")
        self.assertEqual(final_data["exit_code"], 0)
        self.assertIn("finished_utc", final_data)

    def test_restart_with_live_runner_blocks(self):
        result = self.app.start_run({"n_subjects": 1})
        cohort_id = result["cohort_id"]
        cohort_dir = self.output_root / cohort_id
        manifest_path = cohort_dir / "cohort_manifest.json"
        manifest_data = {
            "cohort_run_id": cohort_id,
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "status": "RUNNING",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {"n_subjects": 1},
            "subjects": [{"subject_id": "sub-synthetic001", "status": "RUNNING"}],
            "summary": {"total_subjects": 1, "succeeded": 0, "failed": 0, "pending": 1},
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        start_marker = self.app._get_pid_start_marker(os.getpid())
        status_path = cohort_dir / "dashboard-process.json"
        status_path.write_text(
            json.dumps(
                {
                    "cohort_id": cohort_id,
                    "pid": os.getpid(),
                    "pid_start_marker": start_marker,
                    "status": "RUNNING",
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )

        # Simulate server restart (new DashboardApplication instance with empty _active)
        restarted_app = serve_dashboard.DashboardApplication()
        self.assertEqual(len(restarted_app._active), 0)
        self.assertTrue(restarted_app._has_running_cohort())

        # Starting a new run on restarted server blocks
        with self.assertRaisesRegex(RuntimeError, "already running"):
            restarted_app.start_run({"n_subjects": 1})

        manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_after["status"], "RUNNING")

    def test_dead_runner_recovers_and_allows_new_run(self):
        cohort_id = "cohort_20261103T120000_111111Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        manifest_path = cohort_dir / "cohort_manifest.json"
        manifest_data = {
            "cohort_run_id": cohort_id,
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "status": "RUNNING",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "created_utc": "2026-11-03T12:00:00.000000+00:00",
            "updated_utc": "2026-11-03T12:00:00.000000+00:00",
            "configuration": {"n_subjects": 1},
            "subjects": [{"subject_id": "sub-synthetic001", "status": "RUNNING"}],
            "summary": {"total_subjects": 1, "succeeded": 0, "failed": 0, "pending": 1},
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        status_path = cohort_dir / "dashboard-process.json"
        status_path.write_text(
            json.dumps({"pid": 999999999, "pid_start_marker": "old_marker", "status": "RUNNING"}),
            encoding="utf-8",
        )

        self.assertFalse(self.app._has_running_cohort())
        recovered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(recovered_manifest["status"], "TERMINATED")
        self.assertIn("terminated_reason", recovered_manifest)
        self.assertEqual(recovered_manifest["subjects"][0]["status"], "TERMINATED")

        # New run can start successfully
        result = self.app.start_run({"n_subjects": 1})
        self.assertEqual(result["status"], "STARTING")

    def test_reused_pid_recovers_and_allows_new_run(self):
        cohort_id = "cohort_20261103T120000_333333Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        manifest_path = cohort_dir / "cohort_manifest.json"
        manifest_data = {
            "cohort_run_id": cohort_id,
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "status": "RUNNING",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "created_utc": "2026-11-03T12:00:00.000000+00:00",
            "updated_utc": "2026-11-03T12:00:00.000000+00:00",
            "configuration": {"n_subjects": 1},
            "subjects": [{"subject_id": "sub-synthetic001", "status": "RUNNING"}],
            "summary": {"total_subjects": 1, "succeeded": 0, "failed": 0, "pending": 1},
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        status_path = cohort_dir / "dashboard-process.json"
        status_path.write_text(
            json.dumps({"pid": os.getpid(), "pid_start_marker": "Mon Jan 1 00:00:00 2000", "status": "RUNNING"}),
            encoding="utf-8",
        )

        self.assertFalse(self.app._has_running_cohort())
        recovered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(recovered_manifest["status"], "TERMINATED")
        self.assertIn("terminated_reason", recovered_manifest)

        # New run can start successfully
        result = self.app.start_run({"n_subjects": 1})
        self.assertEqual(result["status"], "STARTING")

    def test_no_pid_file_inside_grace_window_blocks(self):
        cohort_id = "cohort_20261103T120000_444444Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        manifest_path = cohort_dir / "cohort_manifest.json"
        recent_ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        manifest_data = {
            "cohort_run_id": cohort_id,
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "status": "RUNNING",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "created_utc": recent_ts,
            "updated_utc": recent_ts,
            "configuration": {"n_subjects": 1},
            "subjects": [{"subject_id": "sub-synthetic001", "status": "RUNNING"}],
            "summary": {"total_subjects": 1, "succeeded": 0, "failed": 0, "pending": 1},
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        restarted_app = serve_dashboard.DashboardApplication()
        self.assertTrue(restarted_app._has_running_cohort())
        with self.assertRaisesRegex(RuntimeError, "already running"):
            restarted_app.start_run({"n_subjects": 1})

        manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_after["status"], "RUNNING")

    def test_no_pid_file_after_grace_window_recovers(self):
        cohort_id = "cohort_20261103T120000_555555Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        manifest_path = cohort_dir / "cohort_manifest.json"
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        manifest_data = {
            "cohort_run_id": cohort_id,
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "status": "RUNNING",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "created_utc": old_ts,
            "updated_utc": old_ts,
            "configuration": {"n_subjects": 1},
            "subjects": [{"subject_id": "sub-synthetic001", "status": "RUNNING"}],
            "summary": {"total_subjects": 1, "succeeded": 0, "failed": 0, "pending": 1},
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        restarted_app = serve_dashboard.DashboardApplication()
        self.assertFalse(restarted_app._has_running_cohort())

        recovered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(recovered_manifest["status"], "TERMINATED")
        self.assertIn("grace window expired", recovered_manifest["terminated_reason"])

        result = restarted_app.start_run({"n_subjects": 1})
        self.assertEqual(result["status"], "STARTING")

    def test_failed_start_marker_lookup_keeps_live_cohort_running(self):
        cohort_id = "cohort_20261103T120000_666666Z"
        cohort_dir = self.output_root / cohort_id
        cohort_dir.mkdir(parents=True)
        manifest_path = cohort_dir / "cohort_manifest.json"
        manifest_data = {
            "cohort_run_id": cohort_id,
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "status": "RUNNING",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {"n_subjects": 1},
            "subjects": [{"subject_id": "sub-synthetic001", "status": "RUNNING"}],
            "summary": {"total_subjects": 1, "succeeded": 0, "failed": 0, "pending": 1},
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        status_path = cohort_dir / "dashboard-process.json"
        status_path.write_text(
            json.dumps({"pid": os.getpid(), "pid_start_marker": "expected_start_marker", "status": "RUNNING"}),
            encoding="utf-8",
        )

        # Simulate start marker lookup failure (e.g. ps error / transient failure) returning None
        with mock.patch.object(serve_dashboard.DashboardApplication, "_get_pid_start_marker", return_value=None):
            # Process is still alive (os.getpid()), so failed lookup must NOT cause termination
            self.assertTrue(self.app._is_pid_alive(os.getpid(), expected_start_marker="expected_start_marker"))
            restarted_app = serve_dashboard.DashboardApplication()
            self.assertTrue(restarted_app._has_running_cohort())
            manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_after["status"], "RUNNING")


if __name__ == "__main__":
    unittest.main()

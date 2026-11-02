"""Unit tests for src.validation.memory_guard.

Validates OS-level memory cgroup detection, process RSS monitoring,
and container isolation enforcement.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.validation.memory_guard import (
    DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
    get_cgroup_memory_stats,
    get_process_rss_mb,
    verify_os_memory_isolation,
)


class TestMemoryGuard(unittest.TestCase):
    def test_get_process_rss_mb(self):
        rss = get_process_rss_mb()
        self.assertIsInstance(rss, float)
        self.assertGreater(rss, 0.0)

    def test_get_cgroup_memory_stats_host(self):
        # On macOS host without cgroup, should return None for version and memory_max_bytes
        stats = get_cgroup_memory_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("cgroup_version", stats)
        self.assertIn("memory_max_bytes", stats)

    def test_get_cgroup_memory_stats_cgroup_v2_constrained(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_max = tmp_p / "memory.max"
            f_cur = tmp_p / "memory.current"
            f_peak = tmp_p / "memory.peak"

            f_max.write_text("2684354560\n", encoding="utf-8")
            f_cur.write_text("524288000\n", encoding="utf-8")
            f_peak.write_text("1048576000\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", f_max), \
                 patch("src.validation.memory_guard.CGROUP_V2_CURRENT", f_cur), \
                 patch("src.validation.memory_guard.CGROUP_V2_PEAK", f_peak):
                stats = get_cgroup_memory_stats()
                self.assertEqual(stats["cgroup_version"], 2)
                self.assertEqual(stats["memory_max_bytes"], 2684354560)
                self.assertEqual(stats["memory_max_mib"], 2560.0)
                self.assertEqual(stats["memory_current_bytes"], 524288000)
                self.assertEqual(stats["memory_peak_bytes"], 1048576000)

    def test_get_cgroup_memory_stats_cgroup_v2_unlimited(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_max = tmp_p / "memory.max"
            f_cur = tmp_p / "memory.current"
            f_peak = tmp_p / "memory.peak"

            f_max.write_text("max\n", encoding="utf-8")
            f_cur.write_text("104857600\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", f_max), \
                 patch("src.validation.memory_guard.CGROUP_V2_CURRENT", f_cur), \
                 patch("src.validation.memory_guard.CGROUP_V2_PEAK", f_peak):
                stats = get_cgroup_memory_stats()
                self.assertEqual(stats["cgroup_version"], 2)
                self.assertIsNone(stats["memory_max_bytes"])

    def test_get_cgroup_memory_stats_cgroup_v1_constrained(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_limit = tmp_p / "memory.limit_in_bytes"
            f_usage = tmp_p / "memory.usage_in_bytes"
            f_max_usage = tmp_p / "memory.max_usage_in_bytes"

            f_limit.write_text("2684354560\n", encoding="utf-8")
            f_usage.write_text("400000000\n", encoding="utf-8")
            f_max_usage.write_text("800000000\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", Path("/nonexistent/v2/max")), \
                 patch("src.validation.memory_guard.CGROUP_V1_LIMIT", f_limit), \
                 patch("src.validation.memory_guard.CGROUP_V1_USAGE", f_usage), \
                 patch("src.validation.memory_guard.CGROUP_V1_MAX_USAGE", f_max_usage):
                stats = get_cgroup_memory_stats()
                self.assertEqual(stats["cgroup_version"], 1)
                self.assertEqual(stats["memory_max_bytes"], 2684354560)
                self.assertEqual(stats["memory_max_mib"], 2560.0)

    def test_verify_os_memory_isolation_unconstrained_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_max = tmp_p / "memory.max"
            f_max.write_text("max\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", f_max):
                with self.assertRaises(RuntimeError) as ctx:
                    verify_os_memory_isolation(
                        max_allowed_bytes=DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
                        require_enforcement=True,
                    )
                self.assertIn("MEMORY ISOLATION FAILURE", str(ctx.exception))

    def test_verify_os_memory_isolation_unconstrained_permitted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_max = tmp_p / "memory.max"
            f_max.write_text("max\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", f_max):
                res = verify_os_memory_isolation(
                    max_allowed_bytes=DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
                    require_enforcement=False,
                )
                self.assertFalse(res["enforced"])
                self.assertEqual(res["status"], "UNCONSTRAINED_HOST")

    def test_verify_os_memory_isolation_exceeds_max_allowed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_max = tmp_p / "memory.max"
            # 4 GiB exceeds 2.5 GiB
            f_max.write_text("4294967296\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", f_max):
                with self.assertRaises(RuntimeError) as ctx:
                    verify_os_memory_isolation(
                        max_allowed_bytes=DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
                        require_enforcement=True,
                    )
                self.assertIn("exceeds the maximum permitted ceiling", str(ctx.exception))

    def test_verify_os_memory_isolation_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            f_max = tmp_p / "memory.max"
            # Exactly 2560 MiB
            f_max.write_text("2684354560\n", encoding="utf-8")

            with patch("src.validation.memory_guard.CGROUP_V2_MAX", f_max):
                res = verify_os_memory_isolation(
                    max_allowed_bytes=DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
                    require_enforcement=True,
                )
                self.assertTrue(res["enforced"])
                self.assertEqual(res["status"], "ENFORCED_OK")
                self.assertEqual(res["memory_max_mib"], 2560.0)


if __name__ == "__main__":
    unittest.main()

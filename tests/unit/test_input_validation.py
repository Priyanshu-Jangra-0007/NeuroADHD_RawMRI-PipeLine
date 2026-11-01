"""Unit tests for BIDS input validation logic."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.validation.bids_validator import BIDSInputValidator


class TestBIDSInputValidation(unittest.TestCase):
    def setUp(self):
        self.validator = BIDSInputValidator()

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            self.validator.validate_sidecar_json(Path("/non/existent/path.json"))

    def test_corrupted_json_raises_valueerror(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{invalid_json: true")
            f_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                self.validator.validate_sidecar_json(f_path)
            self.assertIn("Corrupted JSON", str(ctx.exception))
        finally:
            if f_path.is_file():
                f_path.unlink()

    def test_missing_repetition_time_raises_keyerror(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"TaskName": "rest"}, f)
            f_path = Path(f.name)

        try:
            with self.assertRaises(KeyError) as ctx:
                self.validator.validate_sidecar_json(f_path)
            self.assertIn("RepetitionTime", str(ctx.exception))
        finally:
            if f_path.is_file():
                f_path.unlink()

    def test_out_of_range_repetition_time_raises_valueerror(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"TaskName": "rest", "RepetitionTime": 15.0}, f)
            f_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                self.validator.validate_sidecar_json(f_path)
            self.assertIn("outside valid physiological range", str(ctx.exception))
        finally:
            if f_path.is_file():
                f_path.unlink()

    def test_non_resting_task_raises_valueerror(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"TaskName": "nback_working_memory", "RepetitionTime": 2.0}, f)
            f_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                self.validator.validate_sidecar_json(f_path)
            self.assertIn("non-resting state", str(ctx.exception))
        finally:
            if f_path.is_file():
                f_path.unlink()

    def test_valid_resting_sidecar_passes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"TaskName": "rest", "RepetitionTime": 2.0}, f)
            f_path = Path(f.name)

        try:
            res = self.validator.validate_sidecar_json(f_path)
            self.assertEqual(res["status"], "VALIDATED")
            self.assertEqual(res["repetition_time"], 2.0)
            self.assertEqual(res["task_name"], "rest")
        finally:
            if f_path.is_file():
                f_path.unlink()


if __name__ == "__main__":
    unittest.main()

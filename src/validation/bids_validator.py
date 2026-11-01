"""BIDS and input format validation module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BIDSInputValidator:
    """Validates raw resting-state fMRI BIDS structure and metadata against project rules."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "input_validation_config.json"
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        if self.config_path.is_file():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

    def validate_sidecar_json(self, sidecar_path: Path) -> Dict[str, Any]:
        """Validate BOLD JSON sidecar metadata."""
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing BOLD sidecar JSON: {sidecar_path}")

        with open(sidecar_path, "r", encoding="utf-8") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupted JSON sidecar: {exc}")

        # Check RepetitionTime
        tr = metadata.get("RepetitionTime")
        if tr is None:
            raise KeyError("BOLD JSON sidecar missing required 'RepetitionTime' key")
        try:
            tr_val = float(tr)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid RepetitionTime value: {tr}")

        if not (0.5 <= tr_val <= 5.0):
            raise ValueError(f"RepetitionTime {tr_val}s is outside valid physiological range [0.5, 5.0]s")

        # Check TaskName if present
        task = metadata.get("TaskName", "")
        if task and "rest" not in task.lower():
            raise ValueError(f"TaskName '{task}' indicates non-resting state fMRI; expected 'rest'")

        return {
            "status": "VALIDATED",
            "repetition_time": tr_val,
            "task_name": task or "rest",
        }

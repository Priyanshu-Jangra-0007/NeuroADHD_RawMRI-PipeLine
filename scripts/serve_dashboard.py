#!/usr/bin/env python3
"""Serve the local NeuroADHD synthetic research dashboard using Python's stdlib.

The server binds to loopback by default, never accepts input paths from the browser,
and can launch only the repository's synthetic cohort runner.
"""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dashboard_cohorts"
RUNNER = PROJECT_ROOT / "scripts" / "run_synthetic_cohort.py"
MAX_REQUEST_BYTES = 4096


class DashboardApplication:
    def __init__(self) -> None:
        from src.dashboard.service import DashboardDataService

        self.service = DashboardDataService(OUTPUT_ROOT)
        self._lock = threading.Lock()
        self._active: Dict[str, Dict[str, Any]] = {}
        self._ephemeral_integrity_key = os.environ.get("NEUROADHD_HMAC_KEY") or os.environ.get("NEUROADHD_INTEGRITY_KEY")
        if not self._ephemeral_integrity_key:
            self._ephemeral_integrity_key = secrets.token_urlsafe(32)

    def health(self) -> Dict[str, Any]:
        pending = None
        try:
            from src.validation.authorization_gate import ResearchAuthorizationGate

            report = ResearchAuthorizationGate(base_dir=PROJECT_ROOT).evaluate_register()
            pending = report.get("status_counts", {}).get("PENDING")
        except Exception:
            # A dashboard is read-only with respect to authorization. Any register
            # validation failure is represented as NOT_AUTHORIZED, never as permission.
            pending = None
        return {
            "application": "NeuroADHD Research Dashboard",
            "status": "READY",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "authorization_state": "NOT_AUTHORIZED",
            "pending_authorization_items": pending,
            "pipeline": "Existing 19-stage synthetic MRI pipeline",
            "output_root": "outputs/dashboard_cohorts",
            "model_dir_available": bool(os.environ.get("NEUROADHD_MODEL_DIR")),
        }

    def _has_running_cohort(self) -> bool:
        for cohort in self.service.list_cohorts():
            if cohort.get("status") == "RUNNING":
                return True
        return False

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"n_subjects", "t_vol", "seed", "atlas_mode", "model_inference"}
        if set(payload) - allowed:
            raise ValueError("Request contains unsupported fields.")
        try:
            n_subjects = int(payload.get("n_subjects", 2))
            t_vol = int(payload.get("t_vol", 60))
            seed = int(payload.get("seed", 42))
        except (TypeError, ValueError) as exc:
            raise ValueError("Subject count, volume count and seed must be integers.") from exc
        if not 1 <= n_subjects <= 4:
            raise ValueError("Dashboard runs support 1–4 synthetic subjects at a time.")
        if not 50 <= t_vol <= 120:
            raise ValueError("Volume count must be between 50 and 120.")
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("Seed must be between 0 and 2,147,483,647.")
        atlas_mode = payload.get("atlas_mode", "synthetic_phantom")
        if atlas_mode not in ("synthetic_phantom", "canonical"):
            raise ValueError("Unsupported atlas mode.")
        model_inference = payload.get("model_inference", False)
        if not isinstance(model_inference, bool):
            raise ValueError("model_inference must be a boolean.")
        model_dir = os.environ.get("NEUROADHD_MODEL_DIR")
        if model_inference and not model_dir:
            raise ValueError(
                "Frozen model inference was requested, but NEUROADHD_MODEL_DIR is not configured on the server."
            )

        with self._lock:
            if self._active or self._has_running_cohort():
                raise RuntimeError("A synthetic cohort is already running. Wait for it to finish before starting another.")

            cohort_id = datetime.now(timezone.utc).strftime("cohort_%Y%m%dT%H%M%S_%fZ")
            cohort_dir = OUTPUT_ROOT / cohort_id
            cohort_dir.mkdir(parents=True, exist_ok=False)
            log_path = cohort_dir / "dashboard-run.log"
            status_path = cohort_dir / "dashboard-process.json"
            command = [
                sys.executable,
                str(RUNNER),
                "--output-dir", str(cohort_dir),
                "--n-subjects", str(n_subjects),
                "--t-vol", str(t_vol),
                "--seed", str(seed),
                "--atlas-mode", atlas_mode,
                "--max-memory-gb", "2.5",
            ]
            if atlas_mode == "canonical":
                command.extend(["--canonical-atlas-path", str(PROJECT_ROOT / "data" / "atlases" / "cc200_roi_atlas.nii.gz")])
            if model_inference:
                command.extend(["--model-dir", model_dir, "--require-model-inference"])

            state = {
                "cohort_id": cohort_id,
                "status": "STARTING",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": None,
                "pid": None,
            }
            self._active[cohort_id] = state
            thread = threading.Thread(
                target=self._run_worker,
                args=(cohort_id, cohort_dir, log_path, status_path, command),
                daemon=True,
                name=f"neuroadhd-{cohort_id}",
            )
            thread.start()
        return {
            "cohort_id": cohort_id,
            "status": "STARTING",
            "message": "Synthetic cohort queued. Pipeline execution is running in the background.",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
        }

    def _run_worker(self, cohort_id: str, cohort_dir: Path, log_path: Path, status_path: Path, command: list[str]) -> None:
        state = self._active[cohort_id]
        child_env = os.environ.copy()
        child_env["NEUROADHD_HMAC_KEY"] = self._ephemeral_integrity_key
        child_env["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
                state["pid"] = process.pid
                state["status"] = "RUNNING"
                exit_code = process.wait()
            state["exit_code"] = int(exit_code)
            state["status"] = "SUCCESS" if exit_code == 0 else "FAILED"
        except Exception as exc:
            state["exit_code"] = -1
            state["status"] = "FAILED"
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            temp_path = status_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(temp_path, status_path)
            with self._lock:
                self._active.pop(cohort_id, None)

    def process_log(self, cohort_id: str) -> Dict[str, Any]:
        cohort_dir = self.service._cohort_dir(cohort_id)
        log_path = cohort_dir / "dashboard-run.log"
        if log_path.is_symlink() or not log_path.is_file():
            return {"available": False, "text": ""}
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16000), os.SEEK_SET)
            text = stream.read()
        return {"available": True, "text": text[-16000:]}

    def process_status(self, cohort_id: str) -> Dict[str, Any]:
        from src.dashboard.service import DashboardDataError

        cohort_dir = self.service._cohort_dir(cohort_id)
        status_file = cohort_dir / "dashboard-process.json"
        if status_file.is_file() and not status_file.is_symlink():
            try:
                with status_file.open("r", encoding="utf-8") as stream:
                    return json.load(stream)
            except (OSError, json.JSONDecodeError):
                pass
        state = self._active.get(cohort_id)
        if state:
            return {key: value for key, value in state.items() if key != "pid"}
        try:
            cohort = self.service.get_cohort(cohort_id)
            return {"cohort_id": cohort_id, "status": cohort["status"], "exit_code": None}
        except DashboardDataError:
            raise


APP = DashboardApplication()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "NeuroADHDLocal/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid logging query strings or filesystem paths; requests are local.
        sys.stderr.write(f"[dashboard] {self.address_string()} {self.command}\n")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        from src.dashboard.service import DashboardDataError

        parsed = urlsplit(self.path)
        route = parsed.path
        try:
            if route == "/api/health":
                return self._send_json(APP.health())
            if route == "/api/runs":
                return self._send_json({"runs": APP.service.list_cohorts()})

            parts = [unquote(part) for part in route.strip("/").split("/") if part]
            if len(parts) >= 3 and parts[0:2] == ["api", "runs"]:
                cohort_id = parts[2]
                if len(parts) == 3:
                    return self._send_json(APP.service.get_cohort(cohort_id))
                if len(parts) >= 5 and parts[3] == "subjects":
                    subject_id = parts[4]
                    if len(parts) == 5:
                        return self._send_json(APP.service.get_subject_detail(cohort_id, subject_id))
                    if len(parts) == 6 and parts[5] == "connectivity":
                        return self._send_json(APP.service.get_connectivity(cohort_id, subject_id))
                    if len(parts) == 6 and parts[5] == "timeseries":
                        query = parse_qs(parsed.query)
                        roi = int(query.get("roi", ["1"])[0])
                        return self._send_json(APP.service.get_timeseries(cohort_id, subject_id, roi))
                    if len(parts) == 7 and parts[5] == "artifact":
                        data, content_type = APP.service.get_artifact(cohort_id, subject_id, parts[6])
                        return self._send_bytes(data, content_type)
                if len(parts) == 4 and parts[3] == "process":
                    return self._send_json(APP.process_status(cohort_id))
                if len(parts) == 4 and parts[3] == "log":
                    return self._send_json(APP.process_log(cohort_id))
            if route.startswith("/api/"):
                return self._error("API endpoint not found.", 404)
            self._serve_static(route)
        except DashboardDataError as exc:
            self._error(str(exc), 404)
        except (ValueError, TypeError) as exc:
            self._error(str(exc), 400)
        except Exception as exc:
            self._error(f"Dashboard request failed: {type(exc).__name__}.", 500)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/runs":
            return self._error("API endpoint not found.", 404)
        try:
            # The dashboard is local-only; reject cross-origin browser submissions
            # to prevent another website from starting runs in the user's browser.
            origin = self.headers.get("Origin")
            if origin:
                parsed_origin = urlsplit(origin)
                origin_port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
                if (
                    parsed_origin.scheme != "http"
                    or parsed_origin.hostname not in ("127.0.0.1", "localhost")
                    or origin_port != self.server.server_port
                ):
                    return self._error("Cross-origin run requests are not allowed.", 403)
            host_header = self.headers.get("Host", "")
            parsed_host = urlsplit(f"//{host_header}")
            if parsed_host.hostname not in ("127.0.0.1", "localhost"):
                return self._error("Dashboard Host header is not allowed.", 403)
            length = int(self.headers.get("Content-Length", "0")
            if length <= 0 or length > MAX_REQUEST_BYTES:
                return self._error("Request body is missing or too large.", 413)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                return self._error("Request body must be a JSON object.")
            result = APP.start_run(payload)
            self._send_json(result, status=202)
        except json.JSONDecodeError:
            self._error("Request body must be valid JSON.")
        except ValueError as exc:
            self._error(str(exc), 400)
        except RuntimeError as exc:
            self._error(str(exc), 409)
        except Exception as exc:
            self._error(f"Could not start synthetic run: {type(exc).__name__}.", 500)

    def _serve_static(self, route: str) -> None:
        file_map = {
            "/": FRONTEND_ROOT / "index.html",
            "/index.html": FRONTEND_ROOT / "index.html",
            "/app.js": FRONTEND_ROOT / "app.js",
            "/styles.css": FRONTEND_ROOT / "styles.css",
        }
        path = file_map.get(route)
        if path is None or not path.is_file():
            return self._error("Page not found.", 404)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), f"{content_type}; charset=utf-8")


def main() -> int:
    host = os.environ.get("NEUROADHD_DASHBOARD_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Safety restriction: dashboard binds to loopback only.", file=sys.stderr)
        return 2
    port = int(os.environ.get("NEUROADHD_DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"NeuroADHD dashboard: http://127.0.0.1:{port}")
    print("Mode: STRICTLY_SYNTHETIC_ONLY | Real-data authorization: FALSE")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

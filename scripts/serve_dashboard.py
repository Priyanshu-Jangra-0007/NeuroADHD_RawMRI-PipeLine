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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intake.service import IntakeError, IntakeService, get_data_root

_REAL_POPEN = subprocess.Popen

FRONTEND_ROOT = PROJECT_ROOT / "frontend"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dashboard_cohorts"
INTAKE_ROOT = get_data_root()
RUNNER = PROJECT_ROOT / "scripts" / "run_synthetic_cohort.py"
MAX_REQUEST_BYTES = 4096


class DashboardApplication:
    def __init__(self) -> None:
        from src.dashboard.service import DashboardDataService

        self.service = DashboardDataService(OUTPUT_ROOT)
        self.intake = IntakeService(INTAKE_ROOT)
        self._lock = threading.RLock()
        self._active: Dict[str, Dict[str, Any]] = {}
        self._ephemeral_integrity_key = os.environ.get("NEUROADHD_HMAC_KEY") or os.environ.get("NEUROADHD_INTEGRITY_KEY")
        if not self._ephemeral_integrity_key:
            self._ephemeral_integrity_key = secrets.token_urlsafe(32)
        self._recover_stale_cohorts()

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

    def _get_pid_start_marker(self, pid: int) -> Optional[str]:
        if not isinstance(pid, int) or pid <= 0:
            return None
        # Linux /proc/<pid>/stat field 22 (starttime)
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            try:
                content = proc_stat.read_text(encoding="utf-8")
                idx = content.rfind(")")
                if idx != -1:
                    fields = content[idx + 1 :].split()
                    if len(fields) >= 20:
                        return fields[19]
            except Exception:
                pass
        # macOS / fallback: ps -o lstart= -p <pid>
        try:
            proc = _REAL_POPEN(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, _ = proc.communicate(timeout=5)
            if proc.returncode == 0:
                stripped = out.strip()
                if stripped:
                    return stripped
        except Exception:
            pass
        return None

    def _is_pid_alive(self, pid: int, expected_start_marker: Optional[str] = None) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # PermissionError indicates the process exists (owned by another user or sandboxed)
            pass
        except OSError:
            return False

        if expected_start_marker is not None:
            current_marker = self._get_pid_start_marker(pid)
            if current_marker is not None and current_marker != expected_start_marker:
                return False
        return True

    def _terminate_stale_cohort(self, cohort_dir: Path, manifest: Dict[str, Any], reason: str) -> None:
        manifest_path = cohort_dir / "cohort_manifest.json"
        now_iso = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "TERMINATED"
        manifest["terminated_reason"] = reason
        manifest["terminated_utc"] = now_iso
        manifest["updated_utc"] = now_iso

        subjects = manifest.get("subjects", [])
        for subj in subjects:
            if subj.get("status") in ("RUNNING", "PENDING", None):
                subj["status"] = "TERMINATED"
                subj["error"] = reason

        succeeded = sum(1 for s in subjects if s.get("status") == "SUCCESS")
        failed = sum(1 for s in subjects if s.get("status") in ("FAILED", "TERMINATED"))
        manifest["summary"] = {
            "total_subjects": len(subjects),
            "succeeded": succeeded,
            "failed": failed,
            "pending": 0,
            "failures": [
                {"subject_id": s.get("subject_id"), "error": s.get("error", reason)}
                for s in subjects
                if s.get("status") in ("FAILED", "TERMINATED")
            ],
        }

        temp_path = manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_path, manifest_path)

        status_file = cohort_dir / "dashboard-process.json"
        proc_data: Dict[str, Any] = {}
        if status_file.is_file() and not status_file.is_symlink():
            try:
                proc_data = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:
                proc_data = {}
        proc_data["status"] = "TERMINATED"
        proc_data["exit_code"] = proc_data.get("exit_code", -1)
        proc_data["terminated_reason"] = reason
        proc_data["terminated_utc"] = now_iso
        temp_proc = status_file.with_suffix(".tmp")
        temp_proc.write_text(json.dumps(proc_data, indent=2), encoding="utf-8")
        os.replace(temp_proc, status_file)

    def _recover_stale_cohorts(self) -> None:
        if not OUTPUT_ROOT.is_dir():
            return
        for candidate in OUTPUT_ROOT.iterdir():
            if not candidate.is_dir() or candidate.is_symlink() or not candidate.name.startswith("cohort_"):
                continue
            manifest_path = candidate / "cohort_manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                continue

            if manifest.get("status") not in ("RUNNING", "STARTING"):
                continue

            cohort_id = candidate.name
            # (a) in self._active -> skip
            if cohort_id in self._active:
                continue

            status_file = candidate / "dashboard-process.json"

            # (b) pid file present -> alive: skip, dead: TERMINATE
            if status_file.is_file() and not status_file.is_symlink():
                proc_data: Dict[str, Any] = {}
                try:
                    with status_file.open("r", encoding="utf-8") as f:
                        proc_data = json.load(f)
                except Exception:
                    proc_data = {}
                pid = proc_data.get("pid")
                marker = proc_data.get("pid_start_marker")
                if self._is_pid_alive(pid, expected_start_marker=marker):
                    continue
                self._terminate_stale_cohort(
                    candidate,
                    manifest,
                    reason="Stale RUNNING cohort recovered: runner process is not alive.",
                )
                continue

            # (c) no pid file -> apply a REAL grace window using manifest updated_utc (30 s),
            # skip while inside it, terminate only after.
            ts_str = manifest.get("updated_utc") or manifest.get("created_utc")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - ts).total_seconds()
                    if age < 30.0:
                        continue
                except Exception:
                    pass

            self._terminate_stale_cohort(
                candidate,
                manifest,
                reason="Stale RUNNING cohort recovered: no process file and grace window expired.",
            )

    def _has_running_cohort(self) -> bool:
        self._recover_stale_cohorts()
        with self._lock:
            if self._active:
                return True
        for cohort in self.service.list_cohorts():
            if cohort.get("status") in ("RUNNING", "STARTING"):
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
                "pid_start_marker": None,
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
                start_marker = self._get_pid_start_marker(process.pid)
                state["pid"] = process.pid
                state["pid_start_marker"] = start_marker
                state["status"] = "RUNNING"

                # Atomically write dashboard-process.json right after Popen
                init_data = {
                    "cohort_id": cohort_id,
                    "pid": process.pid,
                    "pid_start_marker": start_marker,
                    "status": "RUNNING",
                    "started_utc": state["started_utc"],
                }
                temp_init = status_path.with_suffix(".tmp")
                temp_init.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
                os.replace(temp_init, status_path)

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
            if route == "/api/intake/sessions":
                return self._send_json({"sessions": APP.intake.list_sessions()})
            if route == "/api/runs":
                return self._send_json({"runs": APP.service.list_cohorts()})

            parts = [unquote(part) for part in route.strip("/").split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "intake"] and parts[2] == "sessions":
                return self._send_json(APP.intake.get_session(parts[3]))
            if len(parts) == 5 and parts[:3] == ["api", "intake", "sessions"] and parts[4] == "download_bids":
                data, content_type, filename = APP.intake.get_bids_session_zip(parts[3])
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if len(parts) == 6 and parts[:3] == ["api", "intake", "sessions"] and parts[4] == "artifacts":
                data, content_type = APP.intake.get_report_artifact(parts[3], parts[5])
                return self._send_bytes(data, content_type)
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
        except (DashboardDataError, IntakeError) as exc:
            self._error(str(exc), 404)
        except (ValueError, TypeError) as exc:
            self._error(str(exc), 400)
        except Exception as exc:
            self._error(f"Dashboard request failed: {type(exc).__name__}.", 500)

    def _local_request_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            origin_port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
            if (
                parsed_origin.scheme != "http"
                or parsed_origin.hostname not in ("127.0.0.1", "localhost")
                or origin_port != self.server.server_port
            ):
                self._error("Cross-origin requests are not allowed.", 403)
                return False
        host_header = self.headers.get("Host", "")
        parsed_host = urlsplit(f"//{host_header}")
        if parsed_host.hostname not in ("127.0.0.1", "localhost"):
            self._error("Dashboard Host header is not allowed.", 403)
            return False
        return True

    def _content_length(self, maximum: int, minimum: int = 1) -> Optional[int]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error("Content-Length must be an integer.", 400)
            return None
        if length < minimum or length > maximum:
            self._error(f"Request body must be between {minimum} bytes and {maximum} bytes.", 413)
            return None
        return length

    def do_POST(self) -> None:
        if not self._local_request_allowed():
            return
        parsed = urlsplit(self.path)
        route = parsed.path
        try:
            if route == "/api/intake/sessions":
                length = self._content_length(MAX_REQUEST_BYTES)
                if length is None:
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or set(payload) != {"kind"}:
                    return self._error("Request body must contain only kind: scan or report.")
                result = APP.intake.create_session(payload["kind"])
                return self._send_json(result, status=201)

            parts = [unquote(part) for part in route.strip("/").split("/") if part]
            if len(parts) == 5 and parts[:2] == ["api", "intake"] and parts[2] == "sessions":
                session_id, action = parts[3], parts[4]
                if action == "files":
                    # File bytes are streamed to disk; they are not buffered in process memory.
                    length = self._content_length(2 * 1024 * 1024 * 1024)
                    if length is None:
                        return
                    filename = unquote(self.headers.get("X-Upload-Filename", ""))
                    result = APP.intake.upload_file(session_id, filename, self.rfile, length)
                    return self._send_json(result, status=201)
                if action == "validate":
                    result = APP.intake.validate_scan_session(session_id)
                    return self._send_json(result)
                if action == "inspect_dicom":
                    result = APP.intake.inspect_dicom_series(session_id)
                    return self._send_json(result)
                if action == "convert_dicom":
                    length = self._content_length(MAX_REQUEST_BYTES)
                    if length is None:
                        return
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = APP.intake.convert_dicom_to_bids(session_id, payload)
                    return self._send_json(result)
                if action == "process-reports":
                    result = APP.intake.process_report_session(session_id)
                    return self._send_json(result)
            if route != "/api/runs":
                return self._error("API endpoint not found.", 404)

            length = self._content_length(MAX_REQUEST_BYTES)
            if length is None:
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                return self._error("Request body must be a JSON object.")
            result = APP.start_run(payload)
            self._send_json(result, status=202)
        except IntakeError as exc:
            self._error(str(exc), 400)
        except json.JSONDecodeError:
            self._error("Request body must be valid JSON.", 400)
        except ValueError as exc:
            self._error(str(exc), 400)
        except RuntimeError as exc:
            self._error(str(exc), 409)
        except Exception as exc:
            self._error(f"Dashboard request failed: {type(exc).__name__}.", 500)

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

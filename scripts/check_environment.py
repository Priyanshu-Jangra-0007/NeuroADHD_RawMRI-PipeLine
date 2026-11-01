#!/usr/bin/env python3
"""Environment and prerequisite audit script for NeuroADHD_RawMRI."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def main():
    print("=" * 60)
    print("NEUROADHD_RAWMRI SYSTEM ENVIRONMENT AUDIT")
    print("=" * 60)
    print(f"Python Version    : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform          : {platform.platform()}")
    print(f"Architecture      : {platform.machine()}")

    # Check Docker CLI
    docker_bin = shutil.which("docker")
    print(f"Docker Binary     : {docker_bin or 'NOT FOUND'}")
    if docker_bin:
        try:
            res = subprocess.run([docker_bin, "info"], capture_output=True, text=True, timeout=5)
            daemon_status = "RUNNING" if res.returncode == 0 else "STOPPED / INACCESSIBLE"
        except Exception:
            daemon_status = "TIMED OUT / ERROR"
        print(f"Docker Daemon     : {daemon_status}")

    # Check Disk Space
    usage = shutil.disk_usage(".")
    print(f"Disk Total        : {usage.total / (1024**3):.1f} GiB")
    print(f"Disk Available    : {usage.free / (1024**3):.1f} GiB")
    print("=" * 60)


if __name__ == "__main__":
    main()

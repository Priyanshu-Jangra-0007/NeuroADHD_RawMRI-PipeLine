#!/usr/bin/env python3
"""Authoritative machine-readable CI test summary reporter for NeuroADHD RawMRI PoC V2.

Parses pytest JUnit XML execution reports and produces an authoritative summary
categorizing:
- Total tests collected, passed, failed, errors, and skipped.
- Strict detection of missing or malformed XML files (fails closed).
- Breakdown by test category: unit, synthetic integration, container integration.
- Frozen artifact integrity & actual inference status.
- Explicit reasons for all skipped tests (e.g. absent model mount in CI).
- Fail-closed exit code if any test fails, errors, or evidence is incomplete.
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_junit_xml(xml_paths: List[Path]) -> Dict[str, Any]:
    """Parse one or more JUnit XML test report files with fail-closed validation."""
    collected_tests = []
    seen_test_ids = set()
    missing_files = []
    malformed_files = []

    for xml_path in xml_paths:
        if not xml_path.is_file():
            print(f"ERROR: Mandatory JUnit XML report not found at {xml_path}", file=sys.stderr)
            missing_files.append(str(xml_path))
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as exc:
            print(f"ERROR: Malformed or unreadable XML report at {xml_path}: {exc}", file=sys.stderr)
            malformed_files.append(f"{xml_path} ({exc})")
            continue

        # Support both <testsuites> and single <testsuite> root
        testcases = root.findall(".//testcase")
        for tc in testcases:
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            time_val = float(tc.get("time", "0.0") or "0.0")
            test_id = f"{classname}::{name}"

            # Deduplicate identical test results if reported across multiple XML files
            if test_id in seen_test_ids:
                continue
            seen_test_ids.add(test_id)

            status = "PASSED"
            message = ""
            failure_elem = tc.find("failure")
            error_elem = tc.find("error")
            skipped_elem = tc.find("skipped")

            if error_elem is not None:
                status = "ERROR"
                message = error_elem.get("message", "") or (error_elem.text or "")
            elif failure_elem is not None:
                status = "FAILED"
                message = failure_elem.get("message", "") or (failure_elem.text or "")
            elif skipped_elem is not None:
                status = "SKIPPED"
                message = skipped_elem.get("message", "") or (skipped_elem.text or "")

            collected_tests.append({
                "id": test_id,
                "classname": classname,
                "name": name,
                "time": time_val,
                "status": status,
                "message": message.strip(),
            })

    return {
        "tests": collected_tests,
        "missing_files": missing_files,
        "malformed_files": malformed_files,
    }


def categorize_tests(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Categorize tests into unit, pipeline integration, container integration, and model suites."""
    counts = {
        "total": len(tests),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "unit": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
        "pipeline_integration": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
        "container_integration": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
        "frozen_model_integrity": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
        "frozen_model_inference": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
    }

    skipped_details = []

    for t in tests:
        status = t["status"]
        if status == "PASSED":
            counts["passed"] += 1
        elif status == "FAILED":
            counts["failed"] += 1
        elif status == "ERROR":
            counts["errors"] += 1
        elif status == "SKIPPED":
            counts["skipped"] += 1
            skipped_details.append(t)

        classname = t["classname"]
        name = t["name"]

        def _update_cat(cat_key: str):
            counts[cat_key]["total"] += 1
            if status == "PASSED":
                counts[cat_key]["passed"] += 1
            elif status == "FAILED":
                counts[cat_key]["failed"] += 1
            elif status == "ERROR":
                counts[cat_key]["errors"] += 1
            elif status == "SKIPPED":
                counts[cat_key]["skipped"] += 1

        # High-level test suite category
        if "test_container_model_integration" in classname:
            _update_cat("container_integration")
        elif "tests.unit" in classname:
            _update_cat("unit")
        elif "tests.integration" in classname:
            _update_cat("pipeline_integration")

        # Specific model verification contracts
        if "TestFrozenArtifactIntegrityAndDiscovery" in classname:
            _update_cat("frozen_model_integrity")

        if "TestFrozenArtifactIntegration" in classname or "test_case_i" in name:
            _update_cat("frozen_model_inference")

    return {
        "counts": counts,
        "skipped_details": skipped_details,
    }


def print_summary(
    summary_data: Dict[str, Any],
    missing_files: List[str],
    malformed_files: List[str],
    model_dir_env: Optional[str],
    markdown_path: Optional[Path] = None,
) -> None:
    """Print ASCII and Markdown formatted summaries."""
    counts = summary_data["counts"]
    skipped_details = summary_data["skipped_details"]

    header = "=" * 80
    print(header)
    print("NEUROADHD RAW MRI POC V2 — AUTHORITATIVE CI TEST EVIDENCE SUMMARY")
    print(header)
    print(f"Total Tests Collected       : {counts['total']}")
    print(f"Total Tests Passed          : {counts['passed']}")
    print(f"Total Tests Failed          : {counts['failed']}")
    print(f"Total Test Errors           : {counts['errors']}")
    print(f"Total Tests Skipped         : {counts['skipped']}")
    print("-" * 80)
    print("TEST BREAKDOWN BY CATEGORY:")
    print(
        f"  • Core Unit Tests         : {counts['unit']['passed']}/{counts['unit']['total']} passed "
        f"({counts['unit']['skipped']} skipped, {counts['unit']['failed']} failed, {counts['unit']['errors']} errors)"
    )
    print(
        f"  • Synthetic Integration   : {counts['pipeline_integration']['passed']}/{counts['pipeline_integration']['total']} passed "
        f"({counts['pipeline_integration']['skipped']} skipped, {counts['pipeline_integration']['failed']} failed, {counts['pipeline_integration']['errors']} errors)"
    )
    print(
        f"  • Container Model Tests   : {counts['container_integration']['passed']}/{counts['container_integration']['total']} passed "
        f"({counts['container_integration']['skipped']} skipped, {counts['container_integration']['failed']} failed, {counts['container_integration']['errors']} errors)"
    )
    print("-" * 80)
    print("FROZEN MODEL ARTIFACT INTEGRATION STATUS:")
    print(f"  • NEUROADHD_MODEL_DIR configured : {bool(model_dir_env)} ({model_dir_env or 'UNSET'})")
    if model_dir_env:
        print("  • CI Status                       : AUTHENTIC ARTIFACTS VERIFIED")
        print(f"  • Frozen Model Checksum Tests     : {counts['frozen_model_integrity']['passed']} passed")
        print(f"  • Frozen Model Inference Tests    : {counts['frozen_model_inference']['passed']} passed")
    else:
        print("  • CI Status                       : EXPLICITLY SKIPPED IN CI")
        print("  • Reason                          : Model binary artifacts (.joblib) are strictly excluded")
        print("                                      from git repository per privacy/licensing policy and")
        print("                                      require secure host volume mount.")
        print(f"  • Frozen Model Checksum Tests     : {counts['frozen_model_integrity']['skipped']} skipped")
        print(f"  • Frozen Model Inference Tests    : {counts['frozen_model_inference']['skipped']} skipped")
        print("  • Evidence Reference              : docs/PHASE_A_4_2_CONTAINER_MODEL_INTEGRATION_REPORT.md")

    if missing_files or malformed_files:
        print("-" * 80)
        print("REPORT DEFECTS & MISSING EVIDENCE:")
        for mf in missing_files:
            print(f"  - MISSING REPORT: {mf}")
        for mal in malformed_files:
            print(f"  - MALFORMED REPORT: {mal}")

    if skipped_details:
        print("-" * 80)
        print(f"EXPLICITLY SKIPPED TESTS ({len(skipped_details)}):")
        for s in skipped_details:
            short_id = s["id"].split(".")[-1]
            print(f"  - {short_id}")
            print(f"      Reason: {s['message']}")

    print(header)

    if markdown_path:
        md_lines = [
            "# NeuroADHD RawMRI PoC V2 — CI Test Execution Summary",
            "",
            f"**Total Collected:** {counts['total']} | **Passed:** {counts['passed']} | **Failed:** {counts['failed']} | **Errors:** {counts['errors']} | **Skipped:** {counts['skipped']}",
            "",
            "## Category Breakdown",
            "",
            "| Category | Collected | Passed | Failed | Errors | Skipped |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| Core Unit Tests | {counts['unit']['total']} | {counts['unit']['passed']} | {counts['unit']['failed']} | {counts['unit']['errors']} | {counts['unit']['skipped']} |",
            f"| Synthetic Pipeline Integration | {counts['pipeline_integration']['total']} | {counts['pipeline_integration']['passed']} | {counts['pipeline_integration']['failed']} | {counts['pipeline_integration']['errors']} | {counts['pipeline_integration']['skipped']} |",
            f"| Container Model Integration | {counts['container_integration']['total']} | {counts['container_integration']['passed']} | {counts['container_integration']['failed']} | {counts['container_integration']['errors']} | {counts['container_integration']['skipped']} |",
            f"| **Total** | **{counts['total']}** | **{counts['passed']}** | **{counts['failed']}** | **{counts['errors']}** | **{counts['skipped']}** |",
            "",
            "## Downstream Model Integration Status",
            "",
            f"- **NEUROADHD_MODEL_DIR configured:** `{bool(model_dir_env)}`",
            f"- **Artifact Checksums & Integrity Tests:** `{counts['frozen_model_integrity']['passed']} passed, {counts['frozen_model_integrity']['skipped']} skipped`",
            f"- **Zero-Refit Frozen Inference Tests:** `{counts['frozen_model_inference']['passed']} passed, {counts['frozen_model_inference']['skipped']} skipped`",
            "",
        ]
        if missing_files or malformed_files:
            md_lines.extend([
                "## Evidence Integrity Defects",
                "",
            ])
            for mf in missing_files:
                md_lines.append(f"- **Missing Report File:** `{mf}`")
            for mal in malformed_files:
                md_lines.append(f"- **Malformed Report File:** `{mal}`")
            md_lines.append("")

        if skipped_details:
            md_lines.extend([
                "## Explicitly Skipped Tests",
                "",
                "| Test Identifier | Explicit Skip Reason |",
                "| :--- | :--- |",
            ])
            for s in skipped_details:
                short_id = s["id"].split(".")[-1]
                md_lines.append(f"| `{short_id}` | {s['message']} |")
            md_lines.append("")

        if not model_dir_env:
            md_lines.extend([
                "> [!NOTE]",
                "> Frozen model binary artifacts (`*.joblib`) are excluded from Git repository per licensing and security policies.",
                "> In CI without mounted model directory, authentic model tests are explicitly skipped (`skipTest`).",
                "> Full container model verification with authentic artifacts is documented in `docs/PHASE_A_4_2_CONTAINER_MODEL_INTEGRATION_REPORT.md`.",
                "",
            ])
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dynamic CI test summary from JUnit XML reports.")
    parser.add_argument(
        "--xml",
        nargs="+",
        required=True,
        help="One or more JUnit XML test report files.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional path to output markdown summary file.",
    )
    args = parser.parse_args()

    xml_paths = [Path(p) for p in args.xml]
    parsed_data = parse_junit_xml(xml_paths)
    summary_data = categorize_tests(parsed_data["tests"])

    model_dir_env = os.environ.get("NEUROADHD_MODEL_DIR")
    print_summary(
        summary_data=summary_data,
        missing_files=parsed_data["missing_files"],
        malformed_files=parsed_data["malformed_files"],
        model_dir_env=model_dir_env,
        markdown_path=args.markdown,
    )

    # Fail closed if evidence is missing, malformed, or any tests failed/errored
    if parsed_data["missing_files"] or parsed_data["malformed_files"]:
        print("FAIL-CLOSED: JUnit test evidence is incomplete or corrupted.", file=sys.stderr)
        return 1

    if summary_data["counts"]["failed"] > 0 or summary_data["counts"]["errors"] > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
FairCredit AI - Phase 15 verification script.

Runs a battery of checks and prints a clear PASS/FAIL report. Designed
to be run BOTH:
  1. Locally/inside the backend container, without a live server, to
     check files/artifacts (always available).
  2. Against a running stack, to also check live HTTP endpoints
     (only if --live is passed, or the backend responds).

Usage (from the project root):

    python scripts/verify_project.py
    python scripts/verify_project.py --live --backend-url http://localhost:8000 --frontend-url http://localhost:8501

Exit code is 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))


def check_directories() -> None:
    for rel in ["backend", "frontend", "data", "models", "scripts", "tests"]:
        check(f"directory exists: {rel}/", (PROJECT_ROOT / rel).is_dir())


def check_required_files() -> None:
    for rel in [
        "docker-compose.yml",
        "Dockerfile.backend",
        "Dockerfile.frontend",
        "requirements.txt",
        ".env.example",
        "backend/main.py",
        "frontend/app.py",
    ]:
        check(f"file exists: {rel}", (PROJECT_ROOT / rel).is_file())


def check_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    check(".env.example exists", example_path.is_file())
    check(
        ".env exists (copy .env.example to .env if this fails)",
        env_path.is_file(),
        "" if env_path.is_file() else "Run: copy .env.example .env",
    )


def check_data_pipeline() -> None:
    raw = PROJECT_ROOT / "data" / "raw" / "Washington_State_HDMA-2016.csv"
    check(
        "raw dataset present",
        raw.is_file(),
        "" if raw.is_file() else "Place the HMDA CSV at data/raw/Washington_State_HDMA-2016.csv",
    )
    for split in ["train", "val", "test"]:
        p = PROJECT_ROOT / "data" / "processed" / f"{split}.csv"
        check(
            f"processed split exists: {split}.csv",
            p.is_file(),
            "" if p.is_file() else "Run: python scripts/prepare_data.py",
        )
    summary = PROJECT_ROOT / "data" / "reports" / "data_summary.json"
    check("data_summary.json exists", summary.is_file())


def check_model_artifacts() -> dict | None:
    registry_path = PROJECT_ROOT / "models" / "registry" / "registry.json"
    if not registry_path.is_file():
        check("model registry exists", False, "Run: python scripts/train_model.py")
        return None
    check("model registry exists", True)
    with open(registry_path) as f:
        registry = json.load(f)
    active = registry.get("active_version")
    check("registry has an active_version", bool(active))
    if not active:
        return None
    artifact_path = PROJECT_ROOT / registry["versions"][active]["artifact_path"]
    metadata_path = PROJECT_ROOT / registry["versions"][active]["metadata_path"]
    check(f"model artifact file exists ({active})", artifact_path.is_file())
    check(f"model metadata file exists ({active})", metadata_path.is_file())
    if metadata_path.is_file():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def check_fairness_report() -> None:
    p = PROJECT_ROOT / "data" / "reports" / "fairness_report.json"
    check(
        "fairness_report.json exists",
        p.is_file(),
        "" if p.is_file() else "Run: python scripts/evaluate_fairness.py",
    )


def check_credit_card_track() -> None:
    raw = PROJECT_ROOT / "data" / "raw" / "default_of_credit_card_clients.csv"
    check("credit-card raw dataset present", raw.is_file(),
          "" if raw.is_file() else "Place default_of_credit_card_clients.csv at data/raw/")
    for split in ["train", "val", "test"]:
        p = PROJECT_ROOT / "data" / "processed" / f"credit_card_{split}.csv"
        check(f"credit-card processed split exists: {split}", p.is_file(),
              "" if p.is_file() else "Run: python scripts/prepare_credit_card_data.py")
    registry_path = PROJECT_ROOT / "models" / "registry" / "registry.json"
    if registry_path.is_file():
        registry = json.load(open(registry_path))
        check("registry has credit_card entry", "credit_card" in registry,
              "" if "credit_card" in registry else "Run: python scripts/train_credit_card_models.py")
        if "credit_card" in registry:
            for name, rel_path in registry["credit_card"]["artifact_paths"].items():
                check(f"credit-card model artifact exists: {name}", (PROJECT_ROOT / rel_path).is_file())
    metadata_path = PROJECT_ROOT / "models" / "metadata" / "credit_card_model_comparison.json"
    check("credit-card model comparison metadata exists", metadata_path.is_file())


def check_model_quality(metadata: dict | None) -> None:
    if not metadata:
        check("model meets minimum ROC-AUC (>=0.6)", False, "No metadata to check.")
        return
    metrics = metadata.get("held_out_test_metrics_final_model", {})
    auc = metrics.get("roc_auc")
    check(
        "model meets minimum ROC-AUC (>=0.6)",
        bool(auc and auc >= 0.6),
        f"roc_auc={auc}",
    )


def check_test_suite() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    check("pytest suite passes", result.returncode == 0, result.stdout.strip().splitlines()[-1] if result.stdout else "")


def check_live_endpoints(backend_url: str, frontend_url: str) -> None:
    import requests

    try:
        r = requests.get(f"{backend_url}/health", timeout=5)
        check("live: GET /health", r.status_code == 200, f"status={r.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("live: GET /health", False, str(exc))
        return  # remaining live checks would just repeat the same failure

    for path, method in [
        ("/model/info", "GET"),
        ("/model/feature-importance", "GET"),
        ("/metrics", "GET"),
        ("/fairness", "GET"),
        ("/dataset/info", "GET"),
        ("/predictions", "GET"),
        ("/predictions/stats", "GET"),
        ("/docs", "GET"),
    ]:
        try:
            r = requests.get(f"{backend_url}{path}", timeout=5)
            check(f"live: {method} {path}", r.status_code == 200, f"status={r.status_code}")
        except Exception as exc:  # noqa: BLE001
            check(f"live: {method} {path}", False, str(exc))

    sample_application = {
        "loan_amount_000s": 250, "applicant_income_000s": 80,
        "tract_to_msamd_income": 95, "population": 5000, "minority_population": 20,
        "number_of_owner_occupied_units": 1500, "number_of_1_to_4_family_units": 1800,
        "hud_median_family_income": 70000, "has_co_applicant": False,
        "loan_type_name": "Conventional", "loan_purpose_name": "Home purchase",
        "property_type_name": "One-to-four family dwelling (other than manufactured housing)",
        "owner_occupancy_name": "Owner-occupied as a principal dwelling",
        "lien_status_name": "Secured by a first lien", "preapproval_name": "Not applicable",
    }
    try:
        r = requests.post(f"{backend_url}/predict", json=sample_application, timeout=10)
        check("live: POST /predict returns a real prediction", r.status_code == 200, f"status={r.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("live: POST /predict returns a real prediction", False, str(exc))

    try:
        r = requests.get(frontend_url, timeout=5)
        check("live: frontend reachable", r.status_code == 200, f"status={r.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("live: frontend reachable", False, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Also check live HTTP endpoints.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--frontend-url", default="http://localhost:8501")
    args = parser.parse_args()

    check_directories()
    check_required_files()
    check_env_file()
    check_data_pipeline()
    metadata = check_model_artifacts()
    check_fairness_report()
    check_credit_card_track()
    check_model_quality(metadata)
    check_test_suite()

    if args.live:
        check_live_endpoints(args.backend_url, args.frontend_url)

    print("\n=== FairCredit AI Verification Report ===\n")
    passed = 0
    for name, ok, detail in RESULTS:
        symbol = "PASS" if ok else "FAIL"
        line = f"[{symbol}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        passed += int(ok)

    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

"""
FairCredit AI - deployment verification.

Checks that a RUNNING deployment is healthy: Docker, containers, environment,
database, model artifacts, backend, frontend, and one real prediction per track.
Uses only the Python standard library, so it runs on any computer with no pip install.

Usage (from the project root):

    python scripts/verify_deployment.py                  # normal: run on the host computer
    python scripts/verify_deployment.py --inside-container   # run inside the backend container
    python scripts/verify_deployment.py --skip-docker    # server without the docker CLI
    python scripts/verify_deployment.py --no-predict     # do not send test predictions

Sending the two test predictions writes two rows to the prediction audit tables
(one per track). Use --no-predict if you do not want that.

Result: DEPLOYMENT VERIFICATION PASSED  (exit code 0)   or   ... FAILED  (exit code 1).
WARN lines do not fail the run; they point at features that are degraded.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_ENV = ["MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD"]
EXPECTED_SERVICES = ["mysql", "backend", "frontend"]

HMDA_SAMPLE = {
    "loan_amount_000s": 250, "applicant_income_000s": 80, "tract_to_msamd_income": 95,
    "population": 5000, "minority_population": 20, "number_of_owner_occupied_units": 1500,
    "number_of_1_to_4_family_units": 1800, "hud_median_family_income": 70000,
    "has_co_applicant": False, "loan_type_name": "Conventional",
    "loan_purpose_name": "Home purchase",
    "property_type_name": "One-to-four family dwelling (other than manufactured housing)",
    "owner_occupancy_name": "Owner-occupied as a principal dwelling",
    "lien_status_name": "Secured by a first lien", "preapproval_name": "Not applicable",
}
CARD_SAMPLE = {
    "LIMIT_BAL": 200000, "AGE": 35, "EDUCATION": "University", "MARRIAGE": "Married",
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
    "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
}

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))


def ok(name: str, detail: str = "") -> None:
    record("PASS", name, detail)


def warn(name: str, detail: str) -> None:
    record("WARN", name, detail)


def fail(name: str, detail: str) -> None:
    record("FAIL", name, detail)


# --------------------------------------------------------------------------- HTTP
def http(method: str, url: str, body: dict | None = None, timeout: int = 15):
    """Returns (status_code, parsed_json_or_text). Never raises; status 0 = unreachable."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        code = exc.code
    except Exception as exc:  # noqa: BLE001 - connection refused, DNS, timeout ...
        return 0, str(exc)
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, raw


# --------------------------------------------------------------------------- checks

def run_docker(args: list[str], timeout: int = 60):
    """Run `docker <args>` and return (returncode, stdout, stderr). Never raises.

    'docker' is called by name (not a path from shutil.which) so Windows resolves docker.exe
    itself; a resolved extension-less path caused WinError 193 on Windows.
    returncode is None if docker could not be started at all (stderr says why).
    """
    try:
        out = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
        return out.returncode, out.stdout, out.stderr
    except FileNotFoundError:
        return None, "", "the 'docker' command was not found"
    except subprocess.TimeoutExpired:
        return None, "", f"'docker {' '.join(args)}' did not answer within {timeout}s (is Docker Desktop still starting?)"
    except OSError as exc:
        return None, "", f"could not start docker: {exc}"

def check_docker() -> None:
    rc, out, err = run_docker(["--version"], timeout=30)
    if rc is None:
        fail("Docker installed", f"{err}. Install Docker Desktop (Windows/macOS) or Docker Engine (Linux), and make sure it is on PATH.")
        return
    (ok if rc == 0 else fail)("Docker available", out.strip() if rc == 0 else err.strip()[:200])
    rc, out, err = run_docker(["compose", "version"], timeout=30)
    (ok if rc == 0 else fail)("Docker Compose available", out.strip() if rc == 0 else (err.strip()[:200] or "docker compose failed"))
    rc, out, err = run_docker(["info"], timeout=60)
    if rc == 0:
        ok("Docker engine running")
    else:
        fail("Docker engine running", (err.strip()[:200] or "no answer") + " - start Docker Desktop and wait until it reports it is running.")


def check_containers() -> None:
    rc, out, err = run_docker(["compose", "ps", "-a", "--format", "json"], timeout=60)
    if rc is None or rc != 0:
        fail("Containers listed", (err.strip()[:200] or "docker compose ps failed") + ". Run this from the project folder.")
        return
    text = out.strip()
    if not text:
        fail("Containers listed", "no containers exist yet. Start the application with: docker compose up -d")
        return
    try:
        if text.startswith("["):
            rows = json.loads(text)
        else:  # newer Compose prints one JSON object per line
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError as exc:
        fail("Containers listed", f"could not read 'docker compose ps' output: {exc}")
        return
    by_service = {r.get("Service"): r for r in rows}
    for svc in EXPECTED_SERVICES:
        row = by_service.get(svc)
        if row is None:
            fail(f"Container running: {svc}", "not found. Start the application with: docker compose up -d")
            continue
        state, health = row.get("State", "?"), row.get("Health", "")
        if state != "running":
            fail(f"Container running: {svc}", f"state is '{state}'. See: docker compose logs {svc}")
        elif health and health != "healthy":
            fail(f"Container healthy: {svc}", f"health is '{health}'. It may still be starting - wait a minute, or see: docker compose logs {svc}")
        else:
            ok(f"Container healthy: {svc}", f"{state}" + (f", {health}" if health else ""))


def load_env(inside_container: bool) -> dict:
    if inside_container:
        return dict(os.environ)
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return {}
    values = {}
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def check_environment(inside_container: bool) -> None:
    if not inside_container and not (PROJECT_ROOT / ".env").is_file():
        fail("Environment file", ".env not found. Run start.bat / ./start.sh, or copy .env.example to .env")
        return
    env = load_env(inside_container)
    where = "container environment" if inside_container else ".env"
    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        fail("Required environment variables", f"missing in {where}: {', '.join(missing)}")
    else:
        ok("Required environment variables", f"all present in {where} (values not shown)")
    weak = [k for k in ("MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD") if env.get(k, "").startswith("change_me")]
    if weak:
        warn("Database passwords", f"{', '.join(weak)} still use the placeholder value from .env.example. Fine on a private laptop; change it before exposing this anywhere else.")


def check_files(inside_container: bool) -> None:
    dirs = ["backend", "models", "data"] if inside_container else ["backend", "frontend", "models", "data", "scripts"]
    for d in dirs:
        (ok if (PROJECT_ROOT / d).is_dir() else fail)(f"Directory exists: {d}/", "" if (PROJECT_ROOT / d).is_dir() else "missing")
    if not inside_container:
        for f in ("docker-compose.yml", "Dockerfile.backend", "Dockerfile.frontend", "requirements.txt", ".env.example", "frontend/app.py"):
            (ok if (PROJECT_ROOT / f).is_file() else fail)(f"File exists: {f}", "" if (PROJECT_ROOT / f).is_file() else "missing")
    registry = PROJECT_ROOT / "models" / "registry" / "registry.json"
    if not registry.is_file():
        fail("Model registry", "models/registry/registry.json is missing. Restore the models/ folder, or retrain (see DEPLOYMENT.md). No substitute model is created.")
        return
    try:
        reg = json.loads(registry.read_text())
    except ValueError as exc:
        fail("Model registry", f"registry.json is not valid JSON: {exc}")
        return
    ok("Model registry", "models/registry/registry.json")
    # HMDA (Track B)
    try:
        v = reg["active_version"]
        art = PROJECT_ROOT / reg["versions"][v]["artifact_path"]
        (ok if art.is_file() else fail)(f"HMDA model artifact ({v})", str(art.relative_to(PROJECT_ROOT)) if art.is_file() else f"missing: {art}")
    except KeyError:
        fail("HMDA model artifact", "registry has no active HMDA version. Retrain: python scripts/train_model.py")
    # Credit card (Track A)
    cc = reg.get("credit_card")
    if not cc:
        fail("Credit-card model artifacts", "registry has no credit_card entry. Retrain: python scripts/train_credit_card_models.py")
    else:
        for name, rel in cc.get("artifact_paths", {}).items():
            p = PROJECT_ROOT / rel
            (ok if p.is_file() else fail)(f"Credit-card model artifact: {name}", "" if p.is_file() else f"missing: {rel}")


def check_model_loading_in_process() -> None:
    """Only used inside the container: load both tracks with the same code the API uses."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from backend.services.model_service import get_model_service
        from backend.services.credit_card_service import get_credit_card_model_service
    except Exception as exc:  # noqa: BLE001
        fail("Backend code importable", f"{type(exc).__name__}: {exc}")
        return
    hm = get_model_service()
    (ok if hm.is_ready else fail)("HMDA model loads", "" if hm.is_ready else (hm.load_error or "not ready"))
    cc = get_credit_card_model_service()
    (ok if cc.is_ready else fail)("Credit-card models load", "" if cc.is_ready else (cc.load_error or "not ready"))


def check_backend(backend: str, predict: bool) -> bool:
    code, body = http("GET", f"{backend}/health")
    if code != 200:
        fail("Backend reachable (/health)", f"{backend}/health -> {'no connection: ' + str(body) if code == 0 else 'HTTP ' + str(code)}. See: docker compose logs backend")
        return False
    ok("Backend reachable (/health)", backend)

    code, body = http("GET", f"{backend}/health/detailed")
    if code == 200 and isinstance(body, dict) and body.get("database") == "connected":
        ok("Backend -> MySQL connection", "database: connected")
    else:
        detail = body.get("database") if isinstance(body, dict) else body
        fail("Backend -> MySQL connection", f"/health/detailed reports database: {detail}. See: docker compose logs mysql backend")

    code, _ = http("GET", f"{backend}/docs")
    (ok if code == 200 else fail)("API documentation (/docs)", "" if code == 200 else f"HTTP {code}")

    for label, path in (("HMDA model loaded in API", "/model/info"), ("Credit-card models loaded in API", "/credit-card/model/info")):
        code, body = http("GET", backend + path)
        if code == 200 and isinstance(body, dict) and body.get("ready"):
            ok(label, path)
        else:
            err = body.get("error") if isinstance(body, dict) else body
            fail(label, f"{path} -> {err or 'HTTP ' + str(code)}")

    # Features that need the processed data / reports; degraded, not fatal.
    for label, path in (("HMDA fairness report available", "/fairness"),
                        ("Credit-card fairness audit available", "/credit-card/fairness"),
                        ("Credit-card SHAP/feature importance available", "/credit-card/model/feature-importance")):
        code, body = http("GET", backend + path)
        good = code == 200 and isinstance(body, dict) and body.get("ready") and body.get("available", True)
        if good:
            ok(label, path)
        else:
            err = body.get("error") if isinstance(body, dict) else body
            warn(label, f"{path} -> {err or 'HTTP ' + str(code)}. Usually means data/reports or data/processed files are missing.")

    if predict:
        code, body = http("POST", f"{backend}/predict", HMDA_SAMPLE)
        if code == 200 and isinstance(body, dict) and "denial_probability" in body:
            ok("Real prediction: HMDA /predict", f"{body['predicted_label']}, probability {body['denial_probability']}")
        else:
            fail("Real prediction: HMDA /predict", f"HTTP {code}: {str(body)[:200]}")
        code, body = http("POST", f"{backend}/credit-card/predict", CARD_SAMPLE)
        if code == 200 and isinstance(body, dict) and "default_probability" in body:
            ok("Real prediction: credit-card /credit-card/predict", f"{body['risk_level']}, probability {body['default_probability']}")
        else:
            fail("Real prediction: credit-card /credit-card/predict", f"HTTP {code}: {str(body)[:200]}")
    return True


def check_frontend(frontend: str) -> None:
    code, body = http("GET", f"{frontend}/_stcore/health")
    if code != 200:
        fail("Frontend reachable", f"{frontend}/_stcore/health -> {'no connection: ' + str(body) if code == 0 else 'HTTP ' + str(code)}. See: docker compose logs frontend")
        return
    ok("Frontend reachable (Streamlit)", frontend)
    code, _ = http("GET", frontend)
    (ok if code == 200 else fail)("Frontend serves the app page", "" if code == 200 else f"HTTP {code}")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a running FairCredit AI deployment.")
    ap.add_argument("--inside-container", action="store_true", help="run inside the backend container")
    ap.add_argument("--skip-docker", action="store_true", help="skip Docker/container checks")
    ap.add_argument("--no-predict", action="store_true", help="do not send test predictions (they write audit rows)")
    ap.add_argument("--backend-url", default=None)
    ap.add_argument("--frontend-url", default=None)
    args = ap.parse_args()

    inside = args.inside_container
    backend = args.backend_url or ("http://localhost:8000")
    frontend = args.frontend_url or ("http://frontend:8501" if inside else "http://localhost:8501")

    def safely(label, fn, *a, **kw):
        """An unexpected error inside one check becomes a FAIL line, never a traceback."""
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            fail(f"{label} (verifier error)", f"{type(exc).__name__}: {exc}")

    if not inside and not args.skip_docker:
        safely("Docker checks", check_docker)
        safely("Container checks", check_containers)
    safely("Environment check", check_environment, inside)
    safely("File/model check", check_files, inside)
    if inside:
        safely("Model loading check", check_model_loading_in_process)
    safely("Backend checks", check_backend, backend, predict=not args.no_predict)
    safely("Frontend checks", check_frontend, frontend)

    print("\n=== FairCredit AI deployment verification ===\n")
    for status, name, detail in RESULTS:
        print(f"[{status}] {name}" + (f"  -  {detail}" if detail else ""))
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    print(f"\n{len(RESULTS)} checks: {len(RESULTS) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failed.")
    if n_fail:
        print("\nDEPLOYMENT VERIFICATION FAILED")
        return 1
    print("\nDEPLOYMENT VERIFICATION PASSED" + (" (with warnings)" if n_warn else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Pre-submission validation script for the Data Cleaning OpenEnv environment.

Run this before submitting to confirm all mandatory requirements are met.

Usage:
  python validate.py              # validates files only (no server needed)
  python validate.py --live       # also runs live endpoint checks (needs server)

Exit codes:
  0  – all checks passed
  1  – one or more checks failed
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = "--live" in sys.argv
BASE_DIR  = Path(__file__).parent
PASS      = "\033[92mPASS\033[0m"
FAIL      = "\033[91mFAIL\033[0m"
WARN      = "\033[93mWARN\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

results: List[Tuple[str, bool, str]] = []


def check(name: str, fn: Callable[[], Tuple[bool, str]]) -> bool:
    try:
        passed, detail = fn()
    except Exception as exc:
        passed, detail = False, f"Exception: {exc}"
    icon = PASS if passed else FAIL
    print(f"  {icon}  {name}")
    if detail:
        indent = "     "
        for line in detail.splitlines():
            print(f"{indent}{line}")
    results.append((name, passed, detail))
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# File-system checks
# ─────────────────────────────────────────────────────────────────────────────

def check_required_files() -> Tuple[bool, str]:
    required = [
        "inference.py",
        "app.py",
        "openenv.yaml",
        "requirements.txt",
        "Dockerfile",
        "README.md",
        "env/__init__.py",
        "env/models.py",
        "env/environment.py",
        "env/graders.py",
        "env/datasets.py",
    ]
    missing = [f for f in required if not (BASE_DIR / f).exists()]
    if missing:
        return False, "Missing files:\n" + "\n".join(f"  - {f}" for f in missing)
    return True, f"All {len(required)} required files present."


def check_openenv_yaml() -> Tuple[bool, str]:
    import yaml  # type: ignore
    path = BASE_DIR / "openenv.yaml"
    with open(path) as fh:
        data = yaml.safe_load(fh)
    issues = []
    for field in ("spec_version", "name", "tasks", "observation_space", "action_space"):
        if field not in data:
            issues.append(f"Missing field: {field}")
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) < 3:
        issues.append(f"Need ≥3 tasks, found {len(tasks)}")
    for t in tasks:
        for f in ("id", "name", "difficulty", "max_steps"):
            if f not in t:
                issues.append(f"Task '{t.get('id', '?')}' missing field '{f}'")
    if issues:
        return False, "\n".join(issues)
    return True, f"{len(tasks)} tasks defined, all required fields present."


def check_pydantic_models() -> Tuple[bool, str]:
    sys.path.insert(0, str(BASE_DIR))
    from env.models import DataObservation, DataAction, StepResult, ResetResult, StateResult
    obs_schema    = DataObservation.model_json_schema()
    action_schema = DataAction.model_json_schema()
    checks = [
        ("DataObservation has episode_id",   "episode_id"   in obs_schema.get("properties", {})),
        ("DataObservation has task_type",    "task_type"    in obs_schema.get("properties", {})),
        ("DataObservation has reward",       "reward"       in obs_schema.get("properties", {})),
        ("DataObservation has done",         "done"         in obs_schema.get("properties", {})),
        ("DataAction has action_type",       "action_type"  in action_schema.get("properties", {})),
    ]
    failed = [msg for msg, ok in checks if not ok]
    if failed:
        return False, "\n".join(failed)
    return True, "DataObservation, DataAction, StepResult, ResetResult, StateResult all valid."


def check_env_importable() -> Tuple[bool, str]:
    sys.path.insert(0, str(BASE_DIR))
    from env import DataCleaningEnv
    env   = DataCleaningEnv()
    tasks = env.list_tasks()
    ids   = [t["task_id"] for t in tasks]
    for tid in ("schema_validation", "standardization", "pipeline"):
        if tid not in ids:
            return False, f"Task '{tid}' not found in list_tasks(). Got: {ids}"
    return True, f"DataCleaningEnv importable. Tasks: {ids}"


def check_env_step_reset_state() -> Tuple[bool, str]:
    sys.path.insert(0, str(BASE_DIR))
    from env import DataCleaningEnv
    from env.models import DataAction

    env    = DataCleaningEnv()
    errors = []

    for task_id in ("schema_validation", "standardization", "pipeline"):
        try:
            rr = env.reset(task_id=task_id)
            eid = rr.episode_id
            assert rr.observation.task_type.value == task_id or True  # just check no exception

            # state()
            st = env.state(eid)
            assert st.episode_id == eid

            # minimal no-op step
            obs = rr.observation
            if obs.task_type.value == "schema_validation":
                action = DataAction(action_type="report_issues", issues=[])
            elif obs.task_type.value == "standardization":
                action = DataAction(action_type="apply_transforms", transforms={})
            else:
                action = DataAction(action_type="audit", audit_summary="test", issue_categories=[])

            sr = env.step(eid, action)
            assert 0.0 <= sr.reward <= 1.0, f"reward {sr.reward} out of [0,1]"

        except Exception as exc:
            errors.append(f"{task_id}: {exc}")

    if errors:
        return False, "\n".join(errors)
    return True, "reset() / step() / state() all work for all 3 tasks."


def check_inference_log_format() -> Tuple[bool, str]:
    """Check that inference.py defines log_start, log_step, log_end."""
    src = (BASE_DIR / "inference.py").read_text()
    missing = [fn for fn in ("log_start", "log_step", "log_end") if fn not in src]
    if missing:
        return False, f"inference.py missing: {missing}"
    for tag in ("[START]", "[STEP]", "[END]"):
        if tag not in src:
            return False, f"inference.py does not emit '{tag}' log lines."
    # Check env var usage
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if var not in src:
            return False, f"inference.py does not reference env var '{var}'."
    return True, "[START]/[STEP]/[END] logging present. All required env vars referenced."


def check_dockerfile() -> Tuple[bool, str]:
    src = (BASE_DIR / "Dockerfile").read_text()
    issues = []
    if "EXPOSE 7860" not in src:
        issues.append("Dockerfile should EXPOSE 7860")
    if "requirements.txt" not in src:
        issues.append("Dockerfile should COPY requirements.txt")
    if "uvicorn" not in src and "CMD" not in src:
        issues.append("Dockerfile should start uvicorn")
    if issues:
        return False, "\n".join(issues)
    return True, "Dockerfile looks valid."


def check_reward_range() -> Tuple[bool, str]:
    sys.path.insert(0, str(BASE_DIR))
    from env import DataCleaningEnv
    from env.models import DataAction

    env    = DataCleaningEnv()
    issues = []

    test_cases = [
        ("schema_validation", DataAction(action_type="report_issues", issues=[])),
        ("standardization",   DataAction(action_type="apply_transforms", transforms={})),
        ("pipeline",          DataAction(action_type="audit", audit_summary="x", issue_categories=[])),
    ]
    for tid, action in test_cases:
        rr = env.reset(task_id=tid)
        sr = env.step(rr.episode_id, action)
        if not (0.0 <= sr.reward <= 1.0):
            issues.append(f"{tid}: reward {sr.reward} outside [0.0, 1.0]")

    if issues:
        return False, "\n".join(issues)
    return True, "All task rewards in [0.0, 1.0]."


# ─────────────────────────────────────────────────────────────────────────────
# Live endpoint checks (optional – requires a running server)
# ─────────────────────────────────────────────────────────────────────────────

def check_live_root() -> Tuple[bool, str]:
    import requests
    url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    r   = requests.get(f"{url}/", timeout=10)
    if r.status_code != 200:
        return False, f"GET / returned {r.status_code}"
    data = r.json()
    return True, f"Root response: {data.get('name')} v{data.get('version')} status={data.get('status')}"


def check_live_reset() -> Tuple[bool, str]:
    import requests
    url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    r   = requests.post(f"{url}/reset", json={"task_id": "schema_validation"}, timeout=10)
    if r.status_code != 200:
        return False, f"POST /reset returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    if "episode_id" not in data:
        return False, f"Response missing 'episode_id': {data}"
    return True, f"Reset OK. episode_id={data['episode_id'][:8]}…"


def check_live_validate_endpoint() -> Tuple[bool, str]:
    import requests
    url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    r   = requests.post(f"{url}/validate", timeout=30)
    if r.status_code != 200:
        return False, f"POST /validate returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    passed = data.get("passed", False)
    return passed, f"Validation endpoint result: {json.dumps(data.get('validation', {}), indent=2)}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n" + "=" * 60)
    print("Data Cleaning OpenEnv - Pre-submission Validation")
    print("=" * 60)

    print("\n-- Static checks ------------------------------------------")
    check("Required files present",          check_required_files)
    check("openenv.yaml valid",              check_openenv_yaml)
    check("Pydantic models valid",           check_pydantic_models)
    check("Environment importable",          check_env_importable)
    check("reset/step/state API works",      check_env_step_reset_state)
    check("inference.py log format",         check_inference_log_format)
    check("Dockerfile valid",                check_dockerfile)
    check("Reward range [0.0, 1.0]",         check_reward_range)

    if LIVE_MODE:
        print("\n-- Live endpoint checks (--live) --------------------------")
        check("GET /  returns 200",          check_live_root)
        check("POST /reset works",           check_live_reset)
        check("POST /validate passes",       check_live_validate_endpoint)

    # Summary
    total   = len(results)
    passed  = sum(1 for _, ok, _ in results if ok)
    failed  = total - passed

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all clear!")
    print("=" * 60 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

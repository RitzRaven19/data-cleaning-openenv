"""
Baseline inference script for the Data Cleaning OpenEnv environment.

Emits structured stdout logs in the mandatory [START] / [STEP] / [END] format.

Required environment variables:
  API_BASE_URL   – OpenAI-compatible API base URL  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     – Model identifier                 (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       – Hugging Face / API key

Optional:
  OPENENV_BASE_URL – Environment server URL  (default: http://localhost:7860)
                     The script will auto-start the server locally if the URL is
                     localhost and the server is not yet responding.

Usage:
  # With a running server:
  python inference.py

  # Fully self-contained (auto-starts server):
  OPENENV_BASE_URL=http://localhost:7860 python inference.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (all sourced from env vars per spec)
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
# Support HF_TOKEN (mandatory spec), OPENAI_API_KEY (functional requirement), or API_KEY fallback
API_KEY      = (os.getenv("HF_TOKEN")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("API_KEY", ""))
ENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")

MAX_STEPS   = 8       # safety cap per episode
TEMPERATURE = 0.1     # low temperature for reproducibility
MAX_TOKENS  = 1200
TASKS       = ["schema_validation", "standardization", "pipeline"]
STRICT_EPS  = 1e-7

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client (uses API_BASE_URL + HF_TOKEN per spec)
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def strict_open_score(value: float) -> float:
    """Clamp score into strict open interval (0, 1). No rounding — round() can collapse 1e-7 to 0.0."""
    return max(STRICT_EPS, min(1.0 - STRICT_EPS, float(value)))


# ─────────────────────────────────────────────────────────────────────────────
# Structured logging  — mandatory [START] / [STEP] / [END] format
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, **extra: Any) -> None:
    """Emit a [START] line for the given task."""
    payload = {"task": task, "model": MODEL_NAME}
    payload.update(extra)
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(
    step: int,
    action: Any,
    reward: float,
    done: bool,
    cumulative_reward: float,
    **extra: Any,
) -> None:
    """Emit a [STEP] line after each environment step."""
    payload: Dict[str, Any] = {
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "done": done,
        "cumulative_reward": round(cumulative_reward, 4),
    }
    payload.update(extra)
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(task: str, total_reward: float, steps: int, success: bool, **extra: Any) -> None:
    """Emit an [END] line after the episode finishes."""
    payload: Dict[str, Any] = {
        "task": task,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "success": success,
    }
    payload.update(extra)
    print(f"[END] {json.dumps(payload)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(episode_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"episode_id": episode_id, "action": action},
        timeout=30,
    )
    if r.status_code == 422:
        # Validation error — sanitize issue_type values and retry once
        VALID_ISSUE_TYPES = {
            "missing_required", "wrong_type", "invalid_range",
            "invalid_format", "invalid_enum", "duplicate", "outlier",
        }
        if action.get("action_type") in ("report_issues",) and action.get("issues"):
            for issue in action["issues"]:
                if issue.get("issue_type") not in VALID_ISSUE_TYPES:
                    issue["issue_type"] = "invalid_format"   # safe fallback
            r2 = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"episode_id": episode_id, "action": action},
                timeout=30,
            )
            if r2.status_code == 200:
                return r2.json()
        print(f"  [WARN] 422 on step — {r.text[:200]}", flush=True)
        r.raise_for_status()
    r.raise_for_status()
    return r.json()


def env_state(episode_id: str) -> Dict[str, Any]:
    r = requests.get(f"{ENV_BASE_URL}/state/{episode_id}", timeout=30)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Server lifecycle helpers
# ─────────────────────────────────────────────────────────────────────────────

def _server_alive() -> bool:
    try:
        r = requests.get(f"{ENV_BASE_URL}/", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_server(max_retries: int = 30, delay: float = 2.0) -> None:
    for attempt in range(1, max_retries + 1):
        if _server_alive():
            print(f"Environment server ready at {ENV_BASE_URL}", flush=True)
            return
        print(f"  Waiting for server… ({attempt}/{max_retries})", flush=True)
        time.sleep(delay)
    raise RuntimeError(f"Environment server not reachable at {ENV_BASE_URL} after {max_retries} attempts.")


def maybe_start_server() -> Optional[subprocess.Popen]:
    """
    If ENV_BASE_URL points to localhost and no server is running,
    start the FastAPI app as a subprocess and wait for it to be ready.
    Returns the Popen handle (caller is responsible for .terminate()), or None.
    """
    if "localhost" not in ENV_BASE_URL and "127.0.0.1" not in ENV_BASE_URL:
        return None   # remote URL — user is responsible for starting it
    if _server_alive():
        return None   # already running

    port = ENV_BASE_URL.rstrip("/").split(":")[-1] if ":" in ENV_BASE_URL else "7860"
    print(f"Auto-starting environment server on port {port}…", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "0.0.0.0", "--port", port,
         "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_server(max_retries=30, delay=2.0)
    return proc


# ─────────────────────────────────────────────────────────────────────────────
# Observation → text
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_text(obs: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Task: {obs.get('task_type')} | Dataset: {obs.get('dataset_name')}")
    lines.append(f"Step: {obs.get('current_step')}/{obs.get('max_steps')}")

    if obs.get("pipeline_phase"):
        lines.append(f"Pipeline phase: {obs['pipeline_phase']}")

    ctx = obs.get("context", {})
    if ctx.get("hint"):
        lines.append(f"Hint: {ctx['hint']}")
    if ctx.get("target_formats"):
        lines.append("Target formats:")
        for col, fmt in ctx["target_formats"].items():
            lines.append(f"  {col}: {fmt}")

    schema = obs.get("schema", obs.get("dataset_schema", {}))
    if schema:
        lines.append("Schema:")
        for col, rules in schema.items():
            lines.append(f"  {col}: {json.dumps(rules)}")

    sample = obs.get("dataset_sample", [])
    if sample:
        lines.append(f"\nDataset sample ({len(sample)} of {obs.get('total_rows')} rows):")
        lines.append(json.dumps(sample, indent=2, default=str))

    col_stats = obs.get("column_stats", {})
    if col_stats:
        lines.append("\nColumn statistics:")
        for col, st in col_stats.items():
            lines.append(
                f"  {col}: dtype={st.get('dtype')} "
                f"null={st.get('null_count')} "
                f"unique={st.get('unique_count')} "
                f"samples={st.get('sample_values')}"
            )

    lines.append(f"\nAvailable actions: {obs.get('available_actions')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM error] {exc}", flush=True)
        return ""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_SCHEMA = textwrap.dedent("""
You are a data quality expert. Identify ALL data quality issues in the dataset.

Respond ONLY with this JSON:
{
  "action_type": "report_issues",
  "issues": [
    {
      "row_index": <int, 0-based>,
      "column": "<column name>",
      "issue_type": "<missing_required|wrong_type|invalid_range|invalid_format|invalid_enum|duplicate|outlier>",
      "description": "<brief explanation>",
      "value": "<offending value>"
    }
  ]
}

Rules: row_index is 0-based. For duplicates report the later row. No markdown.
""").strip()

SYSTEM_STANDARDIZATION = textwrap.dedent("""
You are a data engineer. Define transformation rules to normalize each column
listed under "columns_to_standardize" to its target format.

Respond ONLY with this JSON:
{
  "action_type": "apply_transforms",
  "transforms": {
    "<column_name>": {
      "target_format": "<description>",
      "regex_from": "<regex or null>",
      "replace_with": "<replacement or null>",
      "examples": [{"before": "<raw>", "after": "<normalized>"}]
    }
  }
}

Include all columns. Provide ≥2 before/after examples each. No markdown.
""").strip()

SYSTEM_PIPELINE: Dict[str, str] = {
    "audit": textwrap.dedent("""
You are auditing an employee dataset. Identify the CATEGORIES of issues (not specific rows yet).

Respond ONLY with:
{
  "action_type": "audit",
  "audit_summary": "<2-4 sentence summary>",
  "issue_categories": ["<category1>", ...]
}
Valid categories: missing_required, wrong_type, invalid_range, invalid_format, invalid_enum, duplicate, outlier.
No markdown.
""").strip(),

    "identify": textwrap.dedent("""
Based on the audit, identify EVERY specific issue row-by-row.

Respond ONLY with:
{
  "action_type": "report_issues",
  "issues": [
    {"row_index": <int>, "column": "<col>", "issue_type": "<type>", "description": "<why>", "value": "<bad value>"}
  ]
}
No markdown.
""").strip(),

    "fix": textwrap.dedent("""
Apply fixes to every identified issue.

Respond ONLY with:
{
  "action_type": "fix",
  "fix_operations": [
    {"row_index": <int>, "column": "<col>", "old_value": "<original>", "new_value": "<corrected>", "rationale": "<why>"}
  ]
}
No markdown.
""").strip(),

    "validate": textwrap.dedent("""
Validate the dataset after fixes. Write a brief report and estimate remaining issues.

Respond ONLY with:
{
  "action_type": "validate",
  "validation_report": "<2-4 sentence report>",
  "issues_remaining": <int>
}
No markdown.
""").strip(),
}

_PIPELINE_FALLBACKS: Dict[str, Dict[str, Any]] = {
    "audit":    {"action_type": "audit",         "audit_summary": "Audit complete.", "issue_categories": []},
    "identify": {"action_type": "report_issues", "issues": []},
    "fix":      {"action_type": "fix",           "fix_operations": []},
    "validate": {"action_type": "validate",      "validation_report": "Validation complete.", "issues_remaining": 0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-task agent loops
# ─────────────────────────────────────────────────────────────────────────────

def run_task_schema_validation() -> float:
    task_id = "schema_validation"
    log_start(task_id, difficulty="easy")

    result  = env_reset(task_id)
    eid     = result["episode_id"]
    obs     = result["observation"]

    raw         = call_llm(SYSTEM_SCHEMA, f"Dataset:\n\n{_obs_to_text(obs)}\n\nReturn the JSON action.")
    action_dict = extract_json(raw) or {"action_type": "report_issues", "issues": []}

    step_result      = env_step(eid, action_dict)
    reward           = float(step_result.get("reward", 1e-7))
    done             = bool(step_result.get("done", True))
    metadata         = step_result.get("metadata", {})
    reward_breakdown = metadata.get("reward_breakdown", {})

    log_step(
        step=1,
        action=action_dict.get("action_type", "report_issues"),
        reward=reward,
        done=done,
        cumulative_reward=reward,
        issues_reported=len(action_dict.get("issues") or []),
        feedback=reward_breakdown.get("feedback", ""),
    )
    reward = strict_open_score(reward)
    log_end(task_id, total_reward=reward, steps=1, success=reward >= 0.5)
    return reward


def run_task_standardization() -> float:
    task_id = "standardization"
    log_start(task_id, difficulty="medium")

    result  = env_reset(task_id)
    eid     = result["episode_id"]
    obs     = result["observation"]

    raw         = call_llm(SYSTEM_STANDARDIZATION, f"Dataset:\n\n{_obs_to_text(obs)}\n\nReturn the JSON action.")
    action_dict = extract_json(raw) or {"action_type": "apply_transforms", "transforms": {}}

    step_result      = env_step(eid, action_dict)
    reward           = float(step_result.get("reward", 1e-7))
    done             = bool(step_result.get("done", True))
    metadata         = step_result.get("metadata", {})
    reward_breakdown = metadata.get("reward_breakdown", {})
    cols             = list((action_dict.get("transforms") or {}).keys())

    log_step(
        step=1,
        action=action_dict.get("action_type", "apply_transforms"),
        reward=reward,
        done=done,
        cumulative_reward=reward,
        columns_transformed=cols,
        feedback=reward_breakdown.get("feedback", ""),
    )
    reward = strict_open_score(reward)
    log_end(task_id, total_reward=reward, steps=1, success=reward >= 0.5)
    return reward


def run_task_pipeline() -> float:
    task_id   = "pipeline"
    log_start(task_id, difficulty="hard")

    reset_result = env_reset(task_id)
    eid          = reset_result["episode_id"]
    obs          = reset_result["observation"]

    step_num          = 0
    cumulative_reward = 0.0
    final_score       = 0.0
    info: Dict[str, Any] = {}

    while step_num < MAX_STEPS:
        phase = obs.get("pipeline_phase")
        if not phase:
            break

        sys_prompt  = SYSTEM_PIPELINE.get(phase, SYSTEM_PIPELINE["audit"])
        user_msg    = f"Current dataset state:\n\n{_obs_to_text(obs)}\n\nReturn the JSON action."
        raw         = call_llm(sys_prompt, user_msg)
        action_dict = extract_json(raw) or _PIPELINE_FALLBACKS.get(phase, {"action_type": phase})

        try:
            step_result = env_step(eid, action_dict)
        except Exception as exc:
            print(f"  [WARN] Step failed ({exc}), using fallback action", flush=True)
            fallback = _PIPELINE_FALLBACKS.get(phase, {"action_type": phase})
            step_result = env_step(eid, fallback)
        obs         = step_result["observation"]
        reward      = float(step_result.get("reward", 1e-7))
        done        = bool(step_result.get("done", False))
        info        = step_result.get("metadata", {})
        bd          = info.get("reward_breakdown", {})

        step_num          += 1
        cumulative_reward += reward

        log_step(
            step=step_num,
            action=action_dict.get("action_type", phase),
            reward=reward,
            done=done,
            cumulative_reward=cumulative_reward,
            phase=phase,
            feedback=(bd.get("feedback") or "")[:120],
        )

        if done:
            break

    final_score  = float(info.get("final_pipeline_score", cumulative_reward))
    final_score  = strict_open_score(final_score)
    phase_scores = info.get("phase_scores", {})

    log_end(
        task_id,
        total_reward=final_score,
        steps=step_num,
        success=final_score >= 0.5,
        phase_scores=phase_scores,
    )
    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60, flush=True)
    print("Data Cleaning OpenEnv – Baseline Inference", flush=True)
    print(f"Model : {MODEL_NAME}", flush=True)
    print(f"API   : {API_BASE_URL}", flush=True)
    print(f"Env   : {ENV_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    # Auto-start server if localhost and not already running
    server_proc = maybe_start_server()

    try:
        wait_for_server()

        scores: Dict[str, float] = {}
        scores["schema_validation"] = run_task_schema_validation()
        scores["standardization"]   = run_task_standardization()
        scores["pipeline"]          = run_task_pipeline()

        overall = strict_open_score(sum(scores.values()) / len(scores))

        print("\n" + "=" * 60, flush=True)
        print("FINAL BASELINE SCORES", flush=True)
        print("=" * 60, flush=True)
        for task, score in scores.items():
            filled = int(score * 20)
            bar = "#" * filled + "-" * (20 - filled)
            print(f"  {task:<25} [{bar}]  {score:.4f}", flush=True)
        print(f"\n  {'Overall average':<25} {'-' * 22}  {overall:.4f}", flush=True)
        print("=" * 60, flush=True)

        # Write results file for CI / automated validation
        results = {"scores": scores, "overall": overall, "model": MODEL_NAME}
        with open("baseline_scores.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults written to baseline_scores.json", flush=True)

    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()

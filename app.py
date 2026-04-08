"""
FastAPI server for the Data Cleaning OpenEnv environment.

Implements the full OpenEnv HTTP spec (ref: meta-pytorch/OpenEnv):
  Simulation endpoints:
    POST /reset      – start / restart an episode
    POST /step       – advance the episode with an action
    GET  /state      – current episode state
  Production / management endpoints:
    GET  /health     – liveness check
    GET  /schema     – action + observation JSON schemas
    GET  /metadata   – environment metadata
  Discovery:
    GET  /           – health alias (validator ping)
    GET  /tasks      – list available task IDs
  Validation helper:
    POST /validate   – quick sanity-check across all tasks
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import DataCleaningEnv
from env.models import DataAction, DataObservation, ResetResult, StateResult, StepResult

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Data Cleaning OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents learn to detect, "
        "diagnose, and fix data quality issues across three difficulty levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENV = DataCleaningEnv()

# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "schema_validation"


class StepRequest(BaseModel):
    episode_id: str
    action: DataAction


# ─────────────────────────────────────────────────────────────────────────────
# Simulation endpoints  (OpenEnv spec §3.1)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResult)
def reset(req: ResetRequest):
    """Start a new episode for the given task. Returns the initial observation."""
    try:
        return ENV.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Advance the episode. Returns observation, reward, done, metadata."""
    try:
        return ENV.step(req.episode_id, req.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state/{episode_id}", response_model=StateResult)
def state(episode_id: str):
    """Return the current internal state of an episode."""
    try:
        return ENV.state(episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Production / management endpoints  (OpenEnv spec §3.2)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — required by openenv validate."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/schema")
def schema():
    """Return JSON schemas for Action and Observation types."""
    return {
        "action":      DataAction.model_json_schema(),
        "observation": DataObservation.model_json_schema(),
    }


@app.get("/metadata")
def metadata():
    """Return structured environment metadata."""
    return {
        "name":        "data-cleaning-env",
        "version":     "1.0.0",
        "description": (
            "OpenEnv environment simulating real-world data quality tasks: "
            "schema validation, format standardization, and multi-step pipeline."
        ),
        "author":      "Ritu Dey",
        "tasks":       ENV.list_tasks(),
        "openenv_spec": "1.0",
        "tags":        ["data-quality", "real-world", "tabular", "openenv"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Discovery / convenience
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_model=Dict[str, Any])
def root():
    """Root health-check — validator pings this URL for a 200 response."""
    return {
        "name":        "data-cleaning-env",
        "version":     "1.0.0",
        "status":      "healthy",
        "tasks":       [t["task_id"] for t in ENV.list_tasks()],
        "openenv_spec": "1.0",
        "docs":        "/docs",
    }


@app.get("/tasks")
def get_tasks():
    return {"tasks": ENV.list_tasks()}


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/validate")
def validate_env():
    """
    Run reset + one no-op step on every task.
    Used by `openenv validate` and CI pipelines.
    """
    results: Dict[str, Any] = {}
    for task in ENV.list_tasks():
        tid = task["task_id"]
        try:
            rr     = ENV.reset(task_id=tid)
            eid    = rr.episode_id
            obs    = rr.observation
            # Minimal (empty) action so grader can evaluate 0-score path
            if obs.task_type.value == "schema_validation":
                action = DataAction(action_type="report_issues", issues=[])
            elif obs.task_type.value == "standardization":
                action = DataAction(action_type="apply_transforms", transforms={})
            else:
                action = DataAction(action_type="audit",
                                    audit_summary="audit test", issue_categories=[])
            sr = ENV.step(eid, action)
            results[tid] = {
                "status": "ok",
                "reward": sr.reward,
                "done":   sr.done,
            }
        except Exception as exc:
            results[tid] = {"status": "error", "detail": str(exc)}
    all_ok = all(v["status"] == "ok" for v in results.values())
    return {"validation": results, "passed": all_ok}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
    except KeyboardInterrupt:
        # Expected when stopping the dev server with Ctrl+C.
        print("Server stopped.")

"""
DataCleaningEnv – core environment logic.

Public API (mirrors OpenEnv spec):
  reset(task_id)          → ResetResult
  step(episode_id, action) → StepResult
  state(episode_id)        → StateResult
  list_tasks()             → list[dict]
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .datasets import TASK_REGISTRY
from .graders import (
    grade_schema_validation,
    grade_standardization,
    grade_pipeline_audit,
    grade_pipeline_identify,
    grade_pipeline_fix,
    grade_pipeline_validate,
    grade_pipeline_episode,
    safe_score,
)
from .models import (
    ColumnStats,
    DataAction,
    DataObservation,
    PipelineStep,
    ResetResult,
    RewardBreakdown,
    StateResult,
    StepResult,
    TaskType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal episode state
# ─────────────────────────────────────────────────────────────────────────────

class _Episode:
    def __init__(self, episode_id: str, task_id: str):
        self.episode_id  = episode_id
        self.task_id     = task_id
        self.task_cfg    = TASK_REGISTRY[task_id]
        self.task_type   = TaskType(self.task_cfg["task_type"])
        self.max_steps   = self.task_cfg["max_steps"]
        self.current_step        = 0
        self.done                = False
        self.cumulative_reward   = 0.0
        self.action_history: List[Dict[str, Any]] = []
        self.pipeline_phase: Optional[PipelineStep] = (
            PipelineStep.AUDIT if self.task_type == TaskType.PIPELINE else None
        )
        # For pipeline – accumulate per-phase scores
        self.pipeline_scores: Dict[str, float] = {}
        self._last_reward: float = 0.11  # reward from the most recent step (safe default)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build observation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_column_stats(rows: List[Dict[str, Any]]) -> Dict[str, ColumnStats]:
    if not rows:
        return {}
    columns = list(rows[0].keys())
    stats: Dict[str, ColumnStats] = {}
    for col in columns:
        values = [r.get(col) for r in rows]
        non_null = [v for v in values if v is not None]
        null_count = len(values) - len(non_null)
        unique_count = len(set(str(v) for v in non_null))
        dtype = "mixed"
        if non_null:
            types = {type(v).__name__ for v in non_null}
            dtype = types.pop() if len(types) == 1 else "mixed"
        nums = [v for v in non_null if isinstance(v, (int, float))]
        stats[col] = ColumnStats(
            dtype=dtype,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=non_null[:5],
            min_value=min(nums) if nums else None,
            max_value=max(nums) if nums else None,
        )
    return stats


def _available_actions(ep: _Episode) -> List[str]:
    if ep.done:
        return []
    t = ep.task_type
    if t == TaskType.SCHEMA_VALIDATION:
        return ["report_issues"]
    if t == TaskType.STANDARDIZATION:
        return ["apply_transforms"]
    # pipeline
    phase_map = {
        PipelineStep.AUDIT:    ["audit"],
        PipelineStep.IDENTIFY: ["report_issues"],
        PipelineStep.FIX:      ["fix"],
        PipelineStep.VALIDATE: ["validate"],
    }
    return phase_map.get(ep.pipeline_phase, [])


def _build_observation(ep: _Episode) -> DataObservation:
    cfg   = ep.task_cfg
    rows  = cfg["rows"]
    n     = cfg.get("sample_size", 10)
    sample = rows[:n]

    context: Dict[str, Any] = {}
    if ep.task_type == TaskType.STANDARDIZATION:
        context["target_formats"] = cfg.get("target_formats", {})
        context["columns_to_standardize"] = list(
            cfg.get("target_formats", {}).keys()
        )
    if ep.task_type == TaskType.PIPELINE:
        context["current_phase"] = ep.pipeline_phase.value if ep.pipeline_phase else None
        context["phases_completed"] = list(ep.pipeline_scores.keys())
        if ep.pipeline_phase == PipelineStep.IDENTIFY:
            context["hint"] = (
                "Based on your audit, now identify every specific issue "
                "(row_index, column, issue_type, description)."
            )
        elif ep.pipeline_phase == PipelineStep.FIX:
            context["hint"] = (
                "Apply fix_operations to address the issues you identified."
            )
        elif ep.pipeline_phase == PipelineStep.VALIDATE:
            context["hint"] = (
                "Write a validation_report and estimate issues_remaining."
            )

    return DataObservation(
        episode_id      = ep.episode_id,
        task_type       = ep.task_type,
        dataset_name    = cfg["name"],
        dataset_sample  = sample,
        total_rows      = len(rows),
        dataset_schema  = cfg.get("schema", {}),
        column_stats    = _compute_column_stats(sample),
        current_step    = ep.current_step,
        max_steps       = ep.max_steps,
        pipeline_phase  = ep.pipeline_phase,
        issues_found    = [],
        available_actions = _available_actions(ep),
        reward          = ep._last_reward,
        done            = ep.done,
        cumulative_reward = safe_score(ep.cumulative_reward),
        context         = context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaningEnv:
    """
    OpenEnv-compliant data cleaning environment.

    Thread safety: episodes are stored in a plain dict. For concurrent
    multi-user deployments wrap access with a lock; this implementation is
    fine for single-session (validator / inference script) usage.
    """

    def __init__(self) -> None:
        self._episodes: Dict[str, _Episode] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id":    tid,
                "name":       cfg["name"],
                "description": cfg["description"],
                "difficulty": cfg["difficulty"],
                "max_steps":  cfg["max_steps"],
                "task_type":  cfg["task_type"],
            }
            for tid, cfg in TASK_REGISTRY.items()
        ]

    def reset(self, task_id: str = "schema_validation") -> ResetResult:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. "
                             f"Valid: {list(TASK_REGISTRY.keys())}")
        episode_id = str(uuid.uuid4())
        ep = _Episode(episode_id, task_id)
        self._episodes[episode_id] = ep
        obs = _build_observation(ep)
        return ResetResult(observation=obs, episode_id=episode_id)

    def step(self, episode_id: str, action: DataAction) -> StepResult:
        if episode_id not in self._episodes:
            raise KeyError(f"No episode '{episode_id}'. Call /reset first.")
        ep = self._episodes[episode_id]

        if ep.done:
            obs = _build_observation(ep)
            return StepResult(
                observation=obs, reward=0.11, done=True,
                metadata={"error": "Episode is already done."}
            )

        ep.current_step += 1
        reward_breakdown, done, next_phase = self._dispatch(ep, action)
        # Clamp to [0.11, 0.98] — strictly inside (0.1, 0.99) inside (0, 1)
        reward = max(0.11, min(0.98, float(reward_breakdown.total)))

        # Record history entry
        ep.action_history.append({
            "step":    ep.current_step,
            "action":  action.action_type,
            "reward":  reward,
            "feedback": reward_breakdown.feedback,
        })
        ep._last_reward    = reward
        ep.cumulative_reward += reward

        # Advance pipeline phase
        if ep.task_type == TaskType.PIPELINE and next_phase:
            ep.pipeline_phase = next_phase

        # Check episode termination
        if done or ep.current_step >= ep.max_steps:
            ep.done = True

        obs = _build_observation(ep)
        metadata: Dict[str, Any] = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "step": ep.current_step,
        }
        if ep.task_type == TaskType.PIPELINE and ep.done:
            final = grade_pipeline_episode(ep.pipeline_scores)
            metadata["final_pipeline_score"] = final
            metadata["phase_scores"] = ep.pipeline_scores

        return StepResult(observation=obs, reward=reward, done=ep.done, metadata=metadata)

    def state(self, episode_id: str) -> StateResult:
        if episode_id not in self._episodes:
            raise KeyError(f"No episode '{episode_id}'.")
        ep = self._episodes[episode_id]
        return StateResult(
            episode_id        = ep.episode_id,
            task_type         = ep.task_type,
            current_step      = ep.current_step,
            max_steps         = ep.max_steps,
            pipeline_phase    = ep.pipeline_phase,
            done              = ep.done,
            cumulative_reward = ep.cumulative_reward,
            action_history    = ep.action_history,
        )

    # ── Internal dispatch ────────────────────────────────────────────────────

    def _dispatch(
        self,
        ep: _Episode,
        action: DataAction,
    ) -> tuple[RewardBreakdown, bool, Optional[PipelineStep]]:
        """
        Returns (reward_breakdown, done, next_pipeline_phase).
        """
        cfg = ep.task_cfg

        if ep.task_type == TaskType.SCHEMA_VALIDATION:
            bd = grade_schema_validation(action, cfg["known_issues"])
            return bd, True, None

        if ep.task_type == TaskType.STANDARDIZATION:
            bd = grade_standardization(action, cfg["rows"], cfg["ground_truth"])
            return bd, True, None

        # ── PIPELINE ──
        phase = ep.pipeline_phase
        next_phase = None
        done = False

        if phase == PipelineStep.AUDIT:
            bd = grade_pipeline_audit(action, cfg["issue_categories"])
            ep.pipeline_scores["audit"] = bd.total
            next_phase = PipelineStep.IDENTIFY

        elif phase == PipelineStep.IDENTIFY:
            bd = grade_pipeline_identify(action, cfg["known_issues"])
            ep.pipeline_scores["identify"] = bd.total
            next_phase = PipelineStep.FIX

        elif phase == PipelineStep.FIX:
            bd = grade_pipeline_fix(action, cfg["known_issues"])
            ep.pipeline_scores["fix"] = bd.total
            # Reconstruct integer count from the normalized ratio stored in components
            issue_n = max(len(cfg["known_issues"]), 1)
            ep._fix_addressed = int(round(
                float(bd.components.get("addressed_ratio", 1e-7)) * issue_n
            ))
            next_phase = PipelineStep.VALIDATE

        elif phase == PipelineStep.VALIDATE:
            issues_fixed = getattr(ep, "_fix_addressed", 0)
            bd = grade_pipeline_validate(
                action, issues_fixed, len(cfg["known_issues"])
            )
            ep.pipeline_scores["validate"] = bd.total
            done = True
            next_phase = None

        else:
            raise ValueError(f"Unknown pipeline phase: {phase}")

        return bd, done, next_phase

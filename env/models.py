"""
Pydantic models for the Data Cleaning OpenEnv environment.

Observation  →  what the agent sees at each step
Action       →  what the agent submits
StepResult   →  what step() returns
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Domain enums
# ─────────────────────────────────────────────

class TaskType(str, Enum):
    SCHEMA_VALIDATION  = "schema_validation"   # easy
    STANDARDIZATION    = "standardization"      # medium
    PIPELINE           = "pipeline"             # hard


class IssueType(str, Enum):
    MISSING_REQUIRED  = "missing_required"
    WRONG_TYPE        = "wrong_type"
    INVALID_RANGE     = "invalid_range"
    INVALID_FORMAT    = "invalid_format"
    INVALID_ENUM      = "invalid_enum"
    DUPLICATE         = "duplicate"
    OUTLIER           = "outlier"


class PipelineStep(str, Enum):
    AUDIT    = "audit"
    IDENTIFY = "identify"
    FIX      = "fix"
    VALIDATE = "validate"


# ─────────────────────────────────────────────
# Sub-models shared by Observation and Action
# ─────────────────────────────────────────────

class ColumnStats(BaseModel):
    """Summary statistics for a single column, surfaced in the observation."""
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None


class DataIssue(BaseModel):
    """Represents one data quality issue reported by the agent."""
    row_index: int          = Field(..., description="0-based row index")
    column:    str          = Field(..., description="Column name where issue occurs")
    issue_type: IssueType   = Field(..., description="Category of the issue")
    description: str        = Field("", description="Human-readable description")
    value: Optional[Any]    = Field(None, description="The offending value (if applicable)")


class ColumnTransform(BaseModel):
    """One transformation rule for a column, used in the standardization task."""
    target_format: str  = Field(..., description="Description of the target format")
    regex_from: Optional[str] = Field(None, description="Regex pattern to match source values")
    replace_with: Optional[str] = Field(None, description="Replacement string or function name")
    examples: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {before, after} examples"
    )


class FixOperation(BaseModel):
    """One atomic fix applied during the pipeline fix step."""
    row_index: int
    column: str
    old_value: Optional[Any]
    new_value: Optional[Any]
    rationale: str = ""


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class DataObservation(BaseModel):
    """Everything the agent sees at the current step."""

    episode_id:    str
    task_type:     TaskType
    dataset_name:  str

    # The first N rows of the dataset (capped to avoid huge context)
    dataset_sample: List[Dict[str, Any]] = Field(default_factory=list)
    total_rows:    int = 0

    # Schema definition (provided for validation & pipeline tasks)
    dataset_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")

    # Per-column statistics
    column_stats: Dict[str, ColumnStats] = Field(default_factory=dict)

    # Step tracking
    current_step:  int = 0
    max_steps:     int = 1
    pipeline_phase: Optional[PipelineStep] = None   # only for pipeline task

    # Running list of issues already found (pipeline task accumulates this)
    issues_found: List[DataIssue] = Field(default_factory=list)

    # List of action type strings the agent may use right now
    available_actions: List[str] = Field(default_factory=list)

    # Per-step reward and terminal flag — matches OpenEnv Observation contract
    # (same pattern as echo_env: EchoObservation has reward + done fields)
    reward: float = 0.0
    done:   bool  = False

    # Cumulative reward so far (informational)
    cumulative_reward: float = 0.0

    # Any extra task-specific hints
    context: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class DataAction(BaseModel):
    """
    A single action submitted by the agent.

    The action_type field selects which other fields are relevant:

    - "report_issues"   → issues (Task 1 & pipeline-identify)
    - "apply_transforms"→ transforms (Task 2)
    - "audit"           → audit_summary (pipeline step 1)
    - "fix"             → fix_operations (pipeline step 3)
    - "validate"        → validation_report (pipeline step 4)
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: 'report_issues', 'apply_transforms', "
            "'audit', 'fix', 'validate'"
        )
    )

    # Task 1 – schema validation / pipeline identify step
    issues: Optional[List[DataIssue]] = None

    # Task 2 – standardization
    transforms: Optional[Dict[str, ColumnTransform]] = None

    # Pipeline – audit step
    audit_summary: Optional[str] = None
    issue_categories: Optional[List[str]] = None   # categories spotted during audit

    # Pipeline – fix step
    fix_operations: Optional[List[FixOperation]] = None

    # Pipeline – validate step
    validation_report: Optional[str] = None
    issues_remaining: Optional[int] = None  # agent's estimate of remaining issues


# ─────────────────────────────────────────────
# API response types
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""


class StepResult(BaseModel):
    observation: DataObservation
    reward:      float
    done:        bool
    metadata:    Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: DataObservation
    episode_id:  str


class StateResult(BaseModel):
    episode_id:        str
    task_type:         TaskType
    current_step:      int
    max_steps:         int
    pipeline_phase:    Optional[PipelineStep]
    done:              bool
    cumulative_reward: float
    action_history:    List[Dict[str, Any]] = Field(default_factory=list)

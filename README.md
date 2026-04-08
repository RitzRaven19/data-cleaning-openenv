---
title: Data Cleaning Env
emoji: 🧹
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: bsd-3-clause
short_description: OpenEnv RL environment for AI agent data cleaning tasks
tags:
  - openenv
  - data-quality
  - real-world
  - tabular
---

# 🧹 Data Cleaning OpenEnv

A **real-world OpenEnv environment** where AI agents learn to detect, diagnose,
and fix data quality issues in tabular datasets — a task that data engineers
and analysts perform every day.

---

## Motivation

Data quality is a universal bottleneck. Industry surveys consistently report
that data professionals spend 60–80 % of their time cleaning data, yet
there is no widely-used benchmark environment for training or evaluating agents
on this task. This environment fills that gap.

Three tasks span an easy-to-hard difficulty range that challenges both
small and frontier models:

| # | Task | Difficulty | Max Steps |
|---|------|-----------|-----------|
| 1 | Schema Validation | Easy | 1 |
| 2 | Format Standardization | Medium | 1 |
| 3 | Multi-step Pipeline | Hard | 8 |

---

## Environment Description

All three tasks operate on realistic synthetic tabular datasets (CSV-like
row/column data) with known ground-truth issues.

### Task 1 – Schema Validation *(easy)*

**Dataset:** 30-row customer CRM export.

**Goal:** Identify all 13 data quality issues by checking each row against
a provided JSON schema.

Issue types present:
- `missing_required` – required fields that are null
- `wrong_type` – a string where an integer is expected
- `invalid_range` – numeric values outside [min, max]
- `invalid_format` – malformed email / phone number / date strings
- `invalid_enum` – value not in the allowed set
- `duplicate` – repeated primary-key values

**Action:** `report_issues` — submit a list of `DataIssue` objects.

**Reward:** F1 score over `(row_index, column, issue_type)` tuples.
Partial credit (×0.3) for correct row+column with wrong issue type.

---

### Task 2 – Format Standardization *(medium)*

**Dataset:** 20-row sales records with 5 columns in wildly inconsistent formats.

| Column | Example inputs | Target |
|--------|---------------|--------|
| `date` | `"01/15/2023"`, `"Jan 15 2023"`, `"2023-01-15"` | `YYYY-MM-DD` |
| `phone` | `"555-123-4567"`, `"+15551234567"`, `"555.123.4567"` | `(XXX) XXX-XXXX` |
| `state` | `"California"`, `"cal"`, `"CALIFORNIA"` | `CA` (2-letter) |
| `amount` | `"$1,234.56"`, `"USD 1234.56"`, `"1,234.56"` | `1234.56` (float) |
| `product_code` | `"sku001"`, `"SKU 001"`, `"001"` | `SKU-001` |

**Action:** `apply_transforms` — submit a `ColumnTransform` per column.
The environment applies the transforms programmatically and compares results
to ground-truth expected values.

**Reward:** Mean per-column accuracy (fraction of rows whose transformed value
matches ground truth).

---

### Task 3 – Multi-step Data Quality Pipeline *(hard)*

**Dataset:** 25-row employee records with 10 issues across 6 categories.

The agent must complete **4 pipeline phases in order**:

```
AUDIT → IDENTIFY → FIX → VALIDATE
```

| Phase | Action type | What to do |
|-------|------------|------------|
| audit | `audit` | Identify which *categories* of issues are present |
| identify | `report_issues` | Enumerate every specific issue (row, column, type) |
| fix | `fix` | Submit `fix_operations` to correct each issue |
| validate | `validate` | Write a validation report; estimate remaining issues |

**Reward (weighted sum):**
- Audit: 0.15 × category recall
- Identify: 0.30 × issue F1
- Fix: 0.40 × fix recall (−0.05 per spurious fix)
- Validate: 0.15 × report quality
- +0.05 efficiency bonus for exactly 4 phases (no wasted steps)

---

## Action & Observation Spaces

### Observation – `DataObservation`

```python
class DataObservation(BaseModel):
    episode_id:       str
    task_type:        TaskType          # schema_validation | standardization | pipeline
    dataset_name:     str
    dataset_sample:   List[Dict]        # first N rows of the dataset
    total_rows:       int
    schema:           Dict              # JSON schema (validation & pipeline tasks)
    column_stats:     Dict[str, ColumnStats]  # per-column statistics
    current_step:     int
    max_steps:        int
    pipeline_phase:   Optional[PipelineStep]  # pipeline task only
    available_actions: List[str]
    cumulative_reward: float
    context:          Dict              # target formats, hints, etc.
```

### Action – `DataAction`

```python
class DataAction(BaseModel):
    action_type: str  # report_issues | apply_transforms | audit | fix | validate

    # report_issues
    issues: Optional[List[DataIssue]]

    # apply_transforms
    transforms: Optional[Dict[str, ColumnTransform]]

    # audit
    audit_summary:    Optional[str]
    issue_categories: Optional[List[str]]

    # fix
    fix_operations: Optional[List[FixOperation]]

    # validate
    validation_report:  Optional[str]
    issues_remaining:   Optional[int]
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + env metadata |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start a new episode `{task_id}` |
| POST | `/step` | Advance episode `{episode_id, action}` |
| GET | `/state/{episode_id}` | Inspect episode state |
| POST | `/validate` | Run openenv validation check |
| GET | `/docs` | Interactive Swagger UI |

---

## Quick Start

### Local Python

```bash
pip install -r requirements.txt

# Start the environment server
python app.py          # listens on http://localhost:7860

# In another terminal, run the baseline agent
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
HF_TOKEN="hf_..." \
python inference.py
```

### Docker

```bash
# Build
docker build -t data-cleaning-env .

# Run the server
docker run -p 7860:7860 data-cleaning-env

# Run inference (in a second terminal)
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
HF_TOKEN="hf_..." \
OPENENV_BASE_URL="http://localhost:7860" \
python inference.py
```

### cURL example

```bash
# Reset a schema_validation episode
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"schema_validation"}' | python -m json.tool

# Step with a report_issues action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id": "<id from reset>",
    "action": {
      "action_type": "report_issues",
      "issues": [
        {"row_index": 2, "column": "email", "issue_type": "invalid_format",
         "description": "missing @ symbol", "value": "invalid-email"}
      ]
    }
  }' | python -m json.tool
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Yes | Model identifier (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `HF_TOKEN` | Yes* | Hugging Face token (used as API key) |
| `OPENAI_API_KEY` | Yes* | OpenAI API key (alternative to `HF_TOKEN`) |
| `OPENENV_BASE_URL` | No | Environment server URL (default `http://localhost:7860`) |

\* Either `HF_TOKEN` or `OPENAI_API_KEY` must be set. `HF_TOKEN` takes precedence if both are present.

---

## Baseline Scores

Approximate scores with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score |
|------|-------|
| Schema Validation (easy) | ~0.72 |
| Standardization (medium) | ~0.65 |
| Pipeline (hard) | ~0.48 |
| **Overall average** | **~0.62** |

These scores represent a reasonable but improvable baseline — the environment
has plenty of headroom for RL fine-tuning.

---

## Project Structure

```
data-cleaning-env/
├── app.py            # FastAPI server (OpenEnv HTTP API)
├── inference.py      # Baseline inference script ([START]/[STEP]/[END] logging)
├── validate.py       # Pre-submission validation script (run before submitting)
├── openenv.yaml      # Environment metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── env/
    ├── __init__.py
    ├── environment.py  # Core DataCleaningEnv class
    ├── models.py       # Pydantic typed models
    ├── datasets.py     # Synthetic datasets with known ground-truth issues
    └── graders.py      # Deterministic grading functions
```

---

## Grader Details

All graders are **fully deterministic** — same action always produces the same score.

### Schema Validation Grader
- Builds sets of `(row_index, column, issue_type)` tuples for predicted vs ground truth
- Computes F1 over exact matches
- Adds 0.3× partial credit for correct `(row, column)` with wrong `issue_type`
- Applies −15% penalty if reported count > 2× ground truth (penalises spam)

### Standardization Grader
- For each of the 5 columns, applies the agent's transform to every raw value
- Compares result to a deterministic ground-truth expected value
- Column score = (matching rows) / total rows
- Final score = mean column score

### Pipeline Grader
- Audit: recall over the 5 known issue categories (`invalid_enum`, `invalid_range`, `duplicate`, `invalid_format`, `missing_required`)
- Identify: F1 with partial credit (reuses schema validation grader)
- Fix: recall over `(row, column)` pairs; −0.05 per spurious fix (max −0.30)
- Validate: rubric scoring (report length, keyword presence, consistency)
- Episode score: weighted average + efficiency bonus

---

## Pre-submission Validation

Run before deploying to confirm all checks pass:

```bash
# Static checks only (no server needed)
python validate.py

# Full check including live endpoint tests (server must be running)
OPENENV_BASE_URL=http://localhost:7860 python validate.py --live
```

Checks performed:
- All required files are present
- `openenv.yaml` has required fields and 3+ tasks
- Pydantic models are valid
- `DataCleaningEnv` is importable and all 3 tasks work
- `reset()` / `step()` / `state()` API functions correctly
- `inference.py` emits `[START]` / `[STEP]` / `[END]` logs and references all required env vars
- Dockerfile is valid
- All task rewards are in `[0.0, 1.0]`

---

## Reward Signal Design

The environment is designed to give the agent **partial credit at every step**, not just binary win/lose:

- **Schema Validation**: F1 over `(row, column, issue_type)` tuples + 0.3× partial credit for right row/column with wrong type → score varies from 0.0 to 1.0 continuously
- **Standardization**: mean column accuracy → each column contributes 0.2 to the total; missing columns are penalised individually
- **Pipeline**: 4 independent per-phase scores accumulated across the episode; the agent sees a reward signal after each of the 4 steps (audit → identify → fix → validate), enabling credit assignment across the trajectory

---

## License

BSD 3-Clause — see `LICENSE`.

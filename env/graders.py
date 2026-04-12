"""
Deterministic graders for the three tasks.

Each grader returns a RewardBreakdown with:
  - total  in open interval (0, 1) — strictly between 0 and 1
  - components: dict of sub-score names -> values
  - feedback: human-readable explanation
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import DataAction, DataIssue, RewardBreakdown


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _issue_key(issue: Dict[str, Any]) -> tuple:
    """Canonical key for matching predicted vs ground-truth issues."""
    return (issue["row_index"], issue["column"], issue["issue_type"])


def _predicted_key(issue: DataIssue) -> tuple:
    return (issue.row_index, issue.column, issue.issue_type.value)


_STRICT_EPS = 1e-7


def _strict_open_score(value: float) -> float:
    """
    Enforce score in the open interval (0, 1) required by Phase 2.
    score = max(eps, min(1 - eps, original_score))
    No rounding — rounding to 4 decimals would collapse 1e-7 back to 0.0.
    """
    return max(_STRICT_EPS, min(1.0 - _STRICT_EPS, float(value)))


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 – Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

def grade_schema_validation(
    action: DataAction,
    known_issues: List[Dict[str, Any]],
) -> RewardBreakdown:
    """
    Score the agent's list of reported issues against the ground truth.
    Uses F1 over (row_index, column, issue_type) tuples.
    Partial credit: row/column match without correct issue_type -> 0.3 credit.
    """
    if action.action_type != "report_issues" or not action.issues:
        return RewardBreakdown(
            total=_strict_open_score(0.0),
            components={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            feedback="No issues reported. Action must be 'report_issues' with a non-empty 'issues' list.",
        )

    gt_keys   = {_issue_key(i) for i in known_issues}
    pred_keys = {_predicted_key(i) for i in action.issues}

    # Exact matches
    exact_tp = len(gt_keys & pred_keys)

    # Partial match: correct row + column but wrong issue_type -> 0.3 credit each
    gt_row_col   = {(i["row_index"], i["column"]) for i in known_issues}
    pred_row_col = {(i.row_index, i.column) for i in action.issues}
    partial_only = (gt_row_col & pred_row_col) - {(k[0], k[1]) for k in (gt_keys & pred_keys)}
    partial_credit = len(partial_only) * 0.3

    effective_tp = exact_tp + partial_credit

    precision = effective_tp / len(pred_keys) if pred_keys else 0.0
    recall    = effective_tp / len(gt_keys)   if gt_keys   else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    total = _strict_open_score(f1)

    # Penalty for excessive false positives (> 2x ground truth count)
    if len(pred_keys) > 2 * len(gt_keys):
        total = _strict_open_score(total * 0.85)

    feedback_parts = [
        f"Ground truth: {len(gt_keys)} issues.",
        f"Reported: {len(pred_keys)} issues.",
        f"Exact matches: {exact_tp}, partial (row+col): {len(partial_only)}.",
        f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}.",
    ]
    missed = gt_keys - pred_keys
    if missed:
        examples = list(missed)[:3]
        feedback_parts.append(f"Sample missed issues: {examples}")

    return RewardBreakdown(
        total=total,
        components={
            "exact_tp": exact_tp,
            "partial_credit": round(partial_credit, 3),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        },
        feedback=" ".join(feedback_parts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 – Standardization
# ─────────────────────────────────────────────────────────────────────────────

# State name -> two-letter code mapping
_STATE_MAP: Dict[str, str] = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
    "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
    "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA",
    "kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD",
    "massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
    "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV",
    "new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY",
    "north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC",
    "south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT",
    "vermont":"VT","virginia":"VA","washington":"WA","west virginia":"WV",
    "wisconsin":"WI","wyoming":"WY",
}

def _apply_date_transform(value: str, _transform) -> Optional[str]:
    """Try to apply the agent's date transform and return ISO string or None.
    US-convention: MM-DD-YYYY takes priority over DD-MM-YYYY for hyphenated dates.
    """
    from datetime import datetime
    # %m-%d-%Y must come before %d-%m-%Y to correctly parse US dates like "07-04-2023"
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%b %d %Y", "%B %d %Y",
               "%m/%d/%y", "%m-%d-%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _apply_phone_transform(value: str, _transform) -> Optional[str]:
    """Normalize phone to (XXX) XXX-XXXX."""
    digits = re.sub(r"\D", "", value)
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return None


def _apply_state_transform(value: str, _transform) -> Optional[str]:
    v = value.strip()
    if len(v) == 2 and v.upper() in _STATE_MAP.values():
        return v.upper()
    abbrev = {"Calif.": "CA", "calif.": "CA"}
    if v in abbrev:
        return abbrev[v]
    return _STATE_MAP.get(v.lower())


def _apply_amount_transform(value: str, _transform) -> Optional[float]:
    cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _apply_sku_transform(value: str, _transform) -> Optional[str]:
    digits = re.sub(r"\D", "", value)
    if digits:
        return f"SKU-{int(digits):03d}"
    return None


_TRANSFORM_FNS = {
    "date":         _apply_date_transform,
    "phone":        _apply_phone_transform,
    "state":        _apply_state_transform,
    "amount":       _apply_amount_transform,
    "product_code": _apply_sku_transform,
}


def grade_standardization(
    action: DataAction,
    rows: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Any]],
) -> RewardBreakdown:
    """
    Apply the agent's transforms against the raw data and compare to ground truth.

    Scoring per column (equal weight):
      - For each row: 1 point if transform produces correct output, 0 otherwise.
      - Column score = (correct rows) / total_rows
    Final score = average column score.
    """
    if action.action_type != "apply_transforms" or not action.transforms:
        return RewardBreakdown(
            total=_strict_open_score(0.0),
            components={},
            feedback="Action must be 'apply_transforms' with a non-empty 'transforms' dict.",
        )

    columns_to_grade = list(ground_truth.keys())
    n = len(rows)
    col_scores: Dict[str, float] = {}

    for col in columns_to_grade:
        transform = action.transforms.get(col)
        if transform is None:
            col_scores[col] = 0.0
            continue

        fn = _TRANSFORM_FNS.get(col)
        if fn is None:
            col_scores[col] = 0.0
            continue

        correct = 0
        gt_values = ground_truth[col]
        for i, row in enumerate(rows):
            raw = row.get(col)
            if raw is None:
                continue
            result = fn(str(raw), transform)
            expected = gt_values[i]
            if result == expected:
                correct += 1

        col_scores[col] = correct / n

    total = _strict_open_score(sum(col_scores.values()) / len(col_scores)) if col_scores else _strict_open_score(0.0)

    feedback_parts = []
    for col, score in col_scores.items():
        feedback_parts.append(f"{col}: {score:.0%}")
    feedback = "Column scores -- " + ", ".join(feedback_parts)

    return RewardBreakdown(
        total=total,
        components={c: round(s, 4) for c, s in col_scores.items()},
        feedback=feedback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 – Pipeline (multi-step)
# ─────────────────────────────────────────────────────────────────────────────

def grade_pipeline_audit(
    action: DataAction,
    known_categories: List[str],
) -> RewardBreakdown:
    """
    Audit step: did the agent identify the right issue categories?
    Score = (# correct categories mentioned) / (total categories present).
    """
    if action.action_type != "audit":
        return RewardBreakdown(
            total=_strict_open_score(0.0),
            components={"categories_found": 0},
            feedback="Action must be 'audit' for the audit phase.",
        )

    text = " ".join([
        action.audit_summary or "",
        " ".join(action.issue_categories or []),
    ]).lower()

    found = 0
    matched = []
    for cat in known_categories:
        if cat.replace("_", " ") in text or cat in text:
            found += 1
            matched.append(cat)

    score = _strict_open_score(found / len(known_categories)) if known_categories else _strict_open_score(0.0)

    return RewardBreakdown(
        total=score,
        components={"categories_found": found, "total_categories": len(known_categories)},
        feedback=(
            f"Found {found}/{len(known_categories)} issue categories: {matched}. "
            f"Missing: {[c for c in known_categories if c not in matched]}"
        ),
    )


def grade_pipeline_identify(
    action: DataAction,
    known_issues: List[Dict[str, Any]],
) -> RewardBreakdown:
    """
    Identify step: uses the same F1-based grader as schema_validation.
    """
    return grade_schema_validation(action, known_issues)


def grade_pipeline_fix(
    action: DataAction,
    known_issues: List[Dict[str, Any]],
) -> RewardBreakdown:
    """
    Fix step: how many of the known issues were addressed by fix_operations?

    A fix_operation addresses an issue if:
      - row_index and column match a known issue
    Score = matched fixes / total known issues (capped at 1.0).
    Bonus: -0.05 per spurious fix (fix with no matching issue), up to -0.3.
    """
    if action.action_type != "fix" or not action.fix_operations:
        return RewardBreakdown(
            total=_strict_open_score(0.0),
            components={"addressed": 0},
            feedback="Action must be 'fix' with non-empty 'fix_operations'.",
        )

    issue_locs = {(i["row_index"], i["column"]) for i in known_issues}
    addressed = set()
    spurious  = 0

    for fix in action.fix_operations:
        loc = (fix.row_index, fix.column)
        if loc in issue_locs:
            addressed.add(loc)
        else:
            spurious += 1

    recall = len(addressed) / len(known_issues) if known_issues else 0.0
    penalty = min(spurious * 0.05, 0.30)
    total = _strict_open_score(max(recall - penalty, 0.0))

    return RewardBreakdown(
        total=total,
        components={
            "addressed": len(addressed),
            "total_issues": len(known_issues),
            "spurious_fixes": spurious,
            "recall": round(recall, 4),
            "penalty": round(penalty, 4),
        },
        feedback=(
            f"Addressed {len(addressed)}/{len(known_issues)} issues. "
            f"{spurious} spurious fix(es) (-{penalty:.2f} penalty). "
            f"Final: {total:.4f}"
        ),
    )


def grade_pipeline_validate(
    action: DataAction,
    issues_fixed_count: int,
    total_issues: int,
) -> RewardBreakdown:
    """
    Validate step: did the agent produce a coherent validation report?

    Checks:
      - Report is non-empty (min 30 chars)
      - Mentions remaining issues or confirms clean data
      - Consistency: if issues_remaining <= (total - issues_fixed), give full credit
    """
    if action.action_type != "validate":
        return RewardBreakdown(
            total=_strict_open_score(0.0),
            components={},
            feedback="Action must be 'validate' for the validate phase.",
        )

    report = (action.validation_report or "").strip()

    length_ok = len(report) >= 30
    mentions_issues = any(
        word in report.lower()
        for word in ["issue", "error", "problem", "clean", "valid", "remain", "fix"]
    )
    remaining_provided = action.issues_remaining is not None
    expected_remaining = total_issues - issues_fixed_count

    consistency_ok = False
    if remaining_provided:
        # Accept if agent's estimate is within 2 of actual remaining
        consistency_ok = abs((action.issues_remaining or 0) - expected_remaining) <= 2

    score = 0.0
    if length_ok:           score += 0.4
    if mentions_issues:     score += 0.3
    if remaining_provided:  score += 0.15
    if consistency_ok:      score += 0.15

    return RewardBreakdown(
        total=_strict_open_score(score),
        components={
            "length_ok":          length_ok,
            "mentions_issues":    mentions_issues,
            "remaining_provided": remaining_provided,
            "consistency_ok":     consistency_ok,
        },
        feedback=(
            f"Report length: {len(report)} chars. "
            f"Length OK: {length_ok}, mentions issues: {mentions_issues}, "
            f"provided remaining count: {remaining_provided}, "
            f"consistent estimate: {consistency_ok}. "
            f"Expected remaining ~{expected_remaining}."
        ),
    )


def grade_pipeline_episode(step_scores: Dict[str, float]) -> float:
    """
    Combine per-phase scores into a final episode score.

    Weights:
      audit    -> 0.15
      identify -> 0.30
      fix      -> 0.40
      validate -> 0.15
    Efficiency bonus: if all 4 phases completed (no extra steps), +0.05 (capped at 1.0).
    """
    weights = {"audit": 0.15, "identify": 0.30, "fix": 0.40, "validate": 0.15}
    total = sum(weights[phase] * step_scores.get(phase, 0.0) for phase in weights)

    if len(step_scores) == 4:   # exactly the 4 required phases, no wasted steps
        total += 0.05

    return _strict_open_score(total)

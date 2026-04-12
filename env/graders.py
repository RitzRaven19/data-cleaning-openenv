"""
Deterministic graders for the three tasks.

All numeric outputs — total and every component value — are clamped through
safe_score() which guarantees the strict open interval (0.1, 0.99).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import DataAction, DataIssue, RewardBreakdown


# ─────────────────────────────────────────────────────────────────────────────
# Global safe clamp — applied to EVERY numeric output without exception
# ─────────────────────────────────────────────────────────────────────────────

def safe_score(x) -> float:
    """Clamp any value to the strict open interval (0.1, 0.99)."""
    return max(0.1, min(0.99, float(x)))


def _safe_components(d: Dict[str, Any]) -> Dict[str, float]:
    """Pass every value in a components dict through safe_score()."""
    return {k: safe_score(v) for k, v in d.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _issue_key(issue: Dict[str, Any]) -> tuple:
    return (issue["row_index"], issue["column"], issue["issue_type"])


def _predicted_key(issue: DataIssue) -> tuple:
    return (issue.row_index, issue.column, issue.issue_type.value)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 – Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

def grade_schema_validation(
    action: DataAction,
    known_issues: List[Dict[str, Any]],
) -> RewardBreakdown:
    if action.action_type != "report_issues" or not action.issues:
        return RewardBreakdown(
            total=safe_score(0),
            components=_safe_components({"precision": 0, "recall": 0, "f1": 0}),
            feedback="No issues reported.",
        )

    gt_keys   = {_issue_key(i) for i in known_issues}
    pred_keys = {_predicted_key(i) for i in action.issues}

    # Exact matches — use set intersection (no double counting)
    exact_matches = gt_keys & pred_keys
    exact_tp = len(exact_matches)

    # Partial: correct row+column but wrong issue_type — exclude already exact-matched locations
    gt_row_col   = {(i["row_index"], i["column"]) for i in known_issues}
    pred_row_col = {(i.row_index,    i.column)    for i in action.issues}
    exact_locs   = {(k[0], k[1]) for k in exact_matches}
    partial_locs = (gt_row_col & pred_row_col) - exact_locs   # no overlap with exact
    partial_credit = len(partial_locs) * 0.3

    effective_tp = exact_tp + partial_credit

    precision = effective_tp / len(pred_keys) if pred_keys else 0.0
    recall    = effective_tp / len(gt_keys)   if gt_keys   else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    raw_total = f1
    if len(pred_keys) > 2 * len(gt_keys):
        raw_total = raw_total * 0.85

    gt_n = max(len(gt_keys), 1)

    missed = gt_keys - pred_keys
    feedback = (
        f"GT: {len(gt_keys)}, reported: {len(pred_keys)}, "
        f"exact: {exact_tp}, partial: {len(partial_locs)}. "
        f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}."
        + (f" Missed sample: {list(missed)[:3]}" if missed else "")
    )

    return RewardBreakdown(
        total=safe_score(raw_total),
        components=_safe_components({
            "exact_tp_ratio":       exact_tp / gt_n,
            "partial_credit_ratio": partial_credit / gt_n,
            "precision":            precision,
            "recall":               recall,
            "f1":                   f1,
        }),
        feedback=feedback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 – Standardization
# ─────────────────────────────────────────────────────────────────────────────

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

def _apply_date_transform(value: str, _t) -> Optional[str]:
    from datetime import datetime
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%b %d %Y", "%B %d %Y",
                "%m/%d/%y", "%m-%d-%Y", "%d-%m-%Y"]:
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def _apply_phone_transform(value: str, _t) -> Optional[str]:
    digits = re.sub(r"\D", "", value)
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return None

def _apply_state_transform(value: str, _t) -> Optional[str]:
    v = value.strip()
    if len(v) == 2 and v.upper() in _STATE_MAP.values():
        return v.upper()
    return {"Calif.": "CA", "calif.": "CA"}.get(v) or _STATE_MAP.get(v.lower())

def _apply_amount_transform(value: str, _t) -> Optional[float]:
    cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None

def _apply_sku_transform(value: str, _t) -> Optional[str]:
    digits = re.sub(r"\D", "", value)
    return f"SKU-{int(digits):03d}" if digits else None

_TRANSFORM_FNS = {
    "date": _apply_date_transform, "phone": _apply_phone_transform,
    "state": _apply_state_transform, "amount": _apply_amount_transform,
    "product_code": _apply_sku_transform,
}

def grade_standardization(
    action: DataAction,
    rows: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Any]],
) -> RewardBreakdown:
    if action.action_type != "apply_transforms" or not action.transforms:
        return RewardBreakdown(
            total=safe_score(0),
            components={},
            feedback="Action must be 'apply_transforms' with non-empty transforms.",
        )

    n = len(rows)
    col_scores: Dict[str, float] = {}

    for col in ground_truth:
        transform = action.transforms.get(col)
        fn = _TRANSFORM_FNS.get(col)
        if transform is None or fn is None:
            col_scores[col] = safe_score(0)
            continue
        correct = sum(
            1 for i, row in enumerate(rows)
            if row.get(col) is not None
            and fn(str(row[col]), transform) == ground_truth[col][i]
        )
        col_scores[col] = safe_score(correct / n if n else 0)

    avg = sum(col_scores.values()) / len(col_scores) if col_scores else 0
    feedback = "Column scores -- " + ", ".join(f"{c}: {s:.2%}" for c, s in col_scores.items())

    return RewardBreakdown(
        total=safe_score(avg),
        components=dict(col_scores),   # already safe_score'd individually
        feedback=feedback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 – Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def grade_pipeline_audit(
    action: DataAction,
    known_categories: List[str],
) -> RewardBreakdown:
    if action.action_type != "audit":
        return RewardBreakdown(
            total=safe_score(0),
            components=_safe_components({"categories_found_ratio": 0}),
            feedback="Action must be 'audit'.",
        )

    text = " ".join([
        action.audit_summary or "",
        " ".join(action.issue_categories or []),
    ]).lower()

    matched = [c for c in known_categories if c.replace("_", " ") in text or c in text]
    cat_n = max(len(known_categories), 1)

    return RewardBreakdown(
        total=safe_score(len(matched) / cat_n),
        components=_safe_components({"categories_found_ratio": len(matched) / cat_n}),
        feedback=(
            f"Found {len(matched)}/{len(known_categories)}: {matched}. "
            f"Missing: {[c for c in known_categories if c not in matched]}"
        ),
    )


def grade_pipeline_identify(
    action: DataAction,
    known_issues: List[Dict[str, Any]],
) -> RewardBreakdown:
    return grade_schema_validation(action, known_issues)


def grade_pipeline_fix(
    action: DataAction,
    known_issues: List[Dict[str, Any]],
) -> RewardBreakdown:
    if action.action_type != "fix" or not action.fix_operations:
        return RewardBreakdown(
            total=safe_score(0),
            components=_safe_components({"addressed_ratio": 0}),
            feedback="Action must be 'fix' with non-empty fix_operations.",
        )

    issue_locs = {(i["row_index"], i["column"]) for i in known_issues}
    addressed  = set()   # set prevents double-counting same location
    spurious   = 0

    for fix in action.fix_operations:
        loc = (fix.row_index, fix.column)
        if loc in issue_locs:
            addressed.add(loc)
        else:
            spurious += 1

    issue_n  = max(len(known_issues), 1)
    fix_n    = max(len(action.fix_operations), 1)
    recall   = len(addressed) / issue_n
    penalty  = min(spurious * 0.05, 0.30)
    raw      = max(recall - penalty, 0.0)

    return RewardBreakdown(
        total=safe_score(raw),
        components=_safe_components({
            "addressed_ratio": len(addressed) / issue_n,
            "spurious_rate":   spurious / fix_n,
            "recall":          recall,
            "penalty":         penalty,
        }),
        feedback=(
            f"Addressed {len(addressed)}/{len(known_issues)} issues. "
            f"{spurious} spurious (-{penalty:.2f}). Final: {safe_score(raw):.4f}"
        ),
    )


def grade_pipeline_validate(
    action: DataAction,
    issues_fixed_count: int,
    total_issues: int,
) -> RewardBreakdown:
    if action.action_type != "validate":
        return RewardBreakdown(
            total=safe_score(0),
            components={},
            feedback="Action must be 'validate'.",
        )

    report = (action.validation_report or "").strip()

    length_ok        = len(report) >= 30
    mentions_issues  = any(w in report.lower()
                           for w in ["issue", "error", "problem", "clean", "valid", "remain", "fix"])
    remaining_provided = action.issues_remaining is not None
    expected_remaining = total_issues - issues_fixed_count
    consistency_ok   = (remaining_provided and
                        abs((action.issues_remaining or 0) - expected_remaining) <= 2)

    score = (0.4 * float(length_ok) + 0.3 * float(mentions_issues)
             + 0.15 * float(remaining_provided) + 0.15 * float(consistency_ok))

    return RewardBreakdown(
        total=safe_score(score),
        components=_safe_components({
            "length_ok":          float(length_ok),
            "mentions_issues":    float(mentions_issues),
            "remaining_provided": float(remaining_provided),
            "consistency_ok":     float(consistency_ok),
        }),
        feedback=(
            f"Report {len(report)} chars. "
            f"length_ok={length_ok}, mentions={mentions_issues}, "
            f"remaining_provided={remaining_provided}, consistent={consistency_ok}. "
            f"Expected remaining ~{expected_remaining}."
        ),
    )


def grade_pipeline_episode(step_scores: Dict[str, float]) -> float:
    weights = {"audit": 0.15, "identify": 0.30, "fix": 0.40, "validate": 0.15}
    # Clamp each input phase score before weighting
    total = sum(weights[p] * safe_score(step_scores.get(p, 0)) for p in weights)
    if len(step_scores) == 4:
        total += 0.05
    return safe_score(total)

"""
Exhaustive bounds test for the grading system.
Verifies: 0.1 < every numeric score < 0.99
"""
import sys
sys.path.insert(0, ".")

from env.graders import (
    grade_schema_validation, grade_standardization,
    grade_pipeline_audit, grade_pipeline_identify,
    grade_pipeline_fix, grade_pipeline_validate,
    grade_pipeline_episode, safe_score,
)
from env.models import (
    DataAction, DataIssue, FixOperation, ColumnTransform, IssueType,
)

FAILURES = []

# ─── validators ───────────────────────────────────────────────────────────────

def chk(val, ctx):
    try:
        f = float(val)
        if not (0.1 < f < 0.99):
            FAILURES.append(f"  FAIL  [{ctx}] = {val!r}")
    except Exception as e:
        FAILURES.append(f"  ERROR [{ctx}] = {val!r}  ({e})")

def chk_bd(bd, label):
    chk(bd.total, f"{label}.total")
    for k, v in bd.components.items():
        chk(v, f"{label}.components.{k}")


# ─── fixtures ─────────────────────────────────────────────────────────────────

KNOWN = [
    {"row_index": 0, "column": "email",  "issue_type": "invalid_format"},
    {"row_index": 2, "column": "age",    "issue_type": "invalid_range"},
    {"row_index": 5, "column": "status", "issue_type": "invalid_enum"},
]

ROWS = [{"date": "01/15/2023", "phone": "5551234567",
         "state": "California", "amount": "1234.56", "product_code": "sku001"}]
GT   = {
    "date":         ["2023-01-15"],
    "phone":        ["(555) 123-4567"],
    "state":        ["CA"],
    "amount":       [1234.56],
    "product_code": ["SKU-001"],
}
CATS = ["missing_required", "invalid_range", "invalid_format", "duplicate", "invalid_enum"]


# ─── schema_validation ────────────────────────────────────────────────────────

# empty issues list (falsy)
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[]), KNOWN), "schema/empty_list")

# wrong action type
chk_bd(grade_schema_validation(
    DataAction(action_type="audit"), KNOWN), "schema/wrong_action")

# zero matches
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[
        DataIssue(row_index=99, column="x", issue_type=IssueType.DUPLICATE, description="")
    ]), KNOWN), "schema/no_match")

# partial match only (right row+col, wrong issue_type)
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[
        DataIssue(row_index=i["row_index"], column=i["column"],
                  issue_type=IssueType.DUPLICATE, description="") for i in KNOWN
    ]), KNOWN), "schema/partial_only")

# perfect match
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[
        DataIssue(row_index=i["row_index"], column=i["column"],
                  issue_type=IssueType(i["issue_type"]), description="") for i in KNOWN
    ]), KNOWN), "schema/perfect")

# spam (> 2x ground truth count)
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[
        DataIssue(row_index=r, column="email",
                  issue_type=IssueType.DUPLICATE, description="") for r in range(20)
    ]), KNOWN), "schema/spam")

# mixed exact + partial
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[
        DataIssue(row_index=0, column="email",
                  issue_type=IssueType.INVALID_FORMAT, description=""),  # exact
        DataIssue(row_index=2, column="age",
                  issue_type=IssueType.DUPLICATE, description=""),        # partial
    ]), KNOWN), "schema/mixed")

# single known issue
chk_bd(grade_schema_validation(
    DataAction(action_type="report_issues", issues=[
        DataIssue(row_index=0, column="email",
                  issue_type=IssueType.INVALID_FORMAT, description="")
    ]), [{"row_index": 0, "column": "email", "issue_type": "invalid_format"}]),
    "schema/single_perfect")


# ─── standardization ──────────────────────────────────────────────────────────

# wrong action
chk_bd(grade_standardization(DataAction(action_type="audit"), ROWS, GT), "std/wrong_action")

# empty transforms dict (falsy)
chk_bd(grade_standardization(
    DataAction(action_type="apply_transforms", transforms={}), ROWS, GT), "std/empty_transforms")

# all columns missing from transforms
chk_bd(grade_standardization(
    DataAction(action_type="apply_transforms", transforms={
        "irrelevant": ColumnTransform(target_format="x")
    }), ROWS, GT), "std/missing_cols")

# all columns correct
chk_bd(grade_standardization(
    DataAction(action_type="apply_transforms", transforms={
        col: ColumnTransform(target_format="x") for col in GT
    }), ROWS, GT), "std/all_correct")

# all columns wrong (transform present but produces wrong result)
chk_bd(grade_standardization(
    DataAction(action_type="apply_transforms", transforms={
        col: ColumnTransform(target_format="wrong") for col in GT
    }), ROWS, GT), "std/all_wrong")


# ─── pipeline audit ───────────────────────────────────────────────────────────

# wrong action
chk_bd(grade_pipeline_audit(
    DataAction(action_type="report_issues"), CATS), "audit/wrong_action")

# empty text (finds nothing)
chk_bd(grade_pipeline_audit(
    DataAction(action_type="audit", audit_summary="", issue_categories=[]), CATS), "audit/empty")

# partial match
chk_bd(grade_pipeline_audit(
    DataAction(action_type="audit",
               audit_summary="missing_required invalid_range",
               issue_categories=[]), CATS), "audit/partial")

# perfect via audit_summary
chk_bd(grade_pipeline_audit(
    DataAction(action_type="audit",
               audit_summary=" ".join(CATS), issue_categories=[]), CATS), "audit/perfect_summary")

# perfect via issue_categories list
chk_bd(grade_pipeline_audit(
    DataAction(action_type="audit", audit_summary="", issue_categories=CATS), CATS),
    "audit/perfect_categories")


# ─── pipeline fix ─────────────────────────────────────────────────────────────

# wrong action
chk_bd(grade_pipeline_fix(
    DataAction(action_type="validate"), KNOWN), "fix/wrong_action")

# empty fix_operations (falsy)
chk_bd(grade_pipeline_fix(
    DataAction(action_type="fix", fix_operations=[]), KNOWN), "fix/empty")

# all spurious (no matching locations)
chk_bd(grade_pipeline_fix(
    DataAction(action_type="fix", fix_operations=[
        FixOperation(row_index=99, column="x", old_value="a", new_value="b")
        for _ in range(10)
    ]), KNOWN), "fix/all_spurious")

# perfect (all issues addressed, no spurious)
chk_bd(grade_pipeline_fix(
    DataAction(action_type="fix", fix_operations=[
        FixOperation(row_index=i["row_index"], column=i["column"],
                     old_value="a", new_value="b") for i in KNOWN
    ]), KNOWN), "fix/perfect")

# duplicate fix operations for same location (set prevents double count)
chk_bd(grade_pipeline_fix(
    DataAction(action_type="fix", fix_operations=[
        FixOperation(row_index=0, column="email", old_value="a", new_value="b"),
        FixOperation(row_index=0, column="email", old_value="a", new_value="c"),  # duplicate
    ]), KNOWN), "fix/duplicate_ops")

# partial fix + some spurious
chk_bd(grade_pipeline_fix(
    DataAction(action_type="fix", fix_operations=[
        FixOperation(row_index=0, column="email", old_value="a", new_value="b"),  # valid
        FixOperation(row_index=99, column="x",    old_value="a", new_value="b"),  # spurious
    ]), KNOWN), "fix/partial_spurious")


# ─── pipeline validate ────────────────────────────────────────────────────────

# wrong action
chk_bd(grade_pipeline_validate(
    DataAction(action_type="audit"), 3, 5), "validate/wrong_action")

# empty report, no remaining provided  -> score=0 -> safe_score(0)=0.1
chk_bd(grade_pipeline_validate(
    DataAction(action_type="validate", validation_report="", issues_remaining=None),
    3, 5), "validate/all_false")

# only length_ok fails (short report)
chk_bd(grade_pipeline_validate(
    DataAction(action_type="validate", validation_report="issues fixed", issues_remaining=2),
    3, 5), "validate/short_report")

# all criteria met
chk_bd(grade_pipeline_validate(
    DataAction(action_type="validate",
               validation_report="All issues have been identified and fixed. No remaining errors.",
               issues_remaining=2),
    3, 5), "validate/perfect")

# remaining provided but wrong estimate (consistency_ok=False)
chk_bd(grade_pipeline_validate(
    DataAction(action_type="validate",
               validation_report="Some issues remain in the dataset after cleaning.",
               issues_remaining=100),
    3, 5), "validate/bad_estimate")


# ─── episode aggregation ──────────────────────────────────────────────────────

for label, scores in [
    ("all_zero",    {}),
    ("all_min",     {"audit": 0.1,  "identify": 0.1,  "fix": 0.1,  "validate": 0.1}),
    ("all_max",     {"audit": 0.99, "identify": 0.99, "fix": 0.99, "validate": 0.99}),
    ("realistic",   {"audit": 0.5,  "identify": 0.7,  "fix": 0.8,  "validate": 0.6}),
    ("partial",     {"audit": 0.5,  "identify": 0.3}),
]:
    chk(grade_pipeline_episode(scores), f"episode/{label}")


# ─── safe_score itself ────────────────────────────────────────────────────────

for val in [0, 0.0, 1, 1.0, -1, 2, 0.5, 0.1, 0.99, 1e-7, 1 - 1e-7, float("inf")]:
    chk(safe_score(val), f"safe_score({val})")


# ─── log functions ────────────────────────────────────────────────────────────

import io, json
sys.path.insert(0, ".")

# Monkey-patch print to capture log output
captured = []
_orig_print = print

def _cap_print(*args, **kwargs):
    captured.append(" ".join(str(a) for a in args))

import builtins
builtins.print = _cap_print

from inference import log_step, log_end, safe_score as inf_safe

for reward_val in [0.0, 1.0, 1e-7, 0.9999999, 0.5, 0.1, 0.99]:
    log_step(1, "test_action", reward_val, False, reward_val)
    log_end("test", reward_val, 1, True)

builtins.print = _orig_print

for line in captured:
    for prefix in ("[STEP]", "[END]"):
        if line.startswith(prefix):
            payload = json.loads(line[len(prefix):].strip())
            for field in ("reward", "cumulative_reward", "total_reward"):
                if field in payload:
                    chk(payload[field], f"log/{prefix}/{field}={payload[field]}")


# ─── results ──────────────────────────────────────────────────────────────────

total_checks = 0  # counted implicitly
if FAILURES:
    print(f"FOUND {len(FAILURES)} BOUND VIOLATIONS:")
    for f in FAILURES:
        print(f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — no bound violations found.")

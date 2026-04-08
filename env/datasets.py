"""
Embedded datasets for the three tasks.

All datasets are generated programmatically so that:
  - Ground-truth issues / transforms are always known
  - Graders can be fully deterministic
  - No external files are required
"""

from __future__ import annotations
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1  –  Schema Validation  (Easy)
# Dataset: 30-row customer CRM export with 13 known issues
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_VALIDATION_SCHEMA: Dict[str, Any] = {
    "id":         {"type": "integer",   "required": True,  "unique": True},
    "name":       {"type": "string",    "required": True,  "min_length": 2},
    "email":      {"type": "email",     "required": True},
    "phone":      {"type": "phone_us",  "required": False},
    "age":        {"type": "integer",   "required": True,  "min": 18, "max": 120},
    "status":     {"type": "enum",      "required": True,
                   "values": ["active", "inactive", "pending"]},
    "created_at": {"type": "date_iso",  "required": True},
}

# 30 rows; issues are explicitly annotated for the grader.
# Each issue dict: {row_index (0-based), column, issue_type, description}
SCHEMA_VALIDATION_ISSUES: List[Dict[str, Any]] = [
    {"row_index": 2,  "column": "email",      "issue_type": "invalid_format",
     "description": "Email 'invalid-email' is missing '@' symbol"},
    {"row_index": 6,  "column": "age",        "issue_type": "invalid_range",
     "description": "Age -5 is below minimum 18"},
    {"row_index": 9,  "column": "id",         "issue_type": "duplicate",
     "description": "ID 2 appears more than once (also row 1)"},
    {"row_index": 11, "column": "email",      "issue_type": "missing_required",
     "description": "Required field 'email' is null"},
    {"row_index": 14, "column": "status",     "issue_type": "invalid_enum",
     "description": "Status 'banned' is not in [active, inactive, pending]"},
    {"row_index": 17, "column": "age",        "issue_type": "invalid_range",
     "description": "Age 150 exceeds maximum 120"},
    {"row_index": 19, "column": "name",       "issue_type": "invalid_format",
     "description": "Name '' has length < 2 (min_length)"},
    {"row_index": 21, "column": "created_at", "issue_type": "invalid_format",
     "description": "Date '2023-13-01' has invalid month 13"},
    {"row_index": 24, "column": "id",         "issue_type": "duplicate",
     "description": "ID 5 appears more than once (also row 4)"},
    {"row_index": 26, "column": "phone",      "issue_type": "invalid_format",
     "description": "Phone 'abc-def-ghij' is not a valid US phone number"},
    {"row_index": 27, "column": "email",      "issue_type": "invalid_format",
     "description": "Email 'test@' is malformed (missing domain)"},
    {"row_index": 28, "column": "age",        "issue_type": "wrong_type",
     "description": "Age 'twenty' is a string, expected integer"},
    {"row_index": 29, "column": "status",     "issue_type": "missing_required",
     "description": "Required field 'status' is null"},
]

def _make_customer_rows() -> List[Dict[str, Any]]:
    base = [
        {"id": 1,  "name": "Alice Smith",   "email": "alice@example.com",   "phone": "(415) 555-0101", "age": 34, "status": "active",   "created_at": "2022-03-10"},
        {"id": 2,  "name": "Bob Jones",     "email": "bob@example.com",     "phone": "(415) 555-0102", "age": 28, "status": "active",   "created_at": "2022-04-01"},
        {"id": 3,  "name": "Clara Wu",      "email": "invalid-email",       "phone": "(415) 555-0103", "age": 22, "status": "inactive", "created_at": "2022-04-15"},  # issue 0
        {"id": 4,  "name": "David Lee",     "email": "david@example.com",   "phone": "(415) 555-0104", "age": 45, "status": "active",   "created_at": "2022-05-01"},
        {"id": 5,  "name": "Eva Green",     "email": "eva@example.com",     "phone": "(415) 555-0105", "age": 31, "status": "pending",  "created_at": "2022-05-12"},
        {"id": 6,  "name": "Frank Hall",    "email": "frank@example.com",   "phone": "(415) 555-0106", "age": 52, "status": "active",   "created_at": "2022-06-01"},
        {"id": 7,  "name": "Grace Kim",     "email": "grace@example.com",   "phone": "(415) 555-0107", "age": -5, "status": "active",   "created_at": "2022-06-15"},  # issue 1
        {"id": 8,  "name": "Hank Moore",    "email": "hank@example.com",    "phone": "(415) 555-0108", "age": 60, "status": "inactive", "created_at": "2022-07-01"},
        {"id": 9,  "name": "Iris Chen",     "email": "iris@example.com",    "phone": "(415) 555-0109", "age": 27, "status": "active",   "created_at": "2022-07-20"},
        {"id": 2,  "name": "Jack Brown",    "email": "jack@example.com",    "phone": "(415) 555-0110", "age": 38, "status": "pending",  "created_at": "2022-08-01"},  # issue 2 (dup id=2)
        {"id": 11, "name": "Karen Davis",   "email": "karen@example.com",   "phone": "(415) 555-0111", "age": 44, "status": "active",   "created_at": "2022-08-15"},
        {"id": 12, "name": "Leo Martin",    "email": None,                  "phone": "(415) 555-0112", "age": 33, "status": "active",   "created_at": "2022-09-01"},  # issue 3 (null email)
        {"id": 13, "name": "Mia White",     "email": "mia@example.com",     "phone": "(415) 555-0113", "age": 29, "status": "inactive", "created_at": "2022-09-20"},
        {"id": 14, "name": "Noah Taylor",   "email": "noah@example.com",    "phone": "(415) 555-0114", "age": 55, "status": "active",   "created_at": "2022-10-01"},
        {"id": 15, "name": "Olivia Wilson", "email": "olivia@example.com",  "phone": "(415) 555-0115", "age": 36, "status": "banned",   "created_at": "2022-10-15"},  # issue 4
        {"id": 16, "name": "Paul Anderson", "email": "paul@example.com",    "phone": "(415) 555-0116", "age": 48, "status": "active",   "created_at": "2022-11-01"},
        {"id": 17, "name": "Quinn Thomas",  "email": "quinn@example.com",   "phone": "(415) 555-0117", "age": 41, "status": "pending",  "created_at": "2022-11-20"},
        {"id": 18, "name": "Rachel Clark",  "email": "rachel@example.com",  "phone": "(415) 555-0118", "age": 150, "status": "active",  "created_at": "2022-12-01"},  # issue 5
        {"id": 19, "name": "Sam Lewis",     "email": "sam@example.com",     "phone": "(415) 555-0119", "age": 25, "status": "inactive", "created_at": "2022-12-15"},
        {"id": 20, "name": "",              "email": "noname@example.com",  "phone": "(415) 555-0120", "age": 30, "status": "active",   "created_at": "2023-01-01"},  # issue 6
        {"id": 21, "name": "Tina Harris",   "email": "tina@example.com",    "phone": "(415) 555-0121", "age": 37, "status": "active",   "created_at": "2023-01-15"},
        {"id": 22, "name": "Uma Young",     "email": "uma@example.com",     "phone": "(415) 555-0122", "age": 43, "status": "pending",  "created_at": "2023-13-01"},  # issue 7
        {"id": 23, "name": "Victor King",   "email": "victor@example.com",  "phone": "(415) 555-0123", "age": 50, "status": "active",   "created_at": "2023-02-01"},
        {"id": 24, "name": "Wendy Scott",   "email": "wendy@example.com",   "phone": "(415) 555-0124", "age": 39, "status": "inactive", "created_at": "2023-02-15"},
        {"id": 5,  "name": "Xavier Green",  "email": "xavier@example.com",  "phone": "(415) 555-0125", "age": 26, "status": "active",   "created_at": "2023-03-01"},  # issue 8 (dup id=5)
        {"id": 26, "name": "Yara Walker",   "email": "yara@example.com",    "phone": "(415) 555-0126", "age": 32, "status": "pending",  "created_at": "2023-03-15"},
        {"id": 27, "name": "Zoe Evans",     "email": "zoe@example.com",     "phone": "abc-def-ghij",   "age": 58, "status": "active",   "created_at": "2023-04-01"},  # issue 9
        {"id": 28, "name": "Aaron Baker",   "email": "test@",               "phone": "(415) 555-0128", "age": 47, "status": "active",   "created_at": "2023-04-15"},  # issue 10
        {"id": 29, "name": "Beth Nelson",   "email": "beth@example.com",    "phone": "(415) 555-0129", "age": "twenty", "status": "inactive", "created_at": "2023-05-01"},  # issue 11
        {"id": 30, "name": "Carl Carter",   "email": "carl@example.com",    "phone": "(415) 555-0130", "age": 63, "status": None,       "created_at": "2023-05-15"},  # issue 12
    ]
    return base

SCHEMA_VALIDATION_ROWS: List[Dict[str, Any]] = _make_customer_rows()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2  –  Standardization  (Medium)
# Dataset: 20-row sales records with 5 columns that need format normalization
# ─────────────────────────────────────────────────────────────────────────────

# Ground-truth expected outputs after standardization (one per column per row)
# Key: column name → list of 20 standardized values
STANDARDIZATION_GROUND_TRUTH: Dict[str, List[Any]] = {
    "date": [
        "2023-01-15", "2023-02-28", "2023-03-10", "2023-04-05", "2023-05-20",
        "2023-06-14", "2023-07-04", "2023-08-31", "2023-09-12", "2023-10-01",
        "2023-11-11", "2023-12-25", "2023-01-30", "2023-02-14", "2023-03-21",
        "2023-04-18", "2023-05-05", "2023-06-30", "2023-07-15", "2023-08-08",
    ],
    "phone": [
        "(555) 123-4567", "(555) 234-5678", "(555) 345-6789", "(555) 456-7890",
        "(555) 567-8901", "(555) 678-9012", "(555) 789-0123", "(555) 890-1234",
        "(555) 901-2345", "(555) 012-3456", "(555) 111-2222", "(555) 222-3333",
        "(555) 333-4444", "(555) 444-5555", "(555) 555-6666", "(555) 666-7777",
        "(555) 777-8888", "(555) 888-9999", "(555) 999-0000", "(555) 100-2000",
    ],
    "state": [
        "CA", "NY", "TX", "FL", "WA",
        "IL", "OH", "PA", "GA", "NC",
        "MA", "AZ", "CO", "VA", "TN",
        "OR", "MN", "MO", "WI", "MD",
    ],
    "amount": [
        1234.56, 2500.00, 899.99,  4500.00, 150.00,
        3200.75, 750.50,  1100.00, 9999.99, 450.25,
        2800.00, 600.00,  1750.50, 3300.00, 875.25,
        2150.00, 490.00,  5500.00, 1250.75, 675.00,
    ],
    "product_code": [
        "SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005",
        "SKU-006", "SKU-007", "SKU-008", "SKU-009", "SKU-010",
        "SKU-011", "SKU-012", "SKU-013", "SKU-014", "SKU-015",
        "SKU-016", "SKU-017", "SKU-018", "SKU-019", "SKU-020",
    ],
}

def _make_sales_rows() -> List[Dict[str, Any]]:
    dates_raw   = ["2023-01-15", "02/28/2023", "Mar 10 2023", "04/05/23",    "2023-05-20",
                   "Jun 14 2023","07-04-2023", "08/31/2023",  "Sep 12 2023", "2023-10-01",
                   "11/11/2023", "Dec 25 2023","01/30/23",    "Feb 14 2023", "2023-03-21",
                   "04/18/2023", "May 5 2023", "06/30/2023",  "2023-07-15",  "Aug 08 2023"]
    phones_raw  = ["(555) 123-4567", "555-234-5678", "+15553456789",  "5554567890",
                   "555.567.8901",   "(555)678-9012","555 789 0123",  "(555) 890-1234",
                   "555-901-2345",   "5550123456",   "(555) 111-2222","555.222.3333",
                   "+15553334444",   "5554445555",   "555-555-6666",  "(555) 666-7777",
                   "555 777 8888",   "5558889999",   "(555)999-0000", "+15551002000"]
    states_raw  = ["California","New York","TX","Florida","Washington",
                   "Illinois","ohio","PA","Georgia","North Carolina",
                   "Massachusetts","AZ","colorado","Virginia","Tennessee",
                   "OR","Minnesota","Missouri","WI","Maryland"]
    amounts_raw = ["$1,234.56","USD 2500.00","899.99","$4,500.00","150",
                   "$3,200.75","750.50 USD","1,100.00","$9,999.99","450.25",
                   "USD 2800","$600.00","1750.50","$3,300.00","875.25",
                   "2,150.00","$490","5500.00 USD","$1,250.75","675"]
    skus_raw    = ["SKU-001","sku002","003","SKU 004","SKU-005",
                   "sku-006","SKU007","SKU 008","009","SKU-010",
                   "sku011","SKU-012","013","SKU 014","SKU-015",
                   "sku-016","SKU017","SKU 018","019","SKU-020"]
    sales_reps  = ["Alice","Bob","Clara","David","Eva",
                   "Frank","Grace","Hank","Iris","Jack",
                   "Karen","Leo","Mia","Noah","Olivia",
                   "Paul","Quinn","Rachel","Sam","Tina"]
    rows = []
    for i in range(20):
        rows.append({
            "sale_id":   i + 1,
            "rep":       sales_reps[i],
            "date":      dates_raw[i],
            "phone":     phones_raw[i],
            "state":     states_raw[i],
            "amount":    amounts_raw[i],
            "product_code": skus_raw[i],
        })
    return rows

STANDARDIZATION_ROWS: List[Dict[str, Any]] = _make_sales_rows()

# Description of target formats to include in the observation context
STANDARDIZATION_TARGET_FORMATS: Dict[str, str] = {
    "date":         "ISO 8601: YYYY-MM-DD",
    "phone":        "US format: (XXX) XXX-XXXX",
    "state":        "Two-letter uppercase abbreviation (e.g. CA, NY)",
    "amount":       "Plain float with two decimal places (e.g. 1234.56)",
    "product_code": "SKU-NNN format, zero-padded to 3 digits (e.g. SKU-001)",
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3  –  Multi-step Pipeline  (Hard)
# Dataset: 25-row employee records with mixed quality issues
# Pipeline phases: audit → identify → fix → validate
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_SCHEMA: Dict[str, Any] = {
    "employee_id": {"type": "integer",  "required": True, "unique": True},
    "full_name":   {"type": "string",   "required": True, "min_length": 2},
    "department":  {"type": "enum",     "required": True,
                    "values": ["Engineering","Marketing","Sales","Finance","HR","Operations"]},
    "salary":      {"type": "float",    "required": True, "min": 20000, "max": 500000},
    "hire_date":   {"type": "date_iso", "required": True},
    "vacation_days":{"type": "integer", "required": True, "min": 0, "max": 60},
    "email":       {"type": "email",    "required": True},
}

PIPELINE_ISSUES: List[Dict[str, Any]] = [
    {"row_index": 2,  "column": "department",   "issue_type": "invalid_enum",
     "description": "Department 'Devops' not in allowed values"},
    {"row_index": 5,  "column": "salary",       "issue_type": "invalid_range",
     "description": "Salary 850000.0 exceeds schema max of 500000"},
    {"row_index": 9,  "column": "employee_id",  "issue_type": "duplicate",
     "description": "employee_id 9 at row 9 duplicates row 8 (later occurrence flagged)"},
    {"row_index": 11, "column": "hire_date",    "issue_type": "invalid_format",
     "description": "hire_date '15-06-2021' is not ISO format YYYY-MM-DD"},
    {"row_index": 14, "column": "vacation_days","issue_type": "invalid_range",
     "description": "vacation_days -3 is below minimum 0"},
    {"row_index": 17, "column": "salary",       "issue_type": "invalid_range",
     "description": "Salary 1200000.0 exceeds schema max of 500000"},
    {"row_index": 19, "column": "department",   "issue_type": "missing_required",
     "description": "Required field 'department' is null"},
    {"row_index": 21, "column": "hire_date",    "issue_type": "invalid_format",
     "description": "hire_date '2021/08/22' uses slashes instead of dashes"},
    {"row_index": 23, "column": "vacation_days","issue_type": "invalid_range",
     "description": "vacation_days -7 is below minimum 0"},
    {"row_index": 24, "column": "email",        "issue_type": "invalid_format",
     "description": "Email 'notanemail' is missing '@'"},
]

# Issue categories present (for audit step grading)
# Matches the actual issue_types used in PIPELINE_ISSUES above
PIPELINE_ISSUE_CATEGORIES: List[str] = [
    "invalid_enum", "invalid_range", "duplicate", "invalid_format", "missing_required",
]

def _make_employee_rows() -> List[Dict[str, Any]]:
    names = [
        "Alice Johnson","Bob Smith","Clara Devlin","David Brown","Eva Martinez",
        "Frank Wilson","Grace Lee","Hank Thomas","Iris Chen","Iris Chen",      # dup at 8
        "Jack Davis","Karen Miller","Leo Taylor","Mia Anderson","Noah White",
        "Olivia Harris","Paul Clark","Quinn Lewis","Rachel Robinson","Sam Walker",
        "Tina Hall","Uma Young","Victor King","Wendy Wright","Xavier Scott",
    ]
    departments = [
        "Engineering","Marketing","Devops","Sales","Finance",         # 2→invalid
        "HR","Operations","Engineering","Marketing","Sales",
        "Finance","HR","Operations","Engineering","Marketing",
        "Sales","Finance","HR","Operations","Engineering",             # 19→None set below
        "Marketing","Sales","Finance","HR","Operations",
    ]
    departments[19] = None  # missing_required at row 19

    salaries = [
        85000, 72000, 91000, 65000, 110000,
        850000, 78000, 95000, 62000, 83000,   # 5→outlier
        74000, 88000, 57000, 120000, 69000,
        105000, 81000, 1200000, 76000, 94000, # 17→outlier
        87000, 63000, 99000, 73000, 115000,
    ]
    hire_dates = [
        "2019-03-15","2020-07-01","2018-11-20","2021-02-10","2017-06-05",
        "2022-01-15","2020-09-30","2019-04-22","2021-08-14","2018-12-01",
        "2020-03-18","15-06-2021","2019-07-11","2022-03-05","2018-08-28",  # 11→invalid
        "2021-11-17","2020-02-24","2019-10-09","2022-06-01","2018-05-14",
        "2021-03-25","2021/08/22","2020-12-07","2019-01-30","2022-09-18",  # 21→invalid
    ]
    vacation_days = [
        15, 20, 10, 25, 30,
        18, 22, 12, 28, 16,
        24, 19, 11, 27, -3,   # 14→invalid_range
        21, 17, 26, 13, 23,
        29, 14, -7, 20, 18,   # 23→invalid_range
    ]
    emails = [
        "alice.j@company.com","bob.s@company.com","clara.d@company.com",
        "david.b@company.com","eva.m@company.com","frank.w@company.com",
        "grace.l@company.com","hank.t@company.com","iris.c@company.com",
        "iris.c@company.com","jack.d@company.com","karen.m@company.com",
        "leo.t@company.com","mia.a@company.com","noah.w@company.com",
        "olivia.h@company.com","paul.c@company.com","quinn.l@company.com",
        "rachel.r@company.com","sam.w@company.com","tina.h@company.com",
        "uma.y@company.com","victor.k@company.com","wendy.w@company.com",
        "notanemail",  # 24→invalid_format
    ]
    rows = []
    for i in range(25):
        eid = i + 1
        if i == 9:      # duplicate of row 8
            eid = 9
        rows.append({
            "employee_id":   eid,
            "full_name":     names[i],
            "department":    departments[i],
            "salary":        float(salaries[i]),
            "hire_date":     hire_dates[i],
            "vacation_days": vacation_days[i],
            "email":         emails[i],
        })
    return rows

PIPELINE_ROWS: List[Dict[str, Any]] = _make_employee_rows()


# ─────────────────────────────────────────────────────────────────────────────
# Registry – maps task IDs to their data + metadata
# ─────────────────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "schema_validation": {
        "name":         "Customer Records Schema Validation",
        "description":  (
            "Identify all data quality issues in a 30-row customer CRM export "
            "by checking against the provided schema definition."
        ),
        "difficulty":   "easy",
        "max_steps":    1,
        "task_type":    "schema_validation",
        "rows":         SCHEMA_VALIDATION_ROWS,
        "schema":       SCHEMA_VALIDATION_SCHEMA,
        "known_issues": SCHEMA_VALIDATION_ISSUES,
        "sample_size":  30,   # show all rows so the agent can find all 13 issues
    },
    "standardization": {
        "name":         "Sales Records Format Standardization",
        "description":  (
            "Define transformation rules to normalize 5 columns in a 20-row "
            "sales dataset into consistent, machine-readable formats."
        ),
        "difficulty":   "medium",
        "max_steps":    1,
        "task_type":    "standardization",
        "rows":         STANDARDIZATION_ROWS,
        "schema":       {},
        "target_formats": STANDARDIZATION_TARGET_FORMATS,
        "ground_truth": STANDARDIZATION_GROUND_TRUTH,
        "sample_size":  10,
    },
    "pipeline": {
        "name":         "Employee Records Quality Pipeline",
        "description":  (
            "Run a full data-quality pipeline on a 25-row employee dataset: "
            "audit for issue categories, identify specific issues, apply fixes, "
            "then validate the result."
        ),
        "difficulty":   "hard",
        "max_steps":    8,
        "task_type":    "pipeline",
        "rows":         PIPELINE_ROWS,
        "schema":       PIPELINE_SCHEMA,
        "known_issues": PIPELINE_ISSUES,
        "issue_categories": PIPELINE_ISSUE_CATEGORIES,
        "sample_size":  25,   # show all rows so all 10 issues are visible
    },
}

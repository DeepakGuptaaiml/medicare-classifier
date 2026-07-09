"""Shared constants for Medicare classifier pipelines."""

RS = 1

TARGET = "is_medicare_reportable"

FEATURES = [
    "data_set",
    "pay_cat",
    "pay_code",
    "pay_type",
    "paid_1",
    "paid_3",
    "amount",
    "proc_unit",
    "cont_num",
    "claim_open",
    "date_v1m_xmit_flag",
    "is_us_claimant",
    "orm_threshold_met",
    "tpoc_threshold_met",
    "is_wc",
    "pay_code_bucket",
    "is_excluded_coverage",
    "is_excluded_line",
    "days_open",
    "age_at_event",
]

# Alias used by API schemas / preprocess exports
MODEL_FEATURES = FEATURES

CAT_COLS = ["data_set", "pay_cat", "pay_type", "pay_code_bucket"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

INT_COLS = [
    "claim_open",
    "date_v1m_xmit_flag",
    "is_us_claimant",
    "orm_threshold_met",
    "tpoc_threshold_met",
    "is_wc",
    "is_excluded_coverage",
    "is_excluded_line",
]


def pay_code_bucket(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "OTHER"
    if code in (113, 120, 135, 137, 153):
        return "SETTLEMENT"
    if 100 <= code <= 199 or code == 390:
        return "TPOC"
    if 300 <= code <= 399:
        return "ORM"
    return "OTHER"

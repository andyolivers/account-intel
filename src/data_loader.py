"""Data loading and cleaning for the account dataset.

Two things to get right here (found during EDA, see notebooks/eda.ipynb):

1. `region` contains the literal category "NA" (North America). Pandas' default
   read_csv treats the string "NA" as a missing-value marker, so a naive load
   makes ~40% of accounts look region-less. We disable default NA sniffing and
   only treat true empty strings as missing.
2. `ai_usage` and `days_since_last_sales_activity` have a small number of
   genuinely missing values (~2-3%). Both feed the priority score, so we impute
   rather than dropping rows or zero-filling (zero would look like "just talked
   to them" / "zero AI adoption", which is misleading). We impute with the
   **industry** median, not segment: an eta-squared check in the notebook showed
   segment explains almost none of the variance in these two columns (~0.0056,
   ~0.0000) while industry explains meaningfully more (~0.0396, ~0.0213) - e.g.
   Technology accounts have a much lower median days-since-contact than
   Manufacturing, and higher AI adoption than Healthcare. Segment is the right
   grouping for scale-related columns (employees, seats, revenue) but not for
   these two.
"""

import pandas as pd

NUMERIC_COLS = [
    "nr_employees",
    "days_to_next_renewal",
    "days_since_last_sales_activity",
    "nr_support_tickets",
    "ai_usage",
    "nr_active_users",
    "nr_licensed_seats",
    "current_revenue",
    "revenue_end_of_quarter",
]


def load_accounts(path: str) -> pd.DataFrame:
    """Load account_data.csv with the NA-region fix and missing-value imputation."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["ai_usage", "days_since_last_sales_activity"]:
        df[col] = df.groupby("industry")[col].transform(lambda s: s.fillna(s.median()))

    df["has_transcript"] = df["call_transcript_summary"].notna()

    df["seat_utilization"] = df["nr_active_users"] / df["nr_licensed_seats"]

    return df
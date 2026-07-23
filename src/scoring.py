"""Scoring logic: risk, opportunity, urgency, and a combined priority score.

See notebooks/eda.ipynb for the correlation analysis behind this. Four inputs
drive the score, chosen because each correlates with actual revenue movement
(current_revenue -> revenue_end_of_quarter) at |r| > 0.4, even though that
outcome is never used as a score input:

- days_since_last_sales_activity (higher = worse)  -> risk
- nr_support_tickets            (higher = worse)  -> risk
- ai_usage                      (higher = better) -> opportunity
- seat_utilization              (higher = better) -> opportunity
- days_to_next_renewal          (lower = more urgent) -> urgency multiplier

priority_score = urgency * max(risk, opportunity)

Rationale: a calm account renewing far out can wait. A volatile account
(whether trending toward churn or toward expansion) with a renewal coming up
soon needs an AE's attention now - so we surface both directions under one
"priority" ranking, then tag which direction it is.
"""

import numpy as np
import pandas as pd


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def score_accounts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["risk_raw"] = (
        _pct_rank(df["days_since_last_sales_activity"]) * 0.5
        + _pct_rank(df["nr_support_tickets"]) * 0.5
    )
    df["opp_raw"] = (
        _pct_rank(df["ai_usage"]) * 0.5 + _pct_rank(df["seat_utilization"]) * 0.5
    )
    df["urgency"] = 1 - _pct_rank(df["days_to_next_renewal"])
    df["priority_score"] = df["urgency"] * df[["risk_raw", "opp_raw"]].max(axis=1)
    df["flag"] = np.where(df["risk_raw"] >= df["opp_raw"], "At Risk", "Growth Opportunity")

    return df


def top_reasons(row: pd.Series, n: int = 2) -> list[str]:
    """Plain-language drivers behind an account's flag, for display and for the LLM prompt."""
    reasons = []
    if row["flag"] == "At Risk":
        if row["days_since_last_sales_activity"] > 90:
            reasons.append(f"No sales contact in {int(row['days_since_last_sales_activity'])} days")
        if row["nr_support_tickets"] >= 5:
            reasons.append(f"{int(row['nr_support_tickets'])} open support tickets")
        if row["seat_utilization"] < 0.4:
            reasons.append(f"Low seat utilization ({row['seat_utilization']:.0%})")
    else:
        if row["ai_usage"] > 0.6:
            reasons.append(f"High AI feature adoption ({row['ai_usage']:.0%})")
        if row["seat_utilization"] > 0.7:
            reasons.append(f"High seat utilization ({row['seat_utilization']:.0%})")
    if row["days_to_next_renewal"] <= 60:
        reasons.append(f"Renewal in {int(row['days_to_next_renewal'])} days")
    return reasons[:n] if reasons else ["Moderate signals across the board"]
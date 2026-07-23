"""Reusable render functions for the account list, KPI strip, and detail card.

Pure Streamlit components only — no custom CSS/HTML injection. Kept separate
from app.py so page logic (filtering, state, LLM calls) stays clean.
"""

import pandas as pd
import streamlit as st


def render_kpi_strip(df: pd.DataFrame, total: int) -> None:
    at_risk = int((df["flag"] == "At Risk").sum())
    opportunity = int((df["flag"] == "Growth Opportunity").sum())
    renewing_soon = int((df["days_to_next_renewal"] <= 30).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accounts in view", len(df), help=f"out of {total} accounts")
    col2.metric("Accounts at risk", at_risk, help="need retention focus")
    col3.metric("Growth opportunity", opportunity, help="ready to expand")
    col4.metric("Renewing ≤30d", renewing_soon, help="time-sensitive")


def render_account_card(row: pd.Series, reasons: list[str], idx: int, build_prompt, generate_insight) -> None:
    flag = row["flag"]
    is_risk = flag == "At Risk"
    icon = "🔴" if is_risk else "🟢"
    pct = int(round(row["priority_score"] * 100))

    with st.container(border=True):
        header_col, score_col = st.columns([4, 1])
        with header_col:
            st.markdown(f"#### #{idx+1} {row['account_name']}")
            st.caption(f"{row['industry']} · {row['segment']} · {row.get('region', 'n/a')}")
            st.write(f"{icon} **{flag}**")
        with score_col:
            st.metric("Priority score", pct)

        st.progress(pct / 100)

        if reasons:
            st.caption(" · ".join(reasons))

        with st.expander("📋 Meeting prep"):
            st.write(row["account_description"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Renewal", f"{int(row['days_to_next_renewal'])}d")
            m2.metric("Last contact", f"{int(row['days_since_last_sales_activity'])}d ago")
            m3.metric("Support tickets", int(row["nr_support_tickets"]))

            m4, m5, m6 = st.columns(3)
            m4.metric("AI usage", f"{row['ai_usage']:.0%}")
            m5.metric("Seat utilization", f"{row['seat_utilization']:.0%}")
            m6.metric("Revenue", f"${row['current_revenue']:,.0f}")

            st.markdown("**Most recent call notes**")
            if row["has_transcript"]:
                st.info(row["call_transcript_summary"])
            else:
                st.caption("No recent call notes available.")

            if st.button(
                "✦ Generate meeting prep brief",
                key=f"gen_{row['account_id']}",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Pulling account signals and drafting the brief…"):
                    try:
                        prompt = build_prompt(row, reasons)
                        insight = generate_insight(row["account_id"], prompt)
                        st.success(insight)
                    except RuntimeError as e:
                        st.error(str(e))


def render_empty_state(on_clear_label: str = "Clear filters") -> bool:
    st.info("No accounts match these filters. Try widening the segment, industry, or region filters.")
    return st.button(on_clear_label, use_container_width=True)

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_accounts
from src.scoring import score_accounts, top_reasons
from src.llm import build_prompt, generate_insight
from src.ui import render_kpi_strip, render_account_card, render_empty_state

load_dotenv()

st.set_page_config(page_title="Account Intelligence Agent", page_icon="🎯", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "account_data.csv")

SORT_OPTIONS = {
    "Priority (default)": ("priority_score", False),
    "Renewal (soonest first)": ("days_to_next_renewal", True),
    "Revenue (highest first)": ("current_revenue", False),
    "Account name (A-Z)": ("account_name", True),
}


@st.cache_data
def get_scored_data() -> pd.DataFrame:
    df = load_accounts(DATA_PATH)
    df = score_accounts(df)
    return df


df = get_scored_data()

# --- Header ------------------------------------------------------------------
st.title("🎯 Account Intelligence Agent")
st.caption("Surfaces the accounts that most need your attention today, ranked by risk and growth signals, helping you prepare the next meeting with the customer.")


with st.expander("ℹ️ How it works"):
    st.markdown("""
    This dashboard ranks your accounts by a **priority score** that blends renewal timing,
    support ticket volume, sales contact recency, and product usage — so you always know
    who needs attention first.

    **How to use it:**
    - Use the **sidebar filters** to narrow by segment, industry, region, or status (At Risk / Growth Opportunity).
    - **Search** by account name to quickly find a specific customer.
    - **Sort** by priority, renewal date, revenue, or name.
    - **Filter** by the number of accounts to show at once (5-50).
    - Click **Meeting prep** on any account to see key metrics, recent call notes, and generate an AI-drafted briefing before your next call.
    """)
st.divider()

# --- Sidebar filters ----------------------------------------------------------
st.sidebar.header("Filter & sort")

if st.sidebar.button("↺ Reset filters", use_container_width=True):
    for key in ["search", "segments", "industries", "regions", "flag_choice", "sort_choice", "show_n"]:
        st.session_state.pop(key, None)
    st.rerun()

search = st.sidebar.text_input("Account name", key="search", placeholder="e.g. Northwind, Acme…")
segments = st.sidebar.multiselect("Segment", sorted(df["segment"].unique()), key="segments")
industries = st.sidebar.multiselect("Industry", sorted(df["industry"].unique()), key="industries")
regions = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()), key="regions")
flag_choice = st.sidebar.multiselect("Status", ["At Risk", "Growth Opportunity"], key="flag_choice")
sort_choice = st.sidebar.selectbox("Sort by", list(SORT_OPTIONS.keys()), key="sort_choice")
show_n = st.sidebar.slider("Accounts to show", 5, 50, 15, key="show_n")

filtered = df.copy()
if search:
    filtered = filtered[filtered["account_name"].str.contains(search, case=False, na=False)]
if segments:
    filtered = filtered[filtered["segment"].isin(segments)]
if industries:
    filtered = filtered[filtered["industry"].isin(industries)]
if regions:
    filtered = filtered[filtered["region"].isin(regions)]
if flag_choice:
    filtered = filtered[filtered["flag"].isin(flag_choice)]

sort_col, sort_asc = SORT_OPTIONS[sort_choice]
filtered = filtered.sort_values(sort_col, ascending=sort_asc)

# --- KPI strip -----------------------------------------------------------------
render_kpi_strip(filtered, total=len(df))

st.divider()

# --- Account list --------------------------------------------------------------
st.subheader("Prioritized Accounts")

if filtered.empty:
    if render_empty_state():
        for key in ["search", "segments", "industries", "regions", "flag_choice"]:
            st.session_state.pop(key, None)
        st.rerun()
else:
    visible = filtered.head(show_n)
    idx = 0
    for _, row in visible.iterrows():
        reasons = top_reasons(row, n=3)
        render_account_card(row, reasons, idx, build_prompt, generate_insight)
        idx +=1

    if len(filtered) > show_n:
        st.caption(
            f"Showing {show_n} of {len(filtered)} matching accounts — "
            f"adjust “Accounts to show” in the sidebar to see more."
        )

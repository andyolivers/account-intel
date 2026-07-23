# Account Intelligence Agent

An Account Executive's daily briefing tool. It answers two questions: **which
accounts should I focus on today**, and **what should I know before my next
meeting with them**.

Built for the take-home challenge: data layer → scoring layer → AI-generated
insight → Streamlit UI, on the provided `account_data.csv` (~1000 synthetic
accounts).

---

## What the app does

- Loads and cleans the account dataset (see **Key decisions** for two data
  issues found and fixed during EDA).
- Scores every account on a **priority score** that blends renewal urgency
  with risk and opportunity signals, and tags each account `At Risk` or
  `Growth Opportunity`.
- Lets an AE filter (segment, industry, region, status), search by name, and
  sort (priority / soonest renewal / revenue / name).
- Shows a set of KPIs (accounts in view, at risk, growth opportunity, renewing
  ≤30 days).
- Presents the prioritized list of accounts with scoring and detailed information on custom cards.
- On any account, click **Meeting prep** to see key metrics, the most recent
  call notes, and generate an **AI-drafted meeting brief** via the Gemini API.

---
## Repository

```
account-intel/
├── README.md                    
├── requirements.txt             
├── .env.example                  
├── .gitignore                    
├── app.py                        
├── src/
│   ├── __init__.py
│   ├── data_loader.py             
│   ├── scoring.py                 
│   ├── ui.py
│   └── llm.py                     # Gemini prompt construction + API call, caching
├── data/
│   └── account_data.csv           # raw input
├── notebooks/
│   └── eda.ipynb                  # the EDA notebook used for the analysis
└── slides.pptx                    # optional slide deck for the demo
```

---

## Setup and run

**Requirements:** Python 3.11+ (any 3.9+ should work; 3.11 is what this was
built and tested on), a free [Gemini API key](https://aistudio.google.com/app/apikey).

```bash
# 1. Clone and enter the repo
git clone https://github.com/andyolivers/account-intel
cd account-intel

# 2. Create a clean virtual environment (not conda - see note below)
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Add your Gemini API key
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your_key_here

# 5. Run it
streamlit run app.py
```

The app will open at `http://localhost:8501`. Everything (data cleaning,
scoring) runs live from `data/account_data.csv` on startup — no separate
pipeline step needed.

---

## Architecture

| Layer | Implementation | What it does |
|---|---|---|
| **1. Data layer** | `data_loader.py` | Loads `account_data.csv`, fixes the NA-region parsing bug, imputes real gaps, derives `seat_utilization` |
| **2. Scoring / intelligence layer** | `scoring.py` | Turns cleaned data into `priority_score`, `flag`, and `top_reasons()` |
| **3. AI-generated insight** | `llm.py` | Builds a prompt from a row + its scoring output, calls Gemini, caches the result |
| **4. UI** | `app.py` + `ui.py` | Filters/sorts/search, KPI strip, account cards |

This runs as a single local Streamlit process — no deployment, no backend
service, no database. `account_data.csv` is read fresh on each app start
(cached in-memory via `st.cache_data`); the only external dependency at
runtime is the Gemini API call in layer 3, which only fires when an AE
clicks "Generate meeting prep brief," not on every page load.

**Note:** `notebooks/eda.ipynb` is the exploratory work behind `data_loader.py` and
`scoring.py` — it's where the data issues were found and the scoring formula
was validated, not something the app depends on at runtime.

### System view

```
┌──────────┐   browser, localhost:8501    ┌───────────────────────────────┐
│    AE    │ ────────────────────────────▶│        Streamlit app          │
│ (client) │ ◀────────────────────────────│   (app.py - local process)    │
└──────────┘                              └─┬────────────────┬────────────┘
                                            │                │
                                   reads once on startup     │  on-demand only,
                                   (cached in memory)        │  when AE clicks
                                            │                │  "Generate brief"
                                            ▼                ▼
                                 ┌────────────────────┐  ┌──────────────────┐
                                 │ account_data.csv   │  │   Gemini API     │
                                 │  (local file)      │  │  (external, over │
                                 └────────────────────┘  │  the network)    │
                                                         └──────────────────┘
```

---

## Key decisions

### 1. The `region` column has a pandas parsing bug
`region` includes the literal category `"NA"` (North America). Pandas'
default `read_csv` treats the string `"NA"` as a missing-value marker, so a
naive load would make those accounts look region-less. Fixed by disabling
default NA-sniffing (`keep_default_na=False, na_values=[""]`) and casting
numeric columns explicitly afterward.

### 2. Missing values are imputed, not dropped
`ai_usage` and `days_since_last_sales_activity` are ~2-3% missing. Two
questions were checked with evidence rather than assumed:

- **Impute vs. drop:** dropping ~2-3% of a 1000-account book would mean an AE's tool silently loses real accounts. Imputing is the safer default.
- **Which grouping to impute within:** tested segment vs. industry vs.
  region using eta-squared (share of variance in the column explained by
  the grouping). Segment barely explains anything (η² ≈ 0.0056 and 0.0000);
  industry explains meaningfully more (η² ≈ 0.0396 and 0.0213) — e.g.
  Technology accounts have a much shorter median days-since-contact than
  Manufacturing, and higher AI adoption than Healthcare. Imputation groups
  by **industry**, not segment nor region, based on this check rather than intuition.

### 3. The scoring model

The dataset gives us `current_revenue` and `revenue_end_of_quarter`. Money value is the ultimate goal for every AE, hence revenue is the main outcome. We treated the delta as a **stand-in for account health outcome** — useful for sanity-checking our score and findings its main drivers.

Correlating candidate features against actual revenue movement
(`current_revenue` → `revenue_end_of_quarter`, used only to validate, never
as a score input) surfaced four consistent drivers:

Four signals move together with revenue outcome, two ways:

| Direction | Signal | Correlation | Read |
|---|---|---|---|
| Risk ↑ | Days since last sales activity | -0.51 | Longer since we talked to them → revenue shrinks |
| Risk ↑ | Open support tickets | -0.42 | More open tickets → revenue shrinks |
| Opportunity ↑ | AI feature adoption | +0.44 | More AI feature adoption → revenue grows |
| Opportunity ↑ | Seat utilization (active/licensed) | +0.54 | More of their seats actually used → revenue grows |

The number of days until the next renewal on its own barely correlates with the outcome, but still matters as an **urgency multiplier**: a risky account renewing in 20 days needs attention *now*, the same risk profile renewing in 200 days can wait.

As a result, the scoring model created is divided into 3 compounds:
- **Risk score** (0-1): normalized blend of `days_since_last_sales_activity` (higher=worse)
  and `nr_support_tickets` (higher=worse).
- **Opportunity score** (0-1): normalized blend of `ai_usage` and `seat_utilization`
  (higher=better).
- **Urgency** (0-1): inverse of `days_to_next_renewal` — renewals coming up soon matter more.

**Priority score** = `urgency * max(risk, opportunity)` — surfaces accounts where
  something is happening (good or bad) *and* it's time-sensitive. A calm, flat account renewing far out sinks to the bottom, while a volatile account with a renewal next week rises to the top, whether the story is "about to churn" or "ready to expand."

```
risk_raw     = 0.5·pct_rank(days_since_last_sales_activity) + 0.5·pct_rank(nr_support_tickets)
opp_raw      = 0.5·pct_rank(ai_usage) + 0.5·pct_rank(seat_utilization)
urgency      = 1 - pct_rank(days_to_next_renewal)
priority_score = urgency × max(risk_raw, opp_raw)
flag         = "At Risk" if risk_raw >= opp_raw else "Growth Opportunity"
```

**Why urgency multiplies rather than adds:** a calm, healthy account
renewing far in the future can wait, a volatile account (in either
direction) with a renewal coming up soon needs attention *now*. Multiplying
means a high risk/opportunity score with a distant renewal still sinks the
final priority, which matches how an AE actually triages a day.

**Why percentile rank, not min-max scaling:** ranks are robust to outliers
and put every input on the same [0,1] scale regardless of its raw
distribution, without needing per-segment normalization.

**Number of support tickets checks:** `nr_support_tickets` is a raw count, so before
trusting it as a risk signal it was checked against `nr_employees` and
`nr_licensed_seats` for a company-size confound (a bigger account could
generate more tickets just by being bigger). Correlation is ~-0.05 and
median ticket count is flat across segments — not a size proxy.

**Validation:** the top 100 `At Risk` accounts by priority score average
**-36.3%** revenue change; the top 100 `Growth Opportunity` accounts average
**+21.9%**, while the dataset-wide average is **-1.4%**. The score was never shown
this number while being built, so this gap is evidence it's tracking the same signal that shows up in actual revenue movement, not an arbitrary one.

### 4. Why not an anomaly-detection model
Considered and deliberately not used. The question "which accounts need attention" isn't the same as "which accounts look statistically unusual" — a healthy
Enterprise account with an unusually large employee count would flag as an
anomaly and mean nothing to an AE. Anomaly detection would earn its place
if this were per-account time-series data (deviation from *that account's*
own baseline), this dataset is a single snapshot, so that comparison isn't
available.

### 5. Prompt design for the meeting-prep brief
The LLM is given the account's facts, the *already-computed* flag and
reasons, and the call transcript (or an explicit "no recent call notes"
note) — it is not asked to re-derive risk from raw numbers. This keeps the
scoring layer responsible for *what* matters and the LLM responsible for
*explaining it and suggesting talking points*, which is more reliable than
asking one model call to both analyze and narrate.

Structurally, the prompt is split into two parts rather than one blob of
text:

- **System instruction** (persona, output format, tone rules) — set via
  Gemini's `system_instruction` config field, kept separate from the data.
- **User content** (the account's facts) — wrapped in `<account_data>` tags
  with an explicit instruction to treat everything inside as data, never as
  commands. In this dataset `account_description` and
  `call_transcript_summary` are synthetic and safe, but they're free text —
  in a production version, a customer-authored call note is exactly the
  kind of field that could carry adversarial text, so the model is told not
  to follow instructions that might appear inside it.

Output is constrained to three sections (situation / why it matters /
talking points), told to lead with the single most important fact first
(since this gets read in under a minute), told to reference the account's
actual revenue figure rather than speak abstractly about "impact," and told
to flag it explicitly if the call notes and the risk/opportunity assessment
seem to be in tension — rather than silently smoothing over a contradiction
the AE should know about. The prompt also carries explicit
anti-hallucination guardrails ("never invent facts not given above").

---

## Future improvements

- Test alternative weightings inside `risk_raw`/`opp_raw` (currently equal
  0.5/0.5, chosen because the underlying correlations are similar in
  magnitude) against the revenue-delta holdout, rather than eyeballing the
  correlation table.
- Add per-account history if it existed, to properly justify (or rule out)
  an anomaly-detection layer for catching an account breaking its *own*
  pattern, rather than comparing accounts to each other.
- Cache/persist AI-generated briefs to disk instead of only in Streamlit's
  in-memory cache, so they survive a server restart.
- Add a lightweight eval of the LLM's brief quality (e.g. a rubric check for
  "did it use the transcript when available") rather than trusting output
  quality by inspection alone.

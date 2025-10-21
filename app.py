# Streamlit App: Pipeline TimeLens — Phase 1 MVP
# -------------------------------------------------------------
# Upload Salesforce exports (opportunities.csv, stage_history.csv) + optional config.yml.
# The app:
#   1) Normalizes + aliases column names from varied Salesforce exports
#   2) Fits empirical per-stage duration distributions
#   3) Runs Monte Carlo to forecast remaining time to close per opportunity
#   4) Shows per-opportunity P10–P90 windows + portfolio $-weighted close curve
# -------------------------------------------------------------

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yaml
except Exception:
    yaml = None

# -------------------------
# Caching + utilities
# -------------------------
@st.cache_data(show_spinner=False)
def _read_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def _read_yaml(uploaded_file: io.BytesIO) -> Dict:
    if yaml is None:
        return {}
    return yaml.safe_load(uploaded_file)

@st.cache_data(show_spinner=False)
def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if any(k in c.lower() for k in ["date", "time", "at"]):
            try:
                out[c] = pd.to_datetime(out[c], errors="coerce")
            except Exception:
                pass
    return out

@st.cache_data(show_spinner=False)
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    out.columns = [c.strip().replace(" ", "_") for c in out.columns]
    return out

def rename_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Rename DataFrame columns based on lists of possible aliases per canonical name.
    Matches are case-insensitive and expect exact header matches (ignoring case/underscores/spaces).
    """
    if df is None or df.empty:
        return df

    # Build a normalized lookup for incoming df columns (lowercased)
    def norm(s: str) -> str:
        return s.lower().strip()

    incoming = {norm(c): c for c in df.columns}  # map normalized -> actual
    col_map = {}

    for canonical, aliases in mapping.items():
        # canonical itself should also be considered as an alias
        for alias in [canonical] + aliases:
            key = norm(alias)
            if key in incoming:
                col_map[incoming[key]] = canonical
                break  # stop at first match for this canonical name

    return df.rename(columns=col_map)

def require_columns(df: pd.DataFrame, required: List[str], label: str) -> List[str]:
    """Return list of missing required columns for a given frame/label."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in {label}: {missing}")
    return missing

# -------------------------
# Data prep
# -------------------------
def infer_stage_order(stage_history: pd.DataFrame) -> List[str]:
    if stage_history is None or stage_history.empty:
        return []
    # Count transitions and greedily assemble the most common path
    order = (
        stage_history
        .dropna(subset=["FromStage", "ToStage"])
        .value_counts(["FromStage", "ToStage"])
        .reset_index(name="n")
    )
    starts = order.groupby("FromStage")["n"].sum().sort_values(ascending=False)
    if starts.empty:
        return []
    current = starts.index[0]
    seen = {current}
    sequence = [current]
    while True:
        nxt = (order.query("FromStage == @current")
                     .sort_values("n", ascending=False)["ToStage"].tolist())
        nxt = [s for s in nxt if s not in seen]
        if not nxt:
            break
        current = nxt[0]
        sequence.append(current)
        seen.add(current)
    return sequence

def compute_stage_durations(stage_history: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-stage durations (days) from historical intervals using either:
      A) EnteredDate/ExitedDate
      B) StartDate/EndDate
      C) Change log intervals between consecutive ChangeDate rows
    """
    if stage_history is None or stage_history.empty:
        return pd.DataFrame(columns=["Stage", "duration_days"])

    sh = stage_history.copy()
    if {"EnteredDate", "ExitedDate"}.issubset(sh.columns):
        sh["start"] = pd.to_datetime(sh["EnteredDate"], errors="coerce")
        sh["end"]   = pd.to_datetime(sh["ExitedDate"], errors="coerce")
        sh["Stage"] = sh.get("StageName", sh.get("Stage", sh.get("ToStage")))
    elif {"StartDate", "EndDate"}.issubset(sh.columns):
        sh["start"] = pd.to_datetime(sh["StartDate"], errors="coerce")
        sh["end"]   = pd.to_datetime(sh["EndDate"], errors="coerce")
        sh["Stage"] = sh.get("StageName", sh.get("Stage", sh.get("ToStage")))
    else:
        req = {"OpportunityId", "StageName", "ChangeDate"}
        if req.issubset(set(sh.columns)):
            sh = sh.sort_values(["OpportunityId", "ChangeDate"]).copy()
            sh["start"] = sh.groupby("OpportunityId")["ChangeDate"].shift(0)
            sh["end"]   = sh.groupby("OpportunityId")["ChangeDate"].shift(-1)
            sh["Stage"] = sh["StageName"]
        else:
            return pd.DataFrame(columns=["Stage", "duration_days"])

    sh = sh.dropna(subset=["start", "end", "Stage"]).copy()
    sh["duration_days"] = (sh["end"] - sh["start"]).dt.total_seconds() / 86400.0
    sh = sh[(sh["duration_days"] >= 0) & (sh["duration_days"] < 3650)]  # guardrails
    return sh[["Stage", "duration_days"]].reset_index(drop=True)

def build_empirical_samplers(durations: pd.DataFrame) -> Dict[str, np.ndarray]:
    samplers = {}
    if durations is None or durations.empty:
        return samplers
    for stage, grp in durations.groupby("Stage"):
        vals = grp["duration_days"].dropna().values.astype(float)
        if len(vals) > 0:
            samplers[stage] = vals
    return samplers

def sample_stage_durations(stage_list: List[str], samplers: Dict[str, np.ndarray],
                           n: int, fallback_days: float = 7.0) -> np.ndarray:
    if len(stage_list) == 0:
        return np.zeros((n, 0))
    out = np.zeros((n, len(stage_list)), dtype=float)
    rng = np.random.default_rng(42)
    for j, s in enumerate(stage_list):
        arr = samplers.get(s)
        out[:, j] = fallback_days if (arr is None or len(arr) == 0) else rng.choice(arr, size=n, replace=True)
    return out

# -------------------------
# Simulation logic
# -------------------------
def get_current_stage(oppty_row: pd.Series, stage_history: pd.DataFrame) -> str:
    for k in ["CurrentStage", "StageName", "Stage"]:
        if k in oppty_row.index and pd.notna(oppty_row[k]):
            return str(oppty_row[k])
    if stage_history is None or stage_history.empty or "OpportunityId" not in oppty_row.index:
        return None
    h = stage_history[stage_history["OpportunityId"] == oppty_row["OpportunityId"]]
    if {"ChangeDate", "ToStage"}.issubset(h.columns) and not h.empty:
        h = h.sort_values("ChangeDate")
        return str(h.iloc[-1]["ToStage"])
    return None

def remaining_stages(current: str, stage_order: List[str]) -> List[str]:
    if not current or not stage_order or (current not in stage_order):
        return []
    idx = stage_order.index(current)
    return stage_order[idx + 1 :]

def simulate_close_dates(opps: pd.DataFrame, stage_history: pd.DataFrame, stage_order: List[str],
                         samplers: Dict[str, np.ndarray], n_sims: int = 2000) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows = []
    per_opp_samples = {}

    for _, r in opps.iterrows():
        oid = str(r["OpportunityId"])
        cur = get_current_stage(r, stage_history)
        rem = remaining_stages(cur, stage_order)
        if len(rem) == 0:
            close_samples = np.zeros(n_sims)
        else:
            stage_days = sample_stage_durations(rem, samplers, n_sims)
            close_samples = stage_days.sum(axis=1)
        per_opp_samples[oid] = close_samples

        p10, p50, p90 = np.percentile(close_samples, [10, 50, 90]) if len(close_samples) else (0, 0, 0)
        rows.append({
            "OpportunityId": oid,
            "Name": r.get("Name", oid),
            "Amount": float(r.get("Amount", 0) or 0),
            "CurrentStage": cur or "Unknown",
            "RemainingStages": ", ".join(rem) if rem else "(none)",
            "days_p10": p10,
            "days_p50": p50,
            "days_p90": p90,
            "Target_CloseDate": r.get("CloseDate"),
        })

    summary = pd.DataFrame(rows)
    return summary, per_opp_samples

def prob_hit_target(close_samples_days: np.ndarray, target_date: pd.Timestamp) -> float:
    if target_date is None or pd.isna(target_date):
        return np.nan
    today = pd.Timestamp.today().normalize()
    target_delta = (pd.to_datetime(target_date).normalize() - today).days
    return float(np.mean(close_samples_days <= target_delta))

# -------------------------
# Plotting
# -------------------------
def gantt_card(row: pd.Series) -> go.Figure:
    today = pd.Timestamp.today().normalize()
    fig = go.Figure()
    p10 = today + pd.Timedelta(days=float(row["days_p10"]))
    p50 = today + pd.Timedelta(days=float(row["days_p50"]))
    p90 = today + pd.Timedelta(days=float(row["days_p90"]))

    # P10–P90 window
    fig.add_trace(go.Scatter(
        x=[p10, p90], y=[row["Name"], row["Name"]],
        mode="lines", line=dict(width=10), name="P10–P90"
    ))

    # Median marker (use safe literal for Plotly %{...} inside Python strings)
    fig.add_trace(go.Scatter(
        x=[p50, p50], y=[row["Name"], row["Name"]],
        mode="lines", line=dict(width=6, dash="dash"), name="Median",
        hovertemplate=f"<b>{row['Name']}</b><br>Median: " + "%{x|%Y-%m-%d}" + "<extra></extra>"
    ))

    tgt = row.get("Target_CloseDate")
    if pd.notna(tgt):
        fig.add_trace(go.Scatter(
            x=[tgt, tgt], y=[row["Name"], row["Name"]],
            mode="lines", line=dict(width=2, dash="dot"), name="Target"
        ))

    fig.update_layout(
        height=140, margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="", showticklabels=False),
        xaxis=dict(title="Date"),
        showlegend=True
    )
    return fig

def portfolio_curve(sim_summary: pd.DataFrame, per_opp_samples: Dict[str, np.ndarray], opps: pd.DataFrame) -> go.Figure:
    today = pd.Timestamp.today().normalize()
    n_sims = max((len(v) for v in per_opp_samples.values()), default=0)
    if n_sims == 0:
        return go.Figure()

    amounts = {str(r["OpportunityId"]): float(r.get("Amount", 0) or 0) for _, r in opps.iterrows()}
    max_days = int(np.nanmax(sim_summary[["days_p90"]].values)) if not sim_summary.empty else 0
    grid = np.arange(0, max(1, max_days + 30))

    cum_by_sim = np.zeros((n_sims, len(grid)))
    for oid, samples in per_opp_samples.items():
        amt = amounts.get(oid, 0.0)
        idxs = np.clip(samples.astype(int), 0, len(grid) - 1)
        for i, idx in enumerate(idxs):
            cum_by_sim[i, idx:] += amt

    p10 = np.percentile(cum_by_sim, 10, axis=0)
    p50 = np.percentile(cum_by_sim, 50, axis=0)
    p90 = np.percentile(cum_by_sim, 90, axis=0)
    dates = [today + pd.Timedelta(days=int(d)) for d in grid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=p90, name="P90", mode="lines"))
    fig.add_trace(go.Scatter(x=dates, y=p50, name="P50", mode="lines"))
    fig.add_trace(go.Scatter(x=dates, y=p10, name="P10", mode="lines", fill="tonexty"))
    fig.update_layout(
        height=360, margin=dict(l=20, r=20, t=40, b=40),
        xaxis_title="Date", yaxis_title="Cumulative Close ($)",
        legend_title="Percentiles"
    )
    return fig

# -------------------------
# UI
# -------------------------
def main():
    st.set_page_config(page_title="Pipeline TimeLens", page_icon="⏱️", layout="wide")
    st.title("⏱️ Pipeline TimeLens")
    st.caption("Probabilistic Salesforce opportunity timeline forecasting")

    with st.sidebar:
        st.header("Inputs")
        opp_file = st.file_uploader(
            "opportunities.csv", type=["csv"],
            help="Export with OpportunityId, Name, Amount, CloseDate, StageName/CurrentStage"
        )
        sh_file = st.file_uploader(
            "stage_history.csv", type=["csv"],
            help="Export with OpportunityId, FromStage, ToStage, and ChangeDate or Entered/Exited"
        )
        cfg_file = st.file_uploader("config.yml (optional)", type=["yml", "yaml"])
        n_sims = st.number_input("Simulations", min_value=200, max_value=20000, value=4000, step=200)
        fallback_days = st.number_input("Fallback days for missing stage data", min_value=1.0, max_value=60.0, value=7.0, step=1.0)

    if not opp_file or not sh_file:
        st.info("Upload opportunities and stage history to begin.")
        st.stop()

    # --- Load + normalize + alias map ---
    opps = normalize_columns(coerce_dates(_read_csv(opp_file)))
    sh   = normalize_columns(coerce_dates(_read_csv(sh_file)))

    # Canonical -> aliases that might show up in Salesforce exports
    oppty_aliases = {
        "OpportunityId": ["opportunity id", "opportunity_id", "id"],
        "Name": ["name", "opportunity name"],
        "Amount": ["amount", "amount usd", "amount ($)"],
        "CloseDate": ["close date", "closedate"],
        "StageName": ["stage", "stage name", "current stage"],
        "IsClosed": ["is closed", "closed"],
        "IsWon": ["is won", "won"],
        "CreatedDate": ["created date", "createddate"],
    }
    stage_aliases = {
        "OpportunityId": ["opportunity id", "opportunity_id", "id"],
        "FromStage": ["from stage", "from_stage", "prior stage"],
        "ToStage": ["to stage", "to_stage", "new stage", "next stage"],
        "ChangeDate": ["change date", "changedate", "date/time", "modified date"],
        # optional alternates:
        "EnteredDate": ["entered date", "entereddate", "start date", "startdate"],
        "ExitedDate": ["exited date", "exiteddate", "end date", "enddate"],
        "StageName": ["stage", "stage name"],
        "StartDate": ["start date", "startdate"],
        "EndDate": ["end date", "enddate"],
    }

    opps = rename_columns(opps, oppty_aliases)
    sh   = rename_columns(sh, stage_aliases)

    # Minimal required columns
    missing_a = require_columns(opps, ["OpportunityId"], "opportunities.csv")
    missing_b = require_columns(sh,   ["OpportunityId"], "stage_history.csv")
    if missing_a or missing_b:
        st.stop()

    # Optional CloseDate parsing standardization if provided
    if "CloseDate" in opps.columns:
        opps["CloseDate"] = pd.to_datetime(opps["CloseDate"], errors="coerce")

    # --- Config (stage order override) ---
    config = {}
    if cfg_file is not None:
        try:
            config = _read_yaml(cfg_file) or {}
        except Exception as e:
            st.warning(f"Failed to read config: {e}")

    stage_order = config.get("stage_order") or infer_stage_order(
        sh.rename(columns={"From_Stage": "FromStage", "To_Stage": "ToStage"})
    )
    if not stage_order:
        st.error("Couldn't determine stage order. Provide config.yml with stage_order or ensure stage_history has FromStage/ToStage and ChangeDate.")
        st.stop()

    st.write("**Stage order:**", " → ".join(stage_order))

    # --- Build duration samplers ---
    durations = compute_stage_durations(
        sh.rename(columns={"From_Stage": "FromStage", "To_Stage": "ToStage", "Stage": "StageName"})
    )
    samplers = build_empirical_samplers(durations)
    if not samplers:
        st.warning("No historical stage durations available; using fallback days for all stages.")

    # Open opp filter (if provided)
    if "IsClosed" in opps.columns:
        # Try robust boolean conversion
        ic = opps["IsClosed"]
        if ic.dtype == object:
            opps["IsClosed"] = ic.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])
        opps_open = opps[~opps["IsClosed"].astype(bool)].copy()
    else:
        opps_open = opps.copy()

    st.subheader("Simulation")
    sim_summary, per_opp_samples = simulate_close_dates(opps_open, sh, stage_order, samplers, int(n_sims))

    if sim_summary.empty:
        st.warning("No open opportunities or insufficient data to simulate.")
        st.stop()

    st.dataframe(
        sim_summary[[
            "OpportunityId","Name","Amount","CurrentStage","RemainingStages",
            "days_p10","days_p50","days_p90","Target_CloseDate"
        ]],
        use_container_width=True
    )

    # Tabs
    tab1, tab2 = st.tabs(["Opportunity view", "Portfolio view"])

    with tab1:
        sel = st.selectbox(
            "Select opportunity",
            sim_summary["Name"] + " (" + sim_summary["OpportunityId"].astype(str) + ")"
        )
        chosen_id = sel.split("(")[-1].strip(")")
        row = sim_summary[sim_summary["OpportunityId"] == chosen_id].iloc[0]

        fig = gantt_card(row)
        st.plotly_chart(fig, use_container_width=True)

        target_dt = row.get("Target_CloseDate")
        p_hit = prob_hit_target(per_opp_samples.get(chosen_id, np.array([])), target_dt)
        if pd.notna(target_dt):
            st.metric(label=f"Probability to hit target ({pd.to_datetime(target_dt).date()})", value=f"{p_hit*100:.1f}%")
        else:
            st.caption("No target close date provided for this opportunity.")

    with tab2:
        figp = portfolio_curve(sim_summary, per_opp_samples, opps_open)
        st.plotly_chart(figp, use_container_width=True)
        total_amt = float(opps_open.get("Amount", pd.Series(dtype=float)).fillna(0).sum())
        st.metric("Total pipeline ($)", f"{total_amt:,.0f}")

    with st.expander("Debug / config"):
        st.json({
            "stage_order": stage_order,
            "n_stages": len(stage_order),
            "n_opps": int(len(opps_open)),
            "n_durations": int(len(durations)),
            "stages_with_samples": sorted(list(samplers.keys())),
        })

if __name__ == "__main__":
    main()

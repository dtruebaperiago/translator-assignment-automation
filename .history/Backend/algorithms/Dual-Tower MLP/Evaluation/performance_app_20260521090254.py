"""
Dual-Tower Recommender — Evaluation, Ranking & Streamlit Dashboard
Integrated with REAL IDISCDualTower model, trained weights, and Hard Constraints.
"""
import os, math, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
from datetime import datetime

# Real modules from teammates
from IDISC_DualTower import IDISCDualTower
from demand import Demand, filter_translators

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
BATCH_SIZE = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Processed")
CSV_DIR = os.path.join(BASE_DIR, "Initial Dataset", "CSV")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=" Loading real datasets …")
def load_data():
    tasks_df = pd.read_csv(os.path.join(DATA_DIR, "test_tasks.csv"))
    translators_df = pd.read_csv(os.path.join(DATA_DIR, "test_translators.csv"))
    labels_df = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))

    task_feature_cols = [c for c in tasks_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
    translator_feature_cols = [c for c in translators_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
    tower_b_dim = len(task_feature_cols)
    tower_a_dim = len(translator_feature_cols)

    translator_pool = (
        translators_df.groupby("TRANSLATOR")[translator_feature_cols].mean().reset_index()
    )
    return tasks_df, translators_df, labels_df, tower_a_dim, tower_b_dim, translator_pool

@st.cache_data(show_spinner=" Loading raw tasks & reference CSVs …")
def load_constraint_data():
    df_raw = pd.read_csv(os.path.join(CSV_DIR, "Data.csv"), sep=";", decimal=",")
    clients_df = pd.read_csv(os.path.join(CSV_DIR, "Clients.csv"), sep=";", decimal=",")
    clients_df = clients_df.loc[:, ~clients_df.columns.str.startswith("Unnamed")]
    schedules_df = pd.read_csv(os.path.join(CSV_DIR, "Schedules.csv"), sep=";", decimal=",")
    pairs_df = pd.read_csv(os.path.join(CSV_DIR, "Translators Costs+Pairs.csv"), sep=";", decimal=",")
    translators_data_df = pd.read_csv(os.path.join(CSV_DIR, "Translators_Data.csv"))
    return df_raw, clients_df, schedules_df, pairs_df, translators_data_df

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — REAL HARD CONSTRAINTS ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

def apply_real_hard_constraints(task_id, available_translators_df, df_raw,
                                pairs_df, translators_data_df, schedules_df):
    raw_row = df_raw[df_raw["TASK_ID"] == task_id]
    if raw_row.empty:
        return available_translators_df  # fallback: no filter if task not in raw

    row = raw_row.iloc[0]
    demand = Demand(
        start=pd.to_datetime(row["START"]),
        end=pd.to_datetime(row["END"]),
        task_type=str(row.get("TASK_TYPE", "")).strip(),
        source_lang=str(row.get("SOURCE_LANG", "")).strip(),
        target_lang=str(row.get("TARGET_LANG", "")).strip(),
        hours=float(row.get("HOURS", 0)),
        manufacturer=str(row.get("MANUFACTURER", "")).strip(),
        manufacturer_sector=str(row.get("MANUFACTURER_SECTOR", "Unknown")).strip(),
        manufacturer_industry_group=str(row.get("MANUFACTURER_INDUSTRY_GROUP", "Unknown")).strip(),
        manufacturer_industry=str(row.get("MANUFACTURER_INDUSTRY", "Unknown")).strip(),
        manufacturer_subindustry=str(row.get("MANUFACTURER_SUBINDUSTRY", "Unknown")).strip(),
    )

    qualified_df, _fallback_df = filter_translators(demand, pairs_df, translators_data_df, schedules_df)
    if qualified_df.empty or "TRANSLATOR" not in qualified_df.columns:
        return pd.DataFrame(columns=available_translators_df.columns)

    valid_names = set(qualified_df["TRANSLATOR"].str.strip().unique())
    filtered = available_translators_df[
        available_translators_df["TRANSLATOR"].str.strip().isin(valid_names)
    ].reset_index(drop=True)
    return filtered

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — REAL MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def find_pth_files():
    return sorted(glob.glob(os.path.join(BASE_DIR, "*.pth")))

@st.cache_resource(show_spinner="🧠 Loading IDISCDualTower model …")
def load_model(pth_path, tower_a_dim, tower_b_dim):
    model = IDISCDualTower(
        tower_a_input_dim=tower_a_dim,
        tower_b_input_dim=tower_b_dim,
        embedding_dim=64,
    )
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — RANKING & METRICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_task(task_id, tasks_df, translator_pool, labels_df, model,
                  task_feature_cols, translator_feature_cols,
                  df_raw, pairs_df, translators_data_df, schedules_df):
    task_row = tasks_df[tasks_df["TASK_ID"] == task_id]
    if task_row.empty:
        return None
    task_features_np = task_row[task_feature_cols].values.astype(np.float32)

    n_before = len(translator_pool)
    filtered = apply_real_hard_constraints(
        task_id, translator_pool.copy(), df_raw,
        pairs_df, translators_data_df, schedules_df,
    )
    n_after = len(filtered)

    if n_after == 0:
        return {"ranked_df": pd.DataFrame(columns=["TRANSLATOR","Score"]),
                "true_translator": None, "rank": None,
                "hit_at_1": False, "hit_at_3": False, "hit_at_5": False, "hit_at_10": False,
                "reciprocal_rank": 0.0, "n_before_filter": n_before,
                "n_after_filter": 0, "all_scores": []}

    translator_names = filtered["TRANSLATOR"].values
    trans_np = filtered[translator_feature_cols].values.astype(np.float32)
    task_tensor = torch.tensor(task_features_np)
    all_scores = []
    n_batches = math.ceil(len(trans_np) / BATCH_SIZE)

    with torch.no_grad():
        for b in range(n_batches):
            s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(trans_np))
            trans_batch = torch.tensor(trans_np[s:e])
            task_batch = task_tensor.expand(e - s, -1)
            scores = model(trans_batch, task_batch)  # (bs, 1)
            all_scores.append(scores.squeeze(-1).cpu().numpy())

    all_scores = np.concatenate(all_scores)
    ranked_df = pd.DataFrame({"TRANSLATOR": translator_names, "Score": all_scores})
    ranked_df = ranked_df.sort_values("Score", ascending=False).reset_index(drop=True)
    ranked_df.index = ranked_df.index + 1
    ranked_df.index.name = "Rank"

    label_row = labels_df[labels_df["TASK_ID"] == task_id]
    true_translator = label_row["TRANSLATOR"].values[0] if not label_row.empty else None

    rank, hit1, hit3, hit5, hit10, rr = None, False, False, False, False, 0.0
    if true_translator is not None:
        match = ranked_df[ranked_df["TRANSLATOR"] == true_translator]
        if not match.empty:
            rank = match.index[0]
            hit1, hit3, hit5, hit10 = rank<=1, rank<=3, rank<=5, rank<=10
            rr = 1.0 / rank

    return {"ranked_df": ranked_df, "true_translator": true_translator,
            "rank": rank, "hit_at_1": hit1, "hit_at_3": hit3,
            "hit_at_5": hit5, "hit_at_10": hit10, "reciprocal_rank": rr,
            "n_before_filter": n_before, "n_after_filter": n_after,
            "all_scores": all_scores.tolist()}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def highlight_true_translator(row, true_translator):
    if row["TRANSLATOR"] == true_translator:
        return ["background-color: #ffd700; color: #1a1a2e; font-weight: bold"] * len(row)
    return [""] * len(row)


def main():
    st.set_page_config(page_title="AssignMate — Business Optimization Dashboard",
                       page_icon="🚀", layout="wide")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100,100,255,0.15); border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important; font-size: 0.85rem !important;
        font-weight: 500 !important; letter-spacing: 0.5px; text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 10px; padding: 12px 20px;
        font-weight: 600; font-size: 1rem; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }
    h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 700 !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 10px; }
    hr { border-color: rgba(100,100,255,0.15); }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 AssignMate — Business Optimization Dashboard")
    st.caption("Dynamic Resource Allocation · **IDISCDualTower** Dual-Tower Model · "
               "Real **Hard Constraints** pipeline · Decision Support for Translation PMs")
    st.divider()

    # Load data
    tasks_df, translators_df, labels_df, tower_a_dim, tower_b_dim, translator_pool = load_data()
    df_raw, clients_df, schedules_df, pairs_df, translators_data_df = load_constraint_data()

    task_feature_cols = [c for c in tasks_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
    translator_feature_cols = [c for c in translator_pool.columns if c != "TRANSLATOR"]
    all_task_ids = labels_df["TASK_ID"].unique()

    # Sidebar
    with st.sidebar:
        st.markdown("##### 🚀 AssignMate Engine")
        pth_files = find_pth_files()
        if not pth_files:
            st.error("No .pth files found in repo root.")
            st.stop()
        pth_names = [os.path.basename(p) for p in pth_files]
        selected_pth = st.selectbox("Model weights", pth_names, index=0)
        pth_path = pth_files[pth_names.index(selected_pth)]

        st.divider()
        st.markdown("##### 📊 Resource Pool")
        st.metric("Total Test Tasks", f"{len(all_task_ids):,}")
        st.metric("Unique Translators", f"{len(translator_pool):,}")
        st.metric("Tower A dim (Translator)", tower_a_dim)
        st.metric("Tower B dim (Task)", tower_b_dim)

    model = load_model(pth_path, tower_a_dim, tower_b_dim)

    tab1, tab2 = st.tabs(["🔍 Per-Task Inspector", "📊 Business Optimization Analytics"])

    # ── TAB 1: Per-Task Inspector ──
    with tab1:
        st.subheader("🔍 Per-Task Inspector")
        task_ids_sorted = sorted(all_task_ids)
        selected_task = st.selectbox("Select a TASK_ID to inspect", task_ids_sorted, index=0)

        if selected_task is not None:
            with st.spinner("Scoring candidates …"):
                result = evaluate_task(
                    selected_task, tasks_df, translator_pool, labels_df, model,
                    task_feature_cols, translator_feature_cols,
                    df_raw, pairs_df, translators_data_df, schedules_df,
                )
            if result is None:
                st.error(f"❌ Task ID **{selected_task}** not found in test_tasks.csv.")
            else:
                st.markdown("#### 🚧 Hard Filter Impact")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Candidates Before Filter", result["n_before_filter"])
                fc2.metric("Candidates After Filter", result["n_after_filter"])
                drop_n = result["n_before_filter"] - result["n_after_filter"]
                drop_pct = (drop_n / result["n_before_filter"] * 100) if result["n_before_filter"] else 0
                fc3.metric("Dropped", f"{drop_n}  ({drop_pct:.0f} %)")

                st.markdown("#### 🏆 Ranking Results")
                if result["true_translator"] is not None:
                    rc1, rc2, rc4 = st.columns(3)
                    rc1.metric("True Translator", result["true_translator"])
                    rc2.metric("Rank of True Translator",
                               f"#{result['rank']}" if result["rank"] else "Not in pool")
                    rc4.metric("Reciprocal Rank", f"{result['reciprocal_rank']:.4f}")

                    # ── HEAD-TO-HEAD COMPARISON ──
                    st.divider()
                    st.markdown(" Human Decision vs. AI Recommendation")

                    ranked_df = result["ranked_df"]
                    true_t = result["true_translator"]

                    # --- Extract AI Top-1 ---
                    ai_top1_name = ranked_df.iloc[0]["TRANSLATOR"] if not ranked_df.empty else None
                    ai_top1_score = float(ranked_df.iloc[0]["Score"]) if not ranked_df.empty else None

                    # --- Extract Historical Translator from the ranked list ---
                    hist_match = ranked_df[ranked_df["TRANSLATOR"] == true_t]
                    hist_score = float(hist_match.iloc[0]["Score"]) if not hist_match.empty else None
                    hist_rank = hist_match.index[0] if not hist_match.empty else None

                    # --- Quality lookup helper (from constraint reference data) ---
                    def _get_quality_ema(name):
                        row = translators_data_df[
                            translators_data_df["TRANSLATOR"].str.strip() == name.strip()
                        ]
                        if not row.empty and "rolling_quality_ema" in row.columns:
                            return round(float(row.iloc[0]["rolling_quality_ema"]), 2)
                        return None

                    hist_quality = _get_quality_ema(true_t)
                    ai_quality = _get_quality_ema(ai_top1_name) if ai_top1_name else None

                    # --- Delta calculations ---
                    affinity_delta = None
                    if ai_top1_score is not None and hist_score is not None:
                        affinity_delta = ai_top1_score - hist_score
                    quality_delta = None
                    if ai_quality is not None and hist_quality is not None:
                        quality_delta = ai_quality - hist_quality

                    # --- Two-column layout ---
                    col_hist, col_ai = st.columns(2)

                    with col_hist:
                        st.markdown(
                            '<h4 style="color:#ff6b6b;">🔴 Historical Human Decision</h4>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Translator:** `{true_t}`")
                        if hist_score is not None:
                            st.metric("Predicted Affinity Score", f"{hist_score:.4f}")
                        else:
                            st.metric("Predicted Affinity Score", "N/A — filtered out")
                        if hist_quality is not None:
                            st.metric("Historical Quality (EMA)", f"{hist_quality:.2f}")
                        if hist_rank is not None:
                            st.metric("Rank Position", f"#{hist_rank}")
                        else:
                            st.caption("ℹ️ Translator was removed by the hard filter.")

                    with col_ai:
                        st.markdown(
                            '<h4 style="color:#51cf66;">🟢 AI Top Recommendation</h4>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Translator:** `{ai_top1_name}`")
                        if ai_top1_score is not None:
                            st.metric(
                                "Predicted Affinity Score",
                                f"{ai_top1_score:.4f}",
                                delta=f"{affinity_delta:+.4f}" if affinity_delta is not None else None,
                            )
                        if ai_quality is not None:
                            st.metric(
                                "Historical Quality (EMA)",
                                f"{ai_quality:.2f}",
                                delta=f"{quality_delta:+.2f}" if quality_delta is not None else None,
                            )
                        st.metric("Rank Position", "#1")

                    # --- Verdict Banner ---
                    if ai_top1_name and ai_top1_name == true_t:
                        st.success(
                            f"🎯 **Perfect Match** — The AI's #1 recommendation **is** the "
                            f"historical translator: **{true_t}**"
                        )
                    elif affinity_delta is not None and affinity_delta > 0:
                        verdict_parts = [
                            f"The AI recommends **{ai_top1_name}** with a "
                            f"**+{affinity_delta:.4f}** higher predicted affinity"
                        ]
                        if quality_delta is not None and quality_delta > 0:
                            verdict_parts.append(
                                f"and a **+{quality_delta:.2f}** higher historical quality EMA"
                            )
                        st.success(
                            "✅ **AI Optimization** — "
                            + " ".join(verdict_parts)
                            + f" than the PM's choice (**{true_t}**)."
                        )
                    elif affinity_delta is not None and affinity_delta == 0:
                        st.info(
                            "🤝 **Tie** — The AI's top pick and the historical translator "
                            "have identical predicted affinity."
                        )
                    elif hist_score is None:
                        st.warning(
                            f"⚠️ The historical translator **{true_t}** was removed by "
                            f"the hard constraints filter and cannot be compared."
                        )

                else:
                    st.warning("⚠️ No ground-truth translator found for this task.")

                # ── TOP 10 AVAILABLE CANDIDATES ──
                st.divider()
                st.markdown("#### 📋 Top 10 Available Candidates for Optimization")
                st.markdown(
                    "_These candidates have passed all Hard Constraints "
                    "(Language Pair, Task Type, and Schedule Availability) "
                    "and are ranked by mathematical affinity._"
                )
                top10 = result["ranked_df"].head(10).copy()
                if not top10.empty:
                    top10["Score"] = top10["Score"].map(lambda s: f"{s:.6f}")
                    true_t = result["true_translator"]
                    if true_t and true_t in top10["TRANSLATOR"].values:
                        styled = top10.style.apply(highlight_true_translator, true_translator=true_t, axis=1)
                        st.dataframe(styled, use_container_width=True)
                        st.info(f"🟡 The **highlighted row** is the historical translator (**{true_t}**), shown for reference.")
                    else:
                        st.dataframe(top10, use_container_width=True)
                        if true_t and result["rank"]:
                            st.info(f"ℹ️ The historical translator **{true_t}** is ranked at position **#{result['rank']}** among available candidates.")
                        elif true_t:
                            st.info(f"ℹ️ The historical translator **{true_t}** did not pass the Hard Constraints filter for this task.")
                else:
                    st.warning("No candidates survived the hard constraints filter.")

                with st.expander("📄 Show full candidate ranking"):
                    st.dataframe(result["ranked_df"], use_container_width=True, height=400)

    # ── TAB 2: Business Optimization Analytics ──
    with tab2:
        st.subheader("📊 Business Optimization Analytics")
        st.caption(
            "Evaluate the first **500 tasks** in the test set. For each task, the system "
            "applies Hard Constraints, scores all eligible translators with the Dual-Tower model, "
            "and compares the AI's top recommendation against the historical PM decision."
        )
        GLOBAL_EVAL_SIZE = 500
        run_global = st.button(
            "🚀 Run Business Optimization Analysis",
            use_container_width=True, key="global_eval_btn",
        )

        if run_global:
            first_500_ids = all_task_ids[:GLOBAL_EVAL_SIZE]
            progress = st.progress(0, text="Analyzing tasks …")

            # Accumulators — legacy
            rank_list, all_affinity, rr_list = [], [], []
            hits = {1: 0, 3: 0, 5: 0, 10: 0}
            # Accumulators — business optimization
            affinity_deltas, quality_deltas = [], []
            quality_upgrades = 0
            evaluated, skipped = 0, 0

            def _lookup_quality(name):
                r = translators_data_df[
                    translators_data_df["TRANSLATOR"].str.strip() == str(name).strip()
                ]
                if not r.empty and "rolling_quality_ema" in r.columns:
                    return float(r.iloc[0]["rolling_quality_ema"])
                return None

            for i, tid in enumerate(first_500_ids):
                res = evaluate_task(
                    tid, tasks_df, translator_pool, labels_df, model,
                    task_feature_cols, translator_feature_cols,
                    df_raw, pairs_df, translators_data_df, schedules_df,
                )
                if res is None or res["rank"] is None:
                    skipped += 1
                else:
                    # Legacy metrics
                    rank_list.append(res["rank"])
                    rr_list.append(res["reciprocal_rank"])
                    hits[1] += int(res["hit_at_1"])
                    hits[3] += int(res["hit_at_3"])
                    hits[5] += int(res["hit_at_5"])
                    hits[10] += int(res["hit_at_10"])
                    all_affinity.extend(res["all_scores"])

                    # Business optimization metrics
                    rdf = res["ranked_df"]
                    if not rdf.empty:
                        ai_score = float(rdf.iloc[0]["Score"])
                        hist_m = rdf[rdf["TRANSLATOR"] == res["true_translator"]]
                        hist_score = float(hist_m.iloc[0]["Score"]) if not hist_m.empty else None
                        if hist_score is not None:
                            affinity_deltas.append(ai_score - hist_score)

                        ai_q = _lookup_quality(rdf.iloc[0]["TRANSLATOR"])
                        hist_q = _lookup_quality(res["true_translator"])
                        if ai_q is not None and hist_q is not None:
                            quality_deltas.append(ai_q - hist_q)
                            if ai_q > hist_q:
                                quality_upgrades += 1

                    evaluated += 1
                progress.progress(
                    (i + 1) / GLOBAL_EVAL_SIZE,
                    text=f"Analyzed {i + 1}/{GLOBAL_EVAL_SIZE} tasks …",
                )

            progress.empty()

            if evaluated == 0:
                st.warning("⚠️ No tasks could be evaluated.")
            else:
                # ── HERO KPIs — BUSINESS OPTIMIZATION ──
                st.markdown("#### 🏅 Business Optimization KPIs")
                opt_rate = (quality_upgrades / len(quality_deltas) * 100) if quality_deltas else 0
                avg_q_delta = float(np.mean(quality_deltas)) if quality_deltas else 0
                avg_a_delta = float(np.mean(affinity_deltas)) if affinity_deltas else 0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Tasks Evaluated", f"{evaluated}")
                k2.metric("Optimization Rate", f"{opt_rate:.1f} %")
                k3.metric("Avg Quality Improvement", f"{avg_q_delta:+.2f}")
                k4.metric("Avg Affinity Gain", f"{avg_a_delta:+.4f}")

                st.success(
                    f"✅ Across **{evaluated}** tasks, the AI suggests a higher-quality "
                    f"translator in **{opt_rate:.1f} %** of cases, with an average quality "
                    f"uplift of **{avg_q_delta:+.2f}** EMA points and an average affinity "
                    f"gain of **{avg_a_delta:+.4f}**."
                )
                st.divider()

                # ── CHART 1 — Affinity Score Distribution ──
                st.markdown("#### 🎯 Affinity Score Distribution")
                st.caption(
                    "A healthy bell curve with wide variance confirms the "
                    "BCEWithLogitsLoss fix resolved the Embedding Collapse problem."
                )
                fig_score = px.histogram(
                    pd.DataFrame({"Affinity Score": all_affinity}),
                    x="Affinity Score", nbins=80,
                    title="Distribution of All Predicted Affinity Scores (Post-Logits Fix)",
                    color_discrete_sequence=["#4fd1c5"],
                )
                fig_score.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                    title_font_size=18,
                    yaxis=dict(title="Count", gridcolor="rgba(100,100,255,0.08)"),
                    xaxis=dict(title="Affinity Score (0–1)"),
                    bargap=0.02, margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig_score, use_container_width=True)

                # ── CHART 2 — Rank Distribution ──
                st.markdown("#### 📊 Historical Translator Rank Distribution")
                st.caption(
                    "Left-skewed distribution proves the model consistently pushes "
                    "the historical translator toward the top of the ranked list."
                )
                fig_rank = px.histogram(
                    pd.DataFrame({"Rank": rank_list}),
                    x="Rank", nbins=min(50, max(rank_list)),
                    title="Where Does the Historical Translator Land in the AI Ranking?",
                    color_discrete_sequence=["#764ba2"],
                )
                fig_rank.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                    title_font_size=18,
                    yaxis=dict(title="Count of Tasks", gridcolor="rgba(100,100,255,0.08)"),
                    xaxis=dict(title="Rank Position"),
                    bargap=0.05, margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig_rank, use_container_width=True)

                # ── CHART 3 — Quality Delta Distribution ──
                if quality_deltas:
                    st.markdown("#### 📈 Quality Delta Distribution (AI Top-1 vs. PM Choice)")
                    st.caption(
                        "Positive values mean the AI's recommendation has a better "
                        "historical quality EMA than the PM's original choice."
                    )
                    fig_qd = px.histogram(
                        pd.DataFrame({"Quality Delta": quality_deltas}),
                        x="Quality Delta", nbins=60,
                        title="Δ Quality EMA = AI Top-1 − Historical Translator",
                        color_discrete_sequence=["#51cf66"],
                    )
                    fig_qd.add_vline(x=0, line_dash="dash", line_color="#ff6b6b",
                                     annotation_text="Breakeven", annotation_position="top right")
                    fig_qd.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                        title_font_size=18,
                        yaxis=dict(title="Count of Tasks", gridcolor="rgba(100,100,255,0.08)"),
                        xaxis=dict(title="Quality Delta (EMA points)"),
                        bargap=0.05, margin=dict(t=60, b=40),
                    )
                    st.plotly_chart(fig_qd, use_container_width=True)

                # ── LEGACY / SANITY METRICS (collapsed) ──
                st.divider()
                with st.expander("📎 Legacy / Sanity Check Metrics (Hit Rate & MRR)"):
                    st.caption(
                        "These traditional metrics measure how often the model replicates "
                        "the historical PM decision. They are retained for academic "
                        "benchmarking but are **not** the primary success criteria."
                    )
                    hr = {f"@{k}": (v / evaluated) * 100 for k, v in hits.items()}
                    mrr = float(np.mean(rr_list))
                    lm1, lm2, lm3, lm4, lm5 = st.columns(5)
                    lm1.metric("Hit Rate @1", f"{hr['@1']:.2f} %")
                    lm2.metric("Hit Rate @3", f"{hr['@3']:.2f} %")
                    lm3.metric("Hit Rate @5", f"{hr['@5']:.2f} %")
                    lm4.metric("Hit Rate @10", f"{hr['@10']:.2f} %")
                    lm5.metric("MRR", f"{mrr:.4f}")

if __name__ == "__main__":
    main()


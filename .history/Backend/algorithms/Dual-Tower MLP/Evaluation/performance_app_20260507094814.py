
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

    qualified_df = filter_translators(demand, pairs_df, translators_data_df, schedules_df)
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
    st.set_page_config(page_title="Dual-Tower Recommender — Evaluation Dashboard",
                       page_icon="🏗️", layout="wide")
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

    st.title("🏗️ Dual-Tower Recommender — Evaluation Dashboard")
    st.caption("Real **IDISCDualTower** model · Real **Hard Constraints** pipeline · "
               "Data from `Processed/test_*.csv` · Batch size = 64")
    st.divider()

    # Load data
    tasks_df, translators_df, labels_df, tower_a_dim, tower_b_dim, translator_pool = load_data()
    df_raw, clients_df, schedules_df, pairs_df, translators_data_df = load_constraint_data()

    task_feature_cols = [c for c in tasks_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
    translator_feature_cols = [c for c in translator_pool.columns if c != "TRANSLATOR"]
    all_task_ids = labels_df["TASK_ID"].unique()

    # Sidebar
    with st.sidebar:
        st.markdown("##### 🧠 Select Trained Model")
        pth_files = find_pth_files()
        if not pth_files:
            st.error("No .pth files found in repo root.")
            st.stop()
        pth_names = [os.path.basename(p) for p in pth_files]
        selected_pth = st.selectbox("Model weights", pth_names, index=0)
        pth_path = pth_files[pth_names.index(selected_pth)]

        st.divider()
        st.markdown("##### 📊 Dataset Summary")
        st.metric("Total Test Tasks", f"{len(all_task_ids):,}")
        st.metric("Unique Translators", f"{len(translator_pool):,}")
        st.metric("Tower A dim (Translator)", tower_a_dim)
        st.metric("Tower B dim (Task)", tower_b_dim)

    model = load_model(pth_path, tower_a_dim, tower_b_dim)

    tab1, tab2 = st.tabs(["🔍 Per-Task Inspector", "📊 Global Analytics"])

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
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("True Translator", result["true_translator"])
                    rc2.metric("Rank of True Translator",
                               f"#{result['rank']}" if result["rank"] else "Not in pool")
                    rc3.metric("Hit @ 3 / Hit @ 5",
                               f"{'✅' if result['hit_at_3'] else '❌'} / {'✅' if result['hit_at_5'] else '❌'}")
                    rc4.metric("Reciprocal Rank", f"{result['reciprocal_rank']:.4f}")
                else:
                    st.warning("⚠️ No ground-truth translator found for this task.")

                st.markdown("#### 📋 Top 10 Ranked Translators")
                top10 = result["ranked_df"].head(10).copy()
                if not top10.empty:
                    top10["Score"] = top10["Score"].map(lambda s: f"{s:.6f}")
                    true_t = result["true_translator"]
                    if true_t and true_t in top10["TRANSLATOR"].values:
                        styled = top10.style.apply(highlight_true_translator, true_translator=true_t, axis=1)
                        st.dataframe(styled, use_container_width=True)
                        st.info(f"🟡 The **highlighted row** is the true historical translator: **{true_t}**")
                    else:
                        st.dataframe(top10, use_container_width=True)
                        if true_t and result["rank"]:
                            st.info(f"ℹ️ True translator **{true_t}** is ranked **#{result['rank']}** (outside Top 10).")
                        elif true_t:
                            st.warning(f"⚠️ True translator **{true_t}** was removed by the hard filter.")
                else:
                    st.warning("No candidates survived the hard constraints filter.")

                with st.expander("📄 Show full ranking table"):
                    st.dataframe(result["ranked_df"], use_container_width=True, height=400)

    # ── TAB 2: Global Analytics ──
    with tab2:
        st.subheader("📊 Global Analytics")
        st.caption("Evaluate the first **500 tasks** in the test set, collect ranks "
                   "and affinity scores, then visualise the algorithm's efficiency.")
        GLOBAL_EVAL_SIZE = 500
        run_global = st.button("🚀 Run Full Global Evaluation", use_container_width=True, key="global_eval_btn")

        if run_global:
            first_500_ids = all_task_ids[:GLOBAL_EVAL_SIZE]
            progress = st.progress(0, text="Evaluating tasks …")
            rank_list, all_affinity, rr_list = [], [], []
            hits = {1: 0, 3: 0, 5: 0, 10: 0}
            evaluated, skipped = 0, 0

            for i, tid in enumerate(first_500_ids):
                res = evaluate_task(tid, tasks_df, translator_pool, labels_df, model,
                                    task_feature_cols, translator_feature_cols,
                                    df_raw, pairs_df, translators_data_df, schedules_df)
                if res is None or res["rank"] is None:
                    skipped += 1
                else:
                    rank_list.append(res["rank"]); rr_list.append(res["reciprocal_rank"])
                    hits[1] += int(res["hit_at_1"]); hits[3] += int(res["hit_at_3"])
                    hits[5] += int(res["hit_at_5"]); hits[10] += int(res["hit_at_10"])
                    all_affinity.extend(res["all_scores"]); evaluated += 1
                progress.progress((i+1)/GLOBAL_EVAL_SIZE, text=f"Evaluated {i+1}/{GLOBAL_EVAL_SIZE} tasks …")

            progress.empty()
            if evaluated == 0:
                st.warning("⚠️ No tasks could be evaluated.")
            else:
                hr = {f"@{k}": (v/evaluated)*100 for k,v in hits.items()}
                mrr = float(np.mean(rr_list))

                st.markdown("#### 🏅 Aggregate Metrics")
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Hit Rate @1", f"{hr['@1']:.2f} %")
                m2.metric("Hit Rate @3", f"{hr['@3']:.2f} %")
                m3.metric("Hit Rate @5", f"{hr['@5']:.2f} %")
                m4.metric("Hit Rate @10", f"{hr['@10']:.2f} %")
                m5.metric("MRR", f"{mrr:.4f}")
                st.success(f"✅ Evaluated **{evaluated}** tasks (skipped {skipped}). "
                           f"**MRR = {mrr:.4f}** · **Hit@3 = {hr['@3']:.2f} %** · **Hit@10 = {hr['@10']:.2f} %**")
                st.divider()

                st.markdown("#### 📈 Chart 1 — Hit Rate Progression")
                hr_df = pd.DataFrame({"Cutoff": ["@1","@3","@5","@10"],
                                      "Hit Rate (%)": [hr["@1"],hr["@3"],hr["@5"],hr["@10"]]})
                fig_hr = px.bar(hr_df, x="Cutoff", y="Hit Rate (%)", text="Hit Rate (%)", color="Cutoff",
                                color_discrete_sequence=["#667eea","#764ba2","#f093fb","#4fd1c5"],
                                title="Hit Rate at Various K Cutoffs")
                fig_hr.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
                fig_hr.update_layout(yaxis_range=[0, max(hr.values())*1.25+1],
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#e2e8f0"), title_font_size=18,
                    showlegend=False, yaxis=dict(gridcolor="rgba(100,100,255,0.08)"),
                    xaxis=dict(title=""), margin=dict(t=60,b=40))
                st.plotly_chart(fig_hr, use_container_width=True)

                st.markdown("#### 📊 Chart 2 — True Translator Rank Distribution")
                fig_rank = px.histogram(pd.DataFrame({"Rank": rank_list}), x="Rank",
                    nbins=min(50, max(rank_list)),
                    title="Distribution of the True Translator's Rank Position",
                    color_discrete_sequence=["#764ba2"])
                fig_rank.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#e2e8f0"), title_font_size=18,
                    yaxis=dict(title="Count of Tasks", gridcolor="rgba(100,100,255,0.08)"),
                    xaxis=dict(title="Rank Position"), bargap=0.05, margin=dict(t=60,b=40))
                st.plotly_chart(fig_rank, use_container_width=True)

                st.markdown("#### 🎯 Chart 3 — Affinity Score Distribution")
                fig_score = px.histogram(pd.DataFrame({"Score": all_affinity}), x="Score", nbins=80,
                    title="Distribution of All Generated Affinity Scores",
                    color_discrete_sequence=["#4fd1c5"])
                fig_score.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#e2e8f0"), title_font_size=18,
                    yaxis=dict(title="Count", gridcolor="rgba(100,100,255,0.08)"),
                    xaxis=dict(title="Affinity Score (0–1)"), bargap=0.02, margin=dict(t=60,b=40))
                st.plotly_chart(fig_score, use_container_width=True)

if __name__ == "__main__":
    main()

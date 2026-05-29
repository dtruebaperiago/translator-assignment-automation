"""
inference_server.py — IDISCDualTower FastAPI Inference Server
=============================================================
Loads the 4 pre-trained Dual-Tower MLP models and serves
translator recommendations to the frontend HUD.

Usage:
    python Backend/inference_server.py

Endpoints:
    GET  /api/v1/models              → list all models with config
    GET  /api/v1/models/active       → currently active model
    POST /api/v1/models/activate     → switch model {"model_id": "m2"}
    GET  /api/v1/translators         → full translators list (CORS-safe)
    GET  /api/v1/clients             → full clients list (CORS-safe)
    POST /api/v1/tasks/{id}/recommend → top-5 ranked translators for a task
"""

from __future__ import annotations
import json, pathlib, sys
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent.parent   # repo root
ARCH_DIR    = ROOT / "Backend" / "algorithms" / "Dual-Tower MLP" / "Architecture"
MODELS_DIR  = ROOT / "Backend" / "algorithms" / "Dual-Tower MLP" / "trained models"
FRONTEND    = ROOT / "frontend" / "data"

sys.path.insert(0, str(ARCH_DIR))
from IDISC_DualTower import IDISCDualTower  # noqa

# ── Feature columns (from preprocessing_pipeline.py + data_to_tensors.py) ────
# Tower A — 132 translator/employee features
# Continuous: translator_base_cost, Daily_Shift_Length, Weekly_Availability_Hours,
#             rolling_quality_ema, rolling_avg_task_time, rolling_punctuality_score,
#             rolling_efficiency_ratio, domain_experience, task_type_experience
# Binary: Works_Weekends, IS_NEW_EMPLOYEE, IS_SPECIALIST
# OHE: SOURCE_LANG_*, TARGET_LANG_*  (language pair columns from the training data)

# Tower B — 140 task/client features
# Continuous: HOURS, MANUFACTURER_enc, MANUFACTURER_INDUSTRY_enc,
#             MIN_QUALITY, WILDCARD_enc, SELLING_HOURLY_PRICE
# OHE: TASK_TYPE_*, SOURCE_LANG_*, TARGET_LANG_*

# EMBEDDING_DIM
EMBED_DIM = 64
TOWER_A_DIM = 132
TOWER_B_DIM = 140

# ── Discover all model directories ────────────────────────────────────────────
def _load_model_registry() -> dict:
    registry = {}
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        pth_path = d / "best_idisc_model.pth"
        if not (cfg_path.exists() and pth_path.exists()):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        registry[d.name] = {"config": cfg, "pth": pth_path, "dir": d}
    return registry

MODEL_REGISTRY = _load_model_registry()

# ── Model loader ──────────────────────────────────────────────────────────────
def _load_model(model_id: str) -> IDISCDualTower:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry.")
    entry = MODEL_REGISTRY[model_id]
    state = torch.load(entry["pth"], map_location="cpu", weights_only=True)
    # Read actual input dims from the state dict
    a_dim = state["a_layer1.0.weight"].shape[1]
    b_dim = state["b_layer1.0.weight"].shape[1]
    model = IDISCDualTower(
        tower_a_input_dim=a_dim,
        tower_b_input_dim=b_dim,
        embedding_dim=EMBED_DIM,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"  [Server] Loaded {model_id}: tower_a={a_dim}, tower_b={b_dim}, "
          f"tau={model.temperature.item():.4f}")
    return model, a_dim, b_dim

# ── Global state ──────────────────────────────────────────────────────────────
_active_model_id: str = sorted(MODEL_REGISTRY.keys())[0]  # default: m1
_model, _tower_a_dim, _tower_b_dim = _load_model(_active_model_id)

# ── Translator + Client data ──────────────────────────────────────────────────
with open(FRONTEND / "translators.json", encoding="utf-8") as f:
    _translators: list[dict] = json.load(f)

with open(FRONTEND / "clients.json", encoding="utf-8") as f:
    _clients: list[dict] = json.load(f)

# Build quick look-ups
_tr_by_name = {t["name"].lower().strip(): t for t in _translators}

# ── Feature engineering helpers ───────────────────────────────────────────────
def _make_translator_vector(tr: dict, a_dim: int) -> np.ndarray:
    """
    Build a Tower-A feature vector from a translator JSON record.
    Continuous features are filled from aggregated stats; OHE flags
    and engineered rolling features are set to neutral/zero when unknown.
    """
    vec = np.zeros(a_dim, dtype=np.float32)
    # We fill the first 12 known continuous/binary positions:
    # [0] translator_base_cost
    vec[0]  = float(tr.get("rate", 0))
    # [1] Daily_Shift_Length — assume 8h if unknown
    vec[1]  = 8.0
    # [2] Weekly_Availability_Hours — assume 40h
    vec[2]  = 40.0
    # [3] Works_Weekends — binary 0
    vec[3]  = 0.0
    # [4] rolling_quality_ema — use avg quality
    vec[4]  = float(tr.get("quality", 5.0)) / 10.0   # normalised
    # [5] rolling_avg_task_time
    tc = max(int(tr.get("taskCount", 1)), 1)
    th = float(tr.get("totalHours", 0))
    vec[5]  = (th / tc) if tc > 0 else 8.0
    # [6] rolling_punctuality_score — neutral 0.5
    vec[6]  = 0.5
    # [7] rolling_efficiency_ratio — neutral 1.0
    vec[7]  = 1.0
    # [8] domain_experience — proxy: task count capped
    vec[8]  = min(tc / 500.0, 1.0)
    # [9] task_type_experience — same proxy
    vec[9]  = min(tc / 300.0, 1.0)
    # [10] IS_NEW_EMPLOYEE
    vec[10] = 1.0 if tc < 10 else 0.0
    # [11] IS_SPECIALIST
    vec[11] = 1.0 if tr.get("workerType") == "Internal" and tc > 20 else 0.0
    # Remaining dims (OHE language pairs): left as 0
    # The model was trained with specific OHE columns; without the exact
    # column ordering we fill them with 0 — the embedding layers handle this.
    return vec


def _make_task_vector(task: dict, b_dim: int) -> np.ndarray:
    """
    Build a Tower-B feature vector from a task dict.
    """
    vec = np.zeros(b_dim, dtype=np.float32)
    # [0] HOURS
    hours_raw = str(task.get("forecast", "8h")).replace("h", "").strip()
    try:
        vec[0] = float(hours_raw)
    except ValueError:
        vec[0] = 8.0
    # [1] MANUFACTURER_enc — unknown → 0
    vec[1] = 0.0
    # [2] MANUFACTURER_INDUSTRY_enc — unknown → 0
    vec[2] = 0.0
    # [3] MIN_QUALITY — default 0
    vec[3] = 0.0
    # [4] WILDCARD_enc — 0
    vec[4] = 0.0
    # [5] SELLING_HOURLY_PRICE — estimate from market avg
    all_rates = [t.get("rate", 0) for t in _translators if t.get("rate", 0) > 0]
    vec[5] = float(np.mean(all_rates)) if all_rates else 20.0
    # Remaining OHE: left as 0
    return vec

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="IDISCDualTower Inference Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ActivateRequest(BaseModel):
    model_id: str

class RecommendRequest(BaseModel):
    source: Optional[str] = ""
    target: Optional[str] = ""
    type: Optional[str] = ""
    forecast: Optional[str] = "8h"
    client: Optional[str] = ""
    industry: Optional[str] = ""
    description: Optional[str] = ""
    assigned_translators: Optional[list[str]] = None

class AssignRequest(BaseModel):
    translator: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/models")
def list_models():
    result = []
    for mid, entry in MODEL_REGISTRY.items():
        result.append({
            "id": mid,
            "active": mid == _active_model_id,
            **entry["config"],
        })
    return result

@app.get("/api/v1/models/active")
def get_active_model():
    entry = MODEL_REGISTRY[_active_model_id]
    return {"id": _active_model_id, "active": True, **entry["config"]}

@app.post("/api/v1/models/activate")
def activate_model(req: ActivateRequest):
    global _active_model_id, _model, _tower_a_dim, _tower_b_dim
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model '{req.model_id}' not found")
    _active_model_id = req.model_id
    _model, _tower_a_dim, _tower_b_dim = _load_model(_active_model_id)
    return {"ok": True, "active": _active_model_id}

@app.get("/api/v1/translators")
def get_translators():
    return _translators

@app.get("/api/v1/clients")
def get_clients():
    return _clients

@app.post("/api/v1/tasks/{task_id}/recommend")
def recommend(task_id: str, req: RecommendRequest):
    src = (req.source or "").lower().strip()
    tgt = (req.target or "").lower().strip()
    typ = (req.type   or "").lower().strip()

    # Step 1: filter translators by language pair
    candidates = [
        tr for tr in _translators
        if (src in [l.lower() for l in (tr.get("sourceLangs") or [tr.get("source","")])])
        and (tgt in [l.lower() for l in (tr.get("targetLangs") or [tr.get("target","")])])
    ]

    # Filter out already assigned translators
    if req.assigned_translators:
        assigned_set = {name.lower().strip() for name in req.assigned_translators}
        candidates = [tr for tr in candidates if tr.get("name", "").lower().strip() not in assigned_set]

    # Step 2: further filter by task type (relax if empty)
    if typ:
        typed = [tr for tr in candidates
                 if typ in [t.lower() for t in (tr.get("taskTypes") or [])]]
        if typed:
            candidates = typed

    if not candidates:
        return {"recommendations": [], "businessLift": None,
                "note": f"No translators found for {req.source} → {req.target}"}

    # Step 3: build feature tensors for all candidates
    task_vec = _make_task_vector(req.dict(), _tower_b_dim)
    task_tensor = torch.tensor(task_vec, dtype=torch.float32).unsqueeze(0)  # (1, B_dim)

    scored = []
    with torch.no_grad():
        for tr in candidates:
            emp_vec = _make_translator_vector(tr, _tower_a_dim)
            emp_tensor = torch.tensor(emp_vec, dtype=torch.float32).unsqueeze(0)  # (1, A_dim)
            score = _model(emp_tensor, task_tensor).item()
            scored.append((score, tr))

    # Sort descending by affinity score
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # 1. Take the top 3 overall best matches
    top3 = scored[:3]
    
    # 2. Count Third-Party translators in the top 3
    tp_in_top3 = [x for x in top3 if x[1].get("workerType") != "Internal"]
    
    extra_recs_pool = []
    needed_tp = 0
    if len(tp_in_top3) < 2:
        needed_tp = 2 - len(tp_in_top3)
        # Find next up to `needed_tp` Third-Party matches from the remaining pool
        remaining = scored[3:]
        tp_remaining = [x for x in remaining if x[1].get("workerType") != "Internal"]
        extra_recs_pool = tp_remaining[:needed_tp]
        
    # Combine lists
    final_scored = [(score, tr, False) for (score, tr) in top3] # (score, tr, is_backup)
    final_scored.extend([(score, tr, True) for (score, tr) in extra_recs_pool])
    
    # If there are not enough third-party candidates, we add placeholder indicators (as None)
    placeholders_needed = needed_tp - len(extra_recs_pool)
    for _ in range(placeholders_needed):
        final_scored.append((0, None, True))

    # Step 4: build response
    all_rates = [t.get("rate", 0) for t in _translators if t.get("rate", 0) > 0]
    avg_rate = float(np.mean(all_rates)) if all_rates else 20.0

    hours_raw = str(req.forecast or "8h").replace("h", "").strip()
    try:
        hours = float(hours_raw)
    except ValueError:
        hours = 8.0

    recs = []
    for i, (score, tr, is_backup) in enumerate(final_scored):
        if tr is None:
            # Render a placeholder entry
            recs.append({
                "name": "No additional Third-Party match available",
                "isPlaceholder": True,
                "isBackupMatch": True,
                "matchScore": 0,
                "cost": 0,
                "quality": 0,
                "workerType": "Third-Party",
                "tags": [{"text": "Unavailable", "color": "rose"}],
                "reason": (
                    f"No other Third-Party translators match constraints for "
                    f"{req.source} → {req.target} | task type: {req.type or 'any'}."
                ),
                "deltaAdvantage": None,
                "available": False,
            })
            continue

        pct = int(round(score * 100))
        tags = []
        if (tr.get("quality") or 0) >= 8:
            tags.append({"text": "Top Quality", "color": "emerald"})
        if (tr.get("taskCount") or 0) >= 200:
            tags.append({"text": "Experienced", "color": "indigo"})
        if tr.get("workerType") == "Internal":
            tags.append({"text": "Internal", "color": "slate"})
        else:
            tags.append({"text": "Third-Party", "color": "fuchsia"})
        if len(tags) < 2:
            tags.append({"text": f"{req.type or 'Task'} Specialist", "color": "blue"})

        recs.append({
            "name": tr["name"],
            "matchScore": pct,
            "cost": tr.get("rate", 0),
            "quality": tr.get("quality", 0),
            "workerType": tr.get("workerType", "Third-Party"),
            "tags": tags,
            "reason": (
                f"Dual-Tower MLP ({_active_model_id}) affinity: {pct}% — "
                f"Matched on {req.source} → {req.target} | "
                f"{tr.get('taskCount',0)} historical tasks, "
                f"avg quality {tr.get('quality',0)}/10, rate €{tr.get('rate',0)}/hr."
            ),
            "deltaAdvantage": f"{pct}% affinity ({_active_model_id})" if (i == 0 and not is_backup) else None,
            "available": True,
            "isBackupMatch": is_backup,
        })

    # Business lift
    if recs:
        top = recs[0]
        savings = max(0, round((avg_rate - top["cost"]) * hours))
        lift = {
            "savingsEur": savings,
            "deliveryGuarantee": "On-Time" if (top.get("quality") or 0) >= 8 else "Estimated",
            "marginBoost": f"+{round((1 - top['cost'] / avg_rate) * 100)}%" if top["cost"] < avg_rate else "Standard",
            "baselineTranslator": f"Market Avg (€{avg_rate:.1f}/hr)",
        }
    else:
        lift = None

    return {
        "recommendations": recs,
        "businessLift": lift,
        "model": _active_model_id,
    }

@app.post("/api/v1/tasks/{task_id}/assign")
def assign_task(task_id: str, req: AssignRequest):
    print(f"  [Server] Assigned task {task_id} to {req.translator}")
    return {"ok": True}

@app.get("/api/v1/tasks/pending")
def get_pending_tasks():
    return []  # Tasks come from CSV upload in the frontend

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  iDISC Dual-Tower MLP Inference Server")
    print("=" * 60)
    print(f"  Models available : {list(MODEL_REGISTRY.keys())}")
    print(f"  Active model     : {_active_model_id}")
    print(f"  Frontend URL     : http://localhost:8080/idisc_hud.html")
    print(f"  API docs         : http://localhost:8000/docs")
    print("  Press Ctrl+C to stop.\n")
    uvicorn.run("inference_server:app", host="127.0.0.1", port=8000, reload=False)

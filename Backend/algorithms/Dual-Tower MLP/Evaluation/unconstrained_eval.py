import sys
import os
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Handle paths based on being in the Evaluation directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))

sys.path.append(os.path.join(root_dir, 'Backend/algorithms/Dual-Tower MLP/Architecture'))
from IDISC_DualTower import IDISCDualTower

DATA_DIR = os.path.join(root_dir, 'DATA/Processed')

tasks_df = pd.read_csv(os.path.join(DATA_DIR, "test_tasks.csv"))
translators_df = pd.read_csv(os.path.join(DATA_DIR, "test_translators.csv"))
labels_df = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))

task_feature_cols = [c for c in tasks_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
translator_feature_cols = [c for c in translators_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
tower_b_dim = len(task_feature_cols)
tower_a_dim = len(translator_feature_cols)

# Slicing the last 50 candidates
translator_pool = translators_df.groupby("TRANSLATOR")[translator_feature_cols].mean().reset_index()
last_50_names = translator_pool["TRANSLATOR"].values[-50:]
translator_pool_np = translator_pool[translator_feature_cols].values.astype(np.float32)[-50:]

pth_path = os.path.join(root_dir, 'Backend/algorithms/Dual-Tower MLP/trained models/m1/best_idisc_model.pth')

model = IDISCDualTower(tower_a_input_dim=tower_a_dim, tower_b_input_dim=tower_b_dim, embedding_dim=64)
model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
model.eval()

all_task_ids = labels_df["TASK_ID"].unique()
eval_size = min(500, len(all_task_ids))
first_ids = all_task_ids[:eval_size]

all_scores = []
BATCH_SIZE = 64
n_batches = math.ceil(len(translator_pool_np) / BATCH_SIZE)

print(f"Evaluating {eval_size} tasks against {len(translator_pool_np)} candidates...")

with torch.no_grad():
    for i, tid in enumerate(first_ids):
        task_row = tasks_df[tasks_df["TASK_ID"] == tid]
        if task_row.empty: continue
        task_feat = task_row[task_feature_cols].values.astype(np.float32)
        task_tensor = torch.tensor(task_feat)
        
        for b in range(n_batches):
            s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(translator_pool_np))
            trans_batch = torch.tensor(translator_pool_np[s:e])
            task_batch = task_tensor.expand(e - s, -1)
            scores = model(trans_batch, task_batch)
            all_scores.extend(scores.squeeze(-1).cpu().numpy().tolist())

# Get true scores for these 50 translators
true_scores = labels_df[labels_df["TRANSLATOR"].isin(last_50_names)]["AFFINITY_LABEL"].values

# Plotting side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(all_scores, bins=80, color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0].set_title('Predicted Scores (Unconstrained, Last 50 Candidates)')
axes[0].set_xlabel('Predicted Affinity Score')
axes[0].set_ylabel('Frequency')

axes[1].hist(true_scores, bins=80, color='#3498db', edgecolor='black', alpha=0.8)
axes[1].set_title('True Labels (Last 50 Candidates in test_labels.csv)')
axes[1].set_xlabel('True Affinity Score')
axes[1].set_ylabel('Frequency')

plt.tight_layout()

artifact_dir = r"C:\Users\Adriv\.gemini\antigravity\brain\93c5a208-b39e-41b6-a9b4-c90ba5e9fbf9\artifacts"
os.makedirs(artifact_dir, exist_ok=True)
out_path = os.path.join(artifact_dir, "last50_predicted_vs_true.png")
plt.savefig(out_path, bbox_inches='tight')
print(f"Plot saved to {out_path}")

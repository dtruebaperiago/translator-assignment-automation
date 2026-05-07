import sys
import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 1. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Processed")

# Add the directory to sys.path so we can import the model
sys.path.append(os.path.dirname(BASE_DIR))
from IDISC_DualTower import IDISCDualTower

# 2. Load Data
tasks_df = pd.read_csv(os.path.join(DATA_DIR, "test_tasks.csv"))
translators_df = pd.read_csv(os.path.join(DATA_DIR, "test_translators.csv"))
labels_df = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))

task_feature_cols = [c for c in tasks_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
translator_feature_cols = [c for c in translators_df.columns if c not in ("TASK_ID", "TRANSLATOR")]
tower_b_dim = len(task_feature_cols)
tower_a_dim = len(translator_feature_cols)

translator_pool = translators_df.groupby("TRANSLATOR")[translator_feature_cols].mean().reset_index()
translator_pool_np = translator_pool[translator_feature_cols].values.astype(np.float32)

# 3. Load Model
pth_files = sorted(glob.glob(os.path.join(BASE_DIR, "*.pth")))
if not pth_files:
    print("No .pth files found in", BASE_DIR)
    sys.exit(1)
pth_path = pth_files[0]
print(f"Loading model from {pth_path}")

model = IDISCDualTower(tower_a_input_dim=tower_a_dim, tower_b_input_dim=tower_b_dim, embedding_dim=64)
model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
model.eval()

# 4. Run Predictions
all_task_ids = labels_df["TASK_ID"].unique()
eval_size = min(500, len(all_task_ids))
first_ids = all_task_ids[:eval_size]

all_scores = []
BATCH_SIZE = 64
n_batches = math.ceil(len(translator_pool_np) / BATCH_SIZE)

print(f"Evaluating {eval_size} tasks against {len(translator_pool_np)} candidates (total combinations: {eval_size * len(translator_pool_np)})...")

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

# 5. Plot and save
print(f"Generated {len(all_scores)} predictions.")
plt.figure(figsize=(10, 6))
pd.Series(all_scores).plot.hist(bins=80, color='#ff9f43', edgecolor='black')
plt.title('Predicted Affinity Scores Distribution (Unconstrained)')
plt.xlabel('Predicted Affinity Score')
plt.ylabel('Frequency')

artifact_dir = r"C:\Users\Adriv\.gemini\antigravity\brain\93c5a208-b39e-41b6-a9b4-c90ba5e9fbf9\artifacts"
os.makedirs(artifact_dir, exist_ok=True)
out_path = os.path.join(artifact_dir, "unconstrained_predicted_affinity.png")
plt.savefig(out_path, bbox_inches='tight')
print(f"Plot saved to {out_path}")

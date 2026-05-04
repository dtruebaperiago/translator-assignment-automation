# IDISCDualTower — Dual-Tower Recommender System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Architecture%20Complete-brightgreen)
![Role](https://img.shields.io/badge/Team%20Role-The%20Architect-blueviolet)
![License](https://img.shields.io/badge/License-Internal%20Use-lightgrey)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
   - [The Two Towers](#21-the-two-towers)
   - [ResNet-Style Skip Connection](#22-resnet-style-skip-connection)
   - [Scoring Mechanism](#23-scoring-mechanism)
   - [The Temperature Parameter](#24-the-temperature-parameter--the-secret-weapon)
   - [Layer-by-Layer Summary](#25-layer-by-layer-summary)
3. [Team Pipeline](#3-team-pipeline)
4. [Installation & Requirements](#4-installation--requirements)
5. [Quick Start](#5-quick-start)
6. [Interpretation & Visualization Hooks](#6-interpretation--visualization-hooks)
7. [Design Decisions & Rationale](#7-design-decisions--rationale)
8. [Known Constraints](#8-known-constraints)

---

## 1. Overview

**IDISCDualTower** is a production-grade deep learning model built with PyTorch that solves the core matching problem of the IDISC Translation Task Assignment System: *given a Translation Employee and a Translation Task, how compatible are they?*

The model outputs a single continuous **Affinity Score** in the range `[0.0, 1.0]` for every Employee–Task pair in a batch, where `1.0` means perfect compatibility and `0.0` means no compatibility. This score can be used directly to rank, filter, or assign tasks to the most suitable translators.

### What problem does this solve?

Manual task assignment in translation teams is slow, subjective, and doesn't scale. This model replaces or augments that process by learning, from data, what makes a translator a good fit for a task — accounting for language pair, domain expertise, availability, CAT tool proficiency, word count, urgency, and any other tabular features surfaced by the data team.

### Why a Dual-Tower architecture?

A Dual-Tower (also called *Two-Tower*) model is the industry-standard approach for large-scale retrieval and matching problems (used at Google, Meta, and Spotify). The core insight is:

> **Both entities are encoded independently into the same embedding space, then compared.**

This means at inference time, Task embeddings can be pre-computed and cached. Only the Employee embedding needs to be computed per query — making real-time assignment dramatically efficient.

---

## 2. Architecture Deep Dive

### 2.1 The Two Towers

The model takes two independent batches of tabular feature vectors as input:

| Input | Description | Feature Dimensions |
|---|---|---|
| `tower_a_x` | Translation Employee features (skills, language pairs, availability, etc.) | **25** |
| `tower_b_x` | Translation Task features (word count, domain, urgency, CAT tool, etc.) | **18** |

Each tower is an independent deep MLP (Multi-Layer Perceptron) that transforms its raw input features into a **64-dimensional embedding vector** in a shared latent space. The towers have identical *architectures* but entirely independent *weights* — this is crucial: it allows each tower to develop a specialised internal representation before the two sides are ever compared.

```
  Employee Features          Task Features
  (batch_size, 25)           (batch_size, 18)
        │                          │
  ┌─────▼──────┐            ┌──────▼──────┐
  │  TOWER A   │            │   TOWER B   │
  │ (Employee) │            │   (Task)    │
  └─────┬──────┘            └──────┬──────┘
        │ emb_a (B, 64)            │ emb_b (B, 64)
        └──────────┬───────────────┘
                   │
          Cosine Similarity
                   │
          × Temperature τ
                   │
             Sigmoid(·)
                   │
          affinity_score (B, 1)
          ∈ [0.0, 1.0]
```

### 2.2 ResNet-Style Skip Connection

Each tower does **not** use a plain sequential stack. Instead, I designed a ResNet-style (He et al., 2015) architecture with an explicit skip (residual) connection across the middle block:

```
Input (e.g. 25-dim)
    │
    ▼
[layer1]  Linear(input_dim → 256) → BatchNorm1d → GELU → Dropout(0.2)
    │ x1
    ├──────────────────────────────────────────┐  ← skip connection
    ▼                                          │
[layer2]  Linear(256 → 256) → BatchNorm1d → GELU → Dropout(0.2)
    │ x2                                       │
    └──────────── x_res = x1 + x2 ────────────┘
                       │
                       ▼
[layer3]  Linear(256 → 64)   ← projection head (no activation)
                       │
                  emb  (B, 64)
```

**Why the skip connection?**
The residual connection `x_res = x1 + x2` forces `layer2` to learn only the *delta* (residual) from `x1`, rather than a full transformation. This has two practical benefits:

1. **Gradient highway**: Gradients can flow directly from `layer3` back to `layer1` via the skip path, preventing vanishing gradients.
2. **Easier optimisation**: If `layer2` is not needed, it can learn to output near-zero, and the model gracefully degrades to a shallower network. This makes training more stable.

### 2.3 Scoring Mechanism

Once both towers produce their 64-D embedding vectors `emb_a` and `emb_b`, the similarity between the two is computed using **Cosine Similarity**:

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

Cosine Similarity is **magnitude-invariant** — it measures only the *angle* between the two vectors in the 64-D space. This is ideal because:

- It is bounded in `[-1, 1]`, which pairs naturally with a sigmoid.
- It is not affected by the scale of the embedding vectors, only their direction.
- It is the natural distance metric in embedding spaces learned via contrastive or similarity objectives.

The raw similarity is then passed through a sigmoid to bound the score to `(0.0, 1.0)`:

$$\text{affinity\_score} = \sigma\!\left(\tau \cdot \text{cos\_sim}(\mathbf{emb\_a},\, \mathbf{emb\_b})\right)$$

where `σ` is the sigmoid function and `τ` (tau) is the learnable temperature.

### 2.4 The Temperature Parameter — The Secret Weapon

```python
self.temperature = nn.Parameter(torch.tensor(1.0))
```

The **learnable temperature** `τ` is a single scalar parameter registered with `nn.Parameter`, meaning PyTorch's autograd tracks it and the optimiser updates it during training — just like any weight in a linear layer.

Its effect is intuitive:

| Temperature τ | Effect on scores |
|---|---|
| `τ = 1.0` *(initial)* | Baseline — standard sigmoid of cosine sim |
| `τ >> 1.0` *(sharpened)* | Scores pushed toward 0 and 1; model becomes more decisive |
| `τ << 1.0` *(softened)* | Scores cluster near 0.5; model expresses more uncertainty |

This is conceptually identical to the temperature used in softmax during knowledge distillation and in contrastive learning frameworks like CLIP (Radford et al., 2021). By making it learnable, the model discovers the optimal sharpness for the score distribution automatically — no manual tuning required.

### 2.5 Layer-by-Layer Summary

The full parameter count for a configuration of `tower_a_input_dim=25`, `tower_b_input_dim=18`, `embedding_dim=64`:

| Layer | Shape | Parameters |
|---|---|---|
| **Tower A** `a_layer1` — Linear | 25 → 256 | 6,656 |
| **Tower A** `a_layer1` — BatchNorm1d | 256 | 512 |
| **Tower A** `a_layer2` — Linear | 256 → 256 | 65,792 |
| **Tower A** `a_layer2` — BatchNorm1d | 256 | 512 |
| **Tower A** `a_layer3` — Linear | 256 → 64 | 16,448 |
| **Tower B** `b_layer1` — Linear | 18 → 256 | 4,864 |
| **Tower B** `b_layer1` — BatchNorm1d | 256 | 512 |
| **Tower B** `b_layer2` — Linear | 256 → 256 | 65,792 |
| **Tower B** `b_layer2` — BatchNorm1d | 256 | 512 |
| **Tower B** `b_layer3` — Linear | 256 → 64 | 16,448 |
| **Temperature** `τ` | scalar | 1 |
| **Total** | | **178,049** |

> **Memory footprint**: ~0.71 MB for parameters, ~1.84 MB estimated total size (params + activations for batch size 64).

---

## 3. Team Pipeline

The full system is divided into four roles, each building on the previous. The `IDISCDualTower` model is the output of **Role 2**, and it is designed to be consumed directly by Roles 3 and 4.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IDISC ML Pipeline                               │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Role 1      │  Role 2      │  Role 3      │  Role 4                │
│ The Plumber  │ The Architect│ The Trainer  │ The Judge              │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│ Raw CSV data │ IDISCDual    │ Training     │ Inference, ranking,    │
│ → feature    │ Tower model  │ loop,        │ UMAP clustering,       │
│ engineering  │ definition   │ loss fn,     │ embedding caching,     │
│ → tensors    │ (this file)  │ optimiser    │ Captum attribution     │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

### Role 1 — The Plumber

Responsible for loading raw translation data (employee profiles, task metadata), applying preprocessing (StandardScaler / MinMaxScaler), and serving `(tower_a_tensor, tower_b_tensor, label)` batches via a `DataLoader`. The model expects **normalised** float32 tensors — raw counts or categorical ordinals must be pre-processed before reaching the model.

### Role 2 — The Architect *(this file)*

The `IDISCDualTower` class in `IDISC_DualTower_Architecture.py`. Defines the full model graph, initialisation strategy, and the interpretation hook interface (`return_embeddings`). This role is **complete**.

### Role 3 — The Trainer

Consumes the model from Role 2. Responsible for:

- Choosing and implementing a loss function (e.g., `nn.BCELoss`, `nn.MSELoss`, or a contrastive loss).
- Setting up an `torch.optim.AdamW` optimiser (recommended; weight decay regularises all Linear layers).
- Implementing the epoch loop, gradient clipping, and learning rate scheduling.
- Checkpointing the model state via `torch.save(model.state_dict(), ...)`.

### Role 4 — The Judge

Consumes the trained model checkpoint. Responsible for:

- Loading the saved `state_dict` and running inference.
- Ranking all tasks for a given employee by their affinity score.
- Using the `return_embeddings=True` hook to extract 64-D vectors for UMAP projections and TensorBoard visualisation.
- Using **Captum** (`IntegratedGradients`) to attribute affinity scores back to input features for explainability.

---

## 4. Installation & Requirements

### System Requirements

| Component | Requirement |
|---|---|
| Python | 3.10 or higher |
| PyTorch | 2.0 or higher |
| Graphviz | System binary required for PNG graph export |

### Python Packages

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` at the project root covers the core dependencies. For the full analysis and explainability stack used by Roles 3 and 4, also install:

```bash
pip install captum umap-learn matplotlib
```

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Core deep learning framework |
| `torchinfo` | ≥ 1.8.0 | Layer summary table with shapes and param counts |
| `torchviz` | ≥ 0.0.2 | Autograd graph → PNG export |
| `graphviz` | ≥ 0.20 | Python wrapper for Graphviz binaries |
| `captum` | ≥ 0.7.0 | Feature attribution and explainability (Role 4) |
| `umap-learn` | ≥ 0.5.0 | Non-linear dimensionality reduction for embedding plots (Role 4) |
| `matplotlib` | ≥ 3.7.0 | Visualisation support for UMAP plots |

### Graphviz System Binary (for PNG graph export only)

The `torchviz` library requires the **Graphviz system program** (`dot`) — this is separate from the Python `graphviz` package.

- **Windows**: Download from [graphviz.org/download](https://graphviz.org/download/) and during installation tick **"Add Graphviz to PATH"**.
  - If PATH was not set automatically, add it permanently via Git Bash:

    ```bash
    echo 'export PATH="$PATH:/c/Program Files/Graphviz/bin"' >> ~/.bashrc && source ~/.bashrc
    ```

- **macOS**: `brew install graphviz`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install graphviz`

> The architecture graph is only generated by the `if __name__ == "__main__"` test block. The `IDISCDualTower` model class itself has no dependency on Graphviz at runtime.

---

## 5. Quick Start

### Instantiate and run a forward pass

```python
import torch
from IDISC_DualTower_Architecture import IDISCDualTower

# --- Instantiate the model ---
model = IDISCDualTower(
    tower_a_input_dim=25,   # number of employee features
    tower_b_input_dim=18,   # number of task features
    embedding_dim=64,       # shared latent space size (default)
)
model.eval()  # disable Dropout, fix BatchNorm stats for inference

# --- Create a batch of 8 employee–task pairs ---
employees = torch.randn(8, 25)  # shape: (batch_size, tower_a_input_dim)
tasks     = torch.randn(8, 18)  # shape: (batch_size, tower_b_input_dim)

# --- Standard inference: get affinity scores only ---
with torch.no_grad():
    scores = model(employees, tasks)

print(scores.shape)   # torch.Size([8, 1])
print(scores)         # tensor([[0.51], [0.49], ...]) — values in [0.0, 1.0]
```

### Run the built-in smoke test and architecture visualisation

```bash
# From the project root:
python "Backend/algorithms/Dual-Tower MLP/IDISC_DualTower_Architecture.py"
```

This will:

1. Run two forward passes (with and without embeddings) and assert output shapes.
2. Print a full `torchinfo` layer summary table.
3. Render and save `idisc_dual_tower_architecture.png` next to the script.

> **Windows PowerShell / Git Bash note**: `torchinfo` prints Unicode characters. Run with `PYTHONUTF8=1` set, or use:
>
> ```bash
> PYTHONUTF8=1 python "Backend/algorithms/Dual-Tower MLP/IDISC_DualTower_Architecture.py"
> ```

---

## 6. Interpretation & Visualization Hooks

The `forward` method exposes a `return_embeddings` flag specifically designed to support downstream analysis in Roles 3 and 4 — without modifying the core training pipeline.

### Activating the hook

```python
output = model(employees, tasks, return_embeddings=True)

# output is a dict with three keys:
affinity_score = output["affinity_score"]  # (B, 1)  — the bounded score
emb_a          = output["emb_a"]           # (B, 64) — employee embedding
emb_b          = output["emb_b"]           # (B, 64) — task embedding
```

### Use Case 1 — UMAP Cluster Analysis (Role 4)

Concatenate employee and task embeddings and reduce to 2D with UMAP to visually inspect whether the embedding space clusters by domain, language pair, or task type:

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

with torch.no_grad():
    out = model(employees, tasks, return_embeddings=True)

# Stack all 64-D vectors into one matrix for UMAP
all_embeddings = torch.cat([out["emb_a"], out["emb_b"]], dim=0).numpy()
labels = ["employee"] * len(employees) + ["task"] * len(tasks)

reducer = umap.UMAP(n_components=2, random_state=42)
coords  = reducer.fit_transform(all_embeddings)

plt.scatter(coords[:, 0], coords[:, 1], c=["blue" if l == "employee" else "red" for l in labels])
plt.title("Employee vs Task Embedding Space (UMAP)")
plt.savefig("umap_embedding_space.png", dpi=150)
```

### Use Case 2 — Embedding Caching for Fast Retrieval (Role 4)

Since Task embeddings do not change once the model is trained, they can be pre-computed and cached. At assignment time, only the Employee's embedding needs to be computed:

```python
# --- OFFLINE: pre-compute and cache all task embeddings ---
with torch.no_grad():
    task_cache = model(dummy_employees, all_tasks, return_embeddings=True)["emb_b"]
    # Save to disk for later
    torch.save(task_cache, "task_embedding_cache.pt")

# --- ONLINE: compute a single employee's embedding and rank against cache ---
cached_tasks = torch.load("task_embedding_cache.pt")
with torch.no_grad():
    out      = model(one_employee.expand(len(cached_tasks), -1), cached_tasks, return_embeddings=True)
    scores   = out["affinity_score"]
    top_task = scores.argmax()
```

### Use Case 3 — Feature Attribution with Captum (Role 4)

Use Captum's `IntegratedGradients` to understand which of the 25 employee features most influence a given affinity score:

```python
from captum.attr import IntegratedGradients

ig      = IntegratedGradients(lambda x: model(x, tasks))
attribs = ig.attribute(employees, target=0)   # shape: (B, 25)
# Higher absolute value → more influential feature for that prediction
```

---

## 7. Design Decisions & Rationale

| Decision | Why |
|---|---|
| **ResNet skip connection** | Prevents vanishing gradients; encourages each block to learn residual deltas rather than full transformations |
| **GELU over ReLU** | GELU is smoother and probabilistically motivated; empirically outperforms ReLU in deep MLPs and transformers |
| **BatchNorm1d in towers** | Stabilises training by normalising layer inputs; allows higher learning rates without divergence |
| **Dropout p=0.2** | Light regularisation; prevents co-adaptation of neurons without aggressively suppressing information flow |
| **Kaiming (He) Normal init** | Optimal for ReLU-family activations; preserves variance through deep forward passes from the first step |
| **No activation on layer3** | The projection head produces raw logits; cosine similarity implicitly normalises their magnitude, giving the model maximum representational freedom |
| **Cosine similarity over dot product** | Magnitude-invariant; naturally bounded in [-1, 1]; more numerically stable as embeddings grow in norm |
| **Learnable temperature τ** | Eliminates a critical hyperparameter; the model discovers the ideal score sharpness from the training data distribution |
| **`return_embeddings` hook** | Clean separation of concerns — training code never needs to change; analysis tools opt in without modifying the graph |
| **Independent tower weights** | Employees and tasks are heterogeneous entities; shared weights would force a misaligned representation |

---

## 8. Known Constraints

- **Tabular data only**: This model is designed exclusively for preprocessed tabular features. Raw text fields (e.g., task descriptions) must be embedded separately (e.g., via a Sentence Transformer) before being passed in as features.
- **Fixed input dimensions at init time**: `tower_a_input_dim` and `tower_b_input_dim` are set at construction. Adding new features requires re-instantiating and re-training the model.
- **Requires normalised inputs**: The model does not include an input normalisation layer. All features must be scaled (e.g., StandardScaler) before being passed in. Unnormalised inputs will cause BatchNorm instability and poor convergence.
- **BatchNorm in eval mode**: When running inference, always call `model.eval()` first. This fixes the BatchNorm running statistics to their trained values. Forgetting to do this is a common source of inconsistent inference results.
- **Temperature can go negative**: `τ` is unconstrained. If training with a BCE loss pushes it negative, affinity scores will invert. Monitor `model.temperature.item()` during training and consider clamping if needed.

---

*Architecture designed and implemented by the IDISC Translation Team — Role 2: The Architect.*

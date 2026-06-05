"""
IDISC Dual-Tower Recommender System Architecture
=================================================
Project:  Translation Task Assignment System (IDISC)
Module:   Backend / Algorithms / Dual-Tower MLP

This module implements the `IDISCDualTower` neural network, a production-grade
Dual-Tower (a.k.a. Two-Tower) deep learning model that computes a continuous
Affinity Score between a Translation Employee (Tower A) and a Translation Task
(Tower B).

Dependencies:
    torch, torchinfo, torchviz, graphviz
    # torchviz also requires Graphviz system binaries:
    # https://graphviz.org/download/
"""

# ---------------------------------------------------------------------------
# Standard & Third-Party Imports
# ---------------------------------------------------------------------------
from __future__ import annotations  # Enable PEP 563 postponed evaluation for type hints

import torch
import torch.nn as nn
import torch.nn.functional as F  # Required for cosine_similarity and activation utilities


# ---------------------------------------------------------------------------
# IDISCDualTower — Main Model Definition
# ---------------------------------------------------------------------------

class IDISCDualTower(nn.Module):
    """A production-grade Dual-Tower Recommender System for Translation Task Assignment.

    This model maps two heterogeneous feature spaces — Translation Employees (Tower A)
    and Translation Tasks (Tower B) — into a shared embedding space, then scores their
    compatibility via scaled cosine similarity.

    Architecture Overview:
        Each tower is a ResNet-style deep MLP:
            Input  →  [Linear(256) → BN → GELU → Dropout]         (layer1)
                   →  [Linear(256) → BN → GELU → Dropout] + Skip  (layer2, residual)
                   →  [Linear(embedding_dim)]                       (layer3, projection head)

    Args:
        tower_a_input_dim (int):
            Dimensionality of the Tower A (Employee) feature vector.
            Example: 25 features (skills, availability, language pairs, etc.)
        tower_b_input_dim (int):
            Dimensionality of the Tower B (Task) feature vector.
            Example: 18 features (word count, domain, urgency, CAT tool, etc.)
        embedding_dim (int):
            Size of the final shared embedding space that both towers project into.
            Defaults to 64. Lower values → more compression; higher values → richer space.

    Forward Input / Output:
        Inputs:
            tower_a_x  (torch.Tensor): Shape ``(batch_size, tower_a_input_dim)``
            tower_b_x  (torch.Tensor): Shape ``(batch_size, tower_b_input_dim)``
            return_embeddings (bool):  Hook flag for downstream analysis tools.

        Outputs (two modes):
            - ``return_embeddings=False`` (training / inference):
                Returns ``affinity_score`` of shape ``(batch_size, 1)``, a sigmoid-bounded
                probability in [0.0, 1.0] representing Employee–Task compatibility.

            - ``return_embeddings=True`` (analysis / explainability):
                Returns a dictionary with three keys:
                    ``'affinity_score'``: Tensor of shape ``(batch_size, 1)``
                    ``'emb_a'``         : Tensor of shape ``(batch_size, embedding_dim)``
                    ``'emb_b'``         : Tensor of shape ``(batch_size, embedding_dim)``
                These raw vectors are used by downstream tools (Role 3 / Role 4):
                    - UMAP projections for cluster analysis
                    - TensorBoard Projector for embedding visualization
                    - Captum for feature attribution / explainability

    Example:
        >>> model = IDISCDualTower(tower_a_input_dim=25, tower_b_input_dim=18)
        >>> emps  = torch.randn(64, 25)
        >>> tasks = torch.randn(64, 18)
        >>> score = model(emps, tasks)                    # shape: (64, 1)
        >>> out   = model(emps, tasks, return_embeddings=True)  # dict with 3 keys
    """

    def __init__(
        self,
        tower_a_input_dim: int,
        tower_b_input_dim: int,
        embedding_dim: int = 64,
    ) -> None:
        """Initializes all layers, parameters, and weights for both towers.

        Args:
            tower_a_input_dim (int): Feature dimensionality of the Employee tower.
            tower_b_input_dim (int): Feature dimensionality of the Task tower.
            embedding_dim (int): Projection space size shared by both towers. Default 64.
        """
        super().__init__()  # Initialize nn.Module base class

        # ----------------------------------------------------------------
        # Learnable Temperature Parameter
        # ----------------------------------------------------------------
        # Temperature τ scales the cosine similarity before the sigmoid.
        # Starting at 1.0, the model can learn to "sharpen" (τ > 1) or
        # "soften" (τ < 1) the score distribution during training.
        # Using nn.Parameter ensures it is updated during backpropagation.
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # ================================================================
        # TOWER A — Translation Employee Encoder
        # ================================================================
        # We define each layer block individually (not as nn.Sequential)
        # so we can wire the skip (residual) connection manually in forward().

        # --- Layer 1: Input projection → hidden space (256-dim) ---
        # Expands the raw feature vector into a richer representational space.
        self.a_layer1 = nn.Sequential(
            nn.Linear(tower_a_input_dim, 256),  # Learned linear projection
            nn.BatchNorm1d(256),                 # Normalize activations for stable training
            nn.GELU(),                           # GELU is preferred over ReLU in modern transformers/MLPs
            nn.Dropout(p=0.2),                   # Stochastic regularization to prevent overfitting
        )

        # --- Layer 2: Residual block (256 → 256, same dim to enable skip connection) ---
        # The output of this block is ADDED to layer1's output, forming a skip connection.
        # This helps gradients flow deeper without vanishing.
        self.a_layer2 = nn.Sequential(
            nn.Linear(256, 256),   # Must be 256→256 to match x1 shape for addition
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )

        # --- Layer 3: Projection head → shared embedding space ---
        # No activation here: the raw logits from this layer are L2-normalised
        # implicitly by cosine_similarity(), giving the model maximum flexibility.
        self.a_layer3 = nn.Linear(256, embedding_dim)

        # ================================================================
        # TOWER B — Translation Task Encoder
        # ================================================================
        # Identical architecture to Tower A but with its own independent weights.
        # Keeping towers separate allows each to develop specialised representations
        # for employees vs. tasks before they are compared in embedding space.

        self.b_layer1 = nn.Sequential(
            nn.Linear(tower_b_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )

        self.b_layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )

        self.b_layer3 = nn.Linear(256, embedding_dim)

        # ----------------------------------------------------------------
        # Weight Initialization
        # ----------------------------------------------------------------
        # Apply Kaiming initialisation to all layers immediately after definition.
        # This is best practice to ensure activations have unit variance at the
        # start of training, which dramatically speeds up convergence.
        self._init_weights()

    # ====================================================================
    # Weight Initialization — Kaiming / He Normal
    # ====================================================================

    def _init_weights(self) -> None:
        """Applies Kaiming Normal initialisation to all Linear and BatchNorm layers.

        Kaiming (He) Normal init is the recommended strategy for networks using
        ReLU-family activations (including GELU). It scales initial weights by
        sqrt(2 / fan_in), preventing exploding/vanishing gradients from the very
        first forward pass.

        BatchNorm layers are reset to identity (weight=1, bias=0), which is the
        standard starting point — the network learns the optimal scale/shift.
        """
        for module in self.modules():
            # Apply Kaiming init to all fully-connected layers
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",           # Scale by input fan; stable for forward pass
                    nonlinearity="relu",     # GELU ≈ ReLU for init purposes (no dedicated mode)
                )
                if module.bias is not None:
                    # Zero-initialize biases: the network starts as a near-linear function
                    nn.init.zeros_(module.bias)

            # Reset BatchNorm to identity transform at initialisation
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)   # γ = 1 → no scaling initially
                nn.init.zeros_(module.bias)    # β = 0 → no shift initially

    # ====================================================================
    # Forward Pass
    # ====================================================================

    def forward(
        self,
        tower_a_x: torch.Tensor,
        tower_b_x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the affinity score between an Employee and a Task batch.

        Args:
            tower_a_x (torch.Tensor): Employee feature matrix, shape ``(B, tower_a_input_dim)``.
            tower_b_x (torch.Tensor): Task feature matrix, shape ``(B, tower_b_input_dim)``.
            return_embeddings (bool): If True, returns raw embeddings for downstream analysis.
                Defaults to False (training / inference mode).

        Returns:
            torch.Tensor | dict:
                - If ``return_embeddings=False``:
                    ``affinity_score`` — Tensor of shape ``(B, 1)`` in range [0.0, 1.0].
                - If ``return_embeddings=True``:
                    Dict with keys ``'affinity_score'``, ``'emb_a'``, ``'emb_b'``.
        """

        # ================================================================
        # TOWER A FORWARD PASS — Encode the Employee features
        # ================================================================

        # Step A-1: Layer 1 — project raw features into the hidden 256-dim space
        a_x1 = self.a_layer1(tower_a_x)   # shape: (B, 256)

        # Step A-2: Layer 2 — apply the residual (bottleneck) transformation
        a_x2 = self.a_layer2(a_x1)        # shape: (B, 256)

        # Step A-3: Residual / Skip Connection
        # Adding x1 directly to x2 creates a "shortcut" path for gradients.
        # This prevents the vanishing gradient problem in deeper networks and
        # encourages each block to learn only the *residual* (delta), which is
        # empirically easier to optimise than learning from scratch.
        a_x_res = a_x1 + a_x2             # shape: (B, 256)

        # Step A-4: Layer 3 — project the residual output to the shared embedding space
        emb_a = self.a_layer3(a_x_res)    # shape: (B, embedding_dim)

        # ================================================================
        # TOWER B FORWARD PASS — Encode the Task features
        # ================================================================

        # Mirrors Tower A exactly; independent weights allow specialised representations.
        b_x1 = self.b_layer1(tower_b_x)   # shape: (B, 256)
        b_x2 = self.b_layer2(b_x1)        # shape: (B, 256)
        b_x_res = b_x1 + b_x2             # shape: (B, 256)  — residual skip connection
        emb_b = self.b_layer3(b_x_res)    # shape: (B, embedding_dim)

        # ================================================================
        # AFFINITY SCORE COMPUTATION
        # ================================================================

        # Step 1: Cosine Similarity
        # Measures the angle between emb_a and emb_b in embedding space.
        # Range: [-1, 1] — direction-only, magnitude-invariant.
        # dim=1 computes element-wise along the embedding dimension for each
        # sample in the batch, yielding one scalar per pair.
        raw_similarity = F.cosine_similarity(emb_a, emb_b, dim=1)  # shape: (B,)

        # Step 2: Temperature Scaling
        # Multiplying by a learnable temperature τ allows the model to sharpen
        # (high τ → more confident predictions) or soften (low τ → uncertain)
        # the distribution without changing the relative ordering of scores.
        scaled_similarity = raw_similarity * self.temperature  # shape: (B,)

        # Step 3: Sigmoid Squashing → Affinity Score in [0.0, 1.0]
        # Sigmoid maps ℝ → (0, 1), giving us a probability-like affinity score.
        # unsqueeze(-1) reshapes (B,) → (B, 1) to be consistent with loss functions
        # that expect a 2D target tensor (e.g., BCELoss, MSELoss).
        affinity_score = torch.sigmoid(scaled_similarity).unsqueeze(-1)  # shape: (B, 1)

        # ================================================================
        # HOOK LOGIC — Interpretation / Analysis Mode
        # ================================================================
        # This branching logic is the "interpretation hook" used by downstream
        # roles (Role 3: Analysis, Role 4: Explainability) to extract raw vectors
        # without modifying the main training pipeline.
        #
        #   Role 3 usage:
        #       out = model(emps, tasks, return_embeddings=True)
        #       umap_input = torch.cat([out['emb_a'], out['emb_b']], dim=0).detach().numpy()
        #
        #   Role 4 usage:
        #       out = model(emps, tasks, return_embeddings=True)
        #       captum.attr.IntegratedGradients(model).attribute(emps, target=out['affinity_score'])

        if return_embeddings:
            # Return enriched dict for UMAP / TensorBoard / Captum analysis
            return {
                "affinity_score": affinity_score,  # (B, 1) — bounded score for ranking
                "emb_a": emb_a,                    # (B, embedding_dim) — employee embedding
                "emb_b": emb_b,                    # (B, embedding_dim) — task embedding
            }

        # Default: return only the score — clean interface for training loops
        return affinity_score


# ===========================================================================
# TESTING & VISUALIZATION BLOCK
# ===========================================================================
# Run this file directly to verify model shapes, min/max score range,
# print a full layer summary, and generate a computational graph image.
#
# Required packages:
#   pip install torchinfo torchviz
#   # torchviz requires system-level Graphviz: https://graphviz.org/download/
# ===========================================================================

if __name__ == "__main__":

    from torchinfo import summary          # Rich model summary table (shape-aware)
    from torchviz import make_dot          # Computational graph visualizer

    print("=" * 65)
    print("  IDISCDualTower — Smoke Test & Visualization")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    BATCH_SIZE         = 64   # Number of Employee–Task pairs per batch
    TOWER_A_INPUT_DIM  = 25   # Employee feature vector size
    TOWER_B_INPUT_DIM  = 18   # Task feature vector size
    EMBEDDING_DIM      = 64   # Shared embedding projection size

    # ------------------------------------------------------------------
    # Instantiate Model
    # ------------------------------------------------------------------
    model = IDISCDualTower(
        tower_a_input_dim=TOWER_A_INPUT_DIM,
        tower_b_input_dim=TOWER_B_INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
    )
    # Set to eval mode for inference (disables Dropout, fixes BatchNorm stats)
    model.eval()

    # ------------------------------------------------------------------
    # Generate Dummy Input Tensors
    # ------------------------------------------------------------------
    # torch.randn simulates normalized feature vectors; real inputs should be
    # preprocessed (StandardScaler / MinMaxScaler) before reaching the model.
    dummy_employees = torch.randn(BATCH_SIZE, TOWER_A_INPUT_DIM)  # (64, 25)
    dummy_tasks     = torch.randn(BATCH_SIZE, TOWER_B_INPUT_DIM)  # (64, 18)

    # ------------------------------------------------------------------
    # Test 1: return_embeddings=False — affinity score only
    # ------------------------------------------------------------------
    print("\n[Test 1] Forward pass — affinity scores only (return_embeddings=False)")
    with torch.no_grad():  # Disable gradient tracking for inference efficiency
        affinity_out = model(dummy_employees, dummy_tasks, return_embeddings=False)

    # Assert the output is a plain Tensor (not a dict)
    assert isinstance(affinity_out, torch.Tensor), \
        f"Expected torch.Tensor, got {type(affinity_out)}"
    assert affinity_out.shape == (BATCH_SIZE, 1), \
        f"Expected shape ({BATCH_SIZE}, 1), got {affinity_out.shape}"

    print(f"  Output type  : {type(affinity_out).__name__}")
    print(f"  Output shape : {tuple(affinity_out.shape)}")
    print(f"  Score min    : {affinity_out.min().item():.6f}")
    print(f"  Score max    : {affinity_out.max().item():.6f}")
    print(f"  Score mean   : {affinity_out.mean().item():.6f}")

    # ------------------------------------------------------------------
    # Test 2: return_embeddings=True — embeddings + score dict
    # ------------------------------------------------------------------
    print("\n[Test 2] Forward pass — with embeddings (return_embeddings=True)")
    with torch.no_grad():
        embed_out = model(dummy_employees, dummy_tasks, return_embeddings=True)

    # Assert the output is a dictionary with the expected keys
    assert isinstance(embed_out, dict), \
        f"Expected dict, got {type(embed_out)}"
    assert set(embed_out.keys()) == {"affinity_score", "emb_a", "emb_b"}, \
        f"Unexpected keys: {embed_out.keys()}"
    assert embed_out["affinity_score"].shape == (BATCH_SIZE, 1)
    assert embed_out["emb_a"].shape == (BATCH_SIZE, EMBEDDING_DIM)
    assert embed_out["emb_b"].shape == (BATCH_SIZE, EMBEDDING_DIM)

    print(f"  Output type       : {type(embed_out).__name__}")
    print(f"  Keys              : {list(embed_out.keys())}")
    print(f"  affinity_score    : {tuple(embed_out['affinity_score'].shape)}")
    print(f"  emb_a             : {tuple(embed_out['emb_a'].shape)}")
    print(f"  emb_b             : {tuple(embed_out['emb_b'].shape)}")

    print("\n[OK] All assertions passed.")

    # ------------------------------------------------------------------
    # Terminal Summary — torchinfo
    # ------------------------------------------------------------------
    # torchinfo traces a dummy forward pass and prints a rich table showing
    # layer names, output shapes, parameter counts, and memory estimates.
    print("\n" + "=" * 65)
    print("  Model Summary (via torchinfo)")
    print("=" * 65)
    summary(
        model,
        input_data=(dummy_employees, dummy_tasks),  # Pass both towers simultaneously
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,  # Expand nested Sequential blocks up to 4 levels deep
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Computational Graph — torchviz
    # ------------------------------------------------------------------
    # make_dot traces the autograd computation graph from the output tensor
    # back through all operations. This is invaluable for debugging the
    # residual connections and verifying the model's data flow visually.
    print("\n" + "=" * 65)
    print("  Generating Computational Graph (via torchviz)")
    print("=" * 65)

    # Re-run WITHOUT no_grad() so the graph is populated with gradient nodes
    graph_score = model(dummy_employees, dummy_tasks, return_embeddings=False)

    # Build the graph, labelling each parameter tensor by its attribute name
    dot_graph = make_dot(
        graph_score,
        params=dict(model.named_parameters()),
        show_attrs=True,    # Show tensor attributes (shape, dtype) on nodes
        show_saved=True,    # Show saved tensors used for backward pass
    )

    # Render and save as PNG (requires Graphviz system binaries)
    # Anchor the path to this script's directory so the PNG always lands
    # next to IDISC_DualTower_Architecture.py, regardless of the CWD.
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    output_path = _os.path.join(_script_dir, "idisc_dual_tower_architecture")
    dot_graph.render(output_path, format="png", cleanup=True)
    print(f"\n[OK] Graph saved -> '{output_path}.png'")
    print("    (Requires Graphviz system binaries: https://graphviz.org/download/)\n")

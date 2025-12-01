# Fusion and Classification Documentation - Cross-Modal Integration and ADHD Prediction

## Table of Contents
1. [Overview](#overview)
2. [Cross-Modal Fusion Architecture](#cross-modal-fusion-architecture)
3. [Projection Layers](#projection-layers)
4. [Bidirectional Cross-Attention](#bidirectional-cross-attention)
5. [Fusion Network](#fusion-network)
6. [Classifier Architecture](#classifier-architecture)
7. [Loss Functions](#loss-functions)
8. [Training Strategy](#training-strategy)
9. [Mathematical Formulations](#mathematical-formulations)
10. [Implementation Details](#implementation-details)

---

## Overview

The fusion and classification components integrate information from the GNN and STAN branches to produce final ADHD vs Control predictions.

**Input:**
- GNN embedding: `(batch, 32)` - Graph structure from FC matrices
- STAN embedding: `(batch, 128)` - Temporal dynamics from ROI timeseries

**Output:**
- Logits: `(batch, 2)` - Raw prediction scores for [Control, ADHD]
- Probabilities: `(batch, 2)` - Softmax normalized probabilities
- Prediction: `(batch,)` - Binary classification (0=Control, 1=ADHD)

**Architecture Flow:**
```
GNN Embedding (32) ──┐
                      ├──> Projection (128 each) ──> Cross-Attention ──>
STAN Embedding (128) ─┘         ↓                          ↓
                           Concatenate ──────────> Fusion Network (512→256→128)
                                                           ↓
                                                    Classifier (128→64→2)
                                                           ↓
                                                      Logits/Predictions
```

**Key Innovations:**
1. **Cross-modal attention:** GNN and STAN attend to each other bidirectionally
2. **Multi-scale fusion:** Combines both projected and cross-attended features
3. **Focal loss:** Addresses class imbalance between ADHD and Control subjects
4. **Residual connections:** Preserves information flow through deep networks

---

## Cross-Modal Fusion Architecture

### Class Definition

```python
class CrossModalFusion(nn.Module):
    def __init__(
        self,
        gnn_dim: int = 32,        # GNN embedding dimension
        stan_dim: int = 128,      # STAN embedding dimension
        fusion_dim: int = 128,    # Common projection dimension
        num_heads: int = 4,       # Attention heads
        dropout: float = 0.3      # Dropout rate
    ):
        super().__init__()
```

**Design Philosophy:**
- Project both modalities to common dimension (128)
- Use cross-attention to exchange information
- Concatenate all representations for rich fusion
- Reduce to final embedding through fusion network

### Layer Components

**Projection Layers:**
```python
self.gnn_proj = nn.Linear(32, 128)      # GNN: 32 → 128
self.stan_proj = nn.Linear(128, 128)    # STAN: 128 → 128 (identity projection)
```

**Cross-Attention Modules:**
```python
self.gnn_cross_attn = nn.MultiheadAttention(
    embed_dim=128,
    num_heads=4,
    dropout=0.1,
    batch_first=True
)

self.stan_cross_attn = nn.MultiheadAttention(
    embed_dim=128,
    num_heads=4,
    dropout=0.1,
    batch_first=True
)
```

**Fusion Network:**
```python
self.fusion_layers = nn.Sequential(
    nn.Linear(512, 256),              # 4×128 = 512 input
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.BatchNorm1d(256),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3)
)
```

---

## Projection Layers

### Purpose
Map GNN and STAN embeddings to a common dimensionality for attention and fusion.

### GNN Projection

**Input:** GNN embedding `(batch, 32)`

```python
gnn_proj = self.gnn_proj(gnn_embedding)
```

**Linear Transformation:**
```
gnn_proj = W_gnn @ gnn_embedding + b_gnn

where:
W_gnn: (128, 32)  - 4,096 parameters
b_gnn: (128,)     - 128 parameters
```

**Example:**
```
gnn_embedding = [0.5, -0.3, 0.8, ..., 0.2]  (32 values)

gnn_proj[0] = W[0,0]×0.5 + W[0,1]×(-0.3) + ... + W[0,31]×0.2 + b[0]
            = 0.42

gnn_proj = [0.42, 0.18, -0.23, ..., 0.31]  (128 values)
```

**Output:** `(batch, 128)`

### STAN Projection

**Input:** STAN embedding `(batch, 128)`

```python
stan_proj = self.stan_proj(stan_embedding)
```

**Linear Transformation:**
```
stan_proj = W_stan @ stan_embedding + b_stan

where:
W_stan: (128, 128)  - 16,384 parameters
b_stan: (128,)      - 128 parameters
```

**Note:** This is NOT an identity transformation despite same dimensions.
- Learns optimal projection into fusion space
- May emphasize certain temporal features
- Aligns with GNN projection space

**Output:** `(batch, 128)`

### Projection Purpose

**Why project to common dimension?**

1. **Attention compatibility:** Queries, keys, values must match dimensions
2. **Feature alignment:** GNN (spatial) and STAN (temporal) in common space
3. **Information bottleneck:** Forces compression of GNN, allows STAN refinement
4. **Cross-modal communication:** Enables meaningful attention between modalities

**Dimension choices:**
- GNN: 32 → 128 (expansion)
  - 32D captures essential graph structure
  - 128D expansion for richer attention
- STAN: 128 → 128 (refinement)
  - 128D already optimal for temporal dynamics
  - Projection refines for fusion context

---

## Bidirectional Cross-Attention

### Purpose
Allow GNN and STAN branches to exchange information and highlight relevant features from each other.

### Cross-Attention Mechanism

**Two directions:**
1. **GNN attends to STAN:** Graph structure focuses on relevant temporal patterns
2. **STAN attends to GNN:** Temporal dynamics focus on important spatial structures

### GNN Cross-Attention

**Query:** GNN features "What graph patterns should I focus on?"
**Key/Value:** STAN features "Here are temporal patterns to consider"

```python
gnn_cross, _ = self.gnn_cross_attn(
    query=gnn_proj.unsqueeze(1),      # (batch, 1, 128)
    key=stan_proj.unsqueeze(1),       # (batch, 1, 128)
    value=stan_proj.unsqueeze(1)      # (batch, 1, 128)
)
```

**Unsqueeze Reasoning:**
- MultiheadAttention expects: `(batch, seq_len, embed_dim)`
- Our embeddings: `(batch, embed_dim)` (single vector per subject)
- Add sequence dimension: `(batch, 1, embed_dim)`

**Attention Computation:**
```
Step 1: Linear projections
Q_gnn = W_Q @ gnn_proj         (batch, 1, 128)
K_stan = W_K @ stan_proj       (batch, 1, 128)
V_stan = W_V @ stan_proj       (batch, 1, 128)

Step 2: Scaled dot-product
scores = (Q_gnn @ K_stan^T) / sqrt(128/4)
       = (Q_gnn @ K_stan^T) / sqrt(32)   # 4 heads, 32 per head

scores: (batch, 1, 1) - similarity between GNN query and STAN key

Step 3: Softmax (trivial for single key)
attn_weights = softmax(scores) = 1.0  (only one key)

Step 4: Apply to values
gnn_cross = attn_weights @ V_stan
          = 1.0 × V_stan
          = V_stan (batch, 1, 128)

Step 5: Output projection
gnn_cross = W_O @ gnn_cross  (batch, 1, 128)
```

**Multi-Head Attention (4 heads):**
```
Head 1 (dim 0-31):
Q1 = W_Q1 @ gnn_proj[:,0:32]
K1 = W_K1 @ stan_proj[:,0:32]
V1 = W_V1 @ stan_proj[:,0:32]
head1 = softmax(Q1@K1^T / sqrt(32)) @ V1

Head 2 (dim 32-63):
... (similar)

Head 3 (dim 64-95):
... (similar)

Head 4 (dim 96-127):
... (similar)

Output: Concat([head1, head2, head3, head4]) @ W_O
      = (batch, 1, 128)
```

**Squeeze Output:**
```python
gnn_cross = gnn_cross.squeeze(1)  # (batch, 128)
```

**Interpretation:**
- GNN features modulated by temporal information
- Graph patterns weighted by their temporal relevance
- Example: FC connections active during specific timepoints emphasized

### STAN Cross-Attention

**Query:** STAN features "What temporal patterns should I focus on?"
**Key/Value:** GNN features "Here are spatial structures to consider"

```python
stan_cross, _ = self.stan_cross_attn(
    query=stan_proj.unsqueeze(1),     # (batch, 1, 128)
    key=gnn_proj.unsqueeze(1),        # (batch, 1, 128)
    value=gnn_proj.unsqueeze(1)       # (batch, 1, 128)
)

stan_cross = stan_cross.squeeze(1)    # (batch, 128)
```

**Attention Computation:**
```
Q_stan = W_Q @ stan_proj       (batch, 1, 128)
K_gnn = W_K @ gnn_proj         (batch, 1, 128)
V_gnn = W_V @ gnn_proj         (batch, 1, 128)

scores = (Q_stan @ K_gnn^T) / sqrt(32)
attn_weights = softmax(scores) = 1.0
stan_cross = W_O @ (attn_weights @ V_gnn)
```

**Interpretation:**
- Temporal features modulated by spatial information
- Time patterns weighted by their spatial context
- Example: Temporal fluctuations in highly connected regions emphasized

### Why Cross-Attention with Single Vectors?

**Question:** Why use attention when there's only one query and one key?

**Answer:** Multi-head attention with learnable projections
1. **Different perspectives:** Each head learns different GNN↔STAN relationships
2. **Learnable gating:** Attention weights are learned, not fixed
3. **Non-linear interaction:** Combines with projections for complex fusion
4. **Architectural consistency:** Same mechanism as other attention layers

**Effective computation:**
```
gnn_cross ≈ f(gnn_proj, stan_proj)  - learned combination function
stan_cross ≈ g(stan_proj, gnn_proj) - different learned function

Not simple concatenation, but learned cross-modal features
```

### Cross-Attention Outputs

**GNN cross-attended:** `(batch, 128)`
- Graph features enhanced with temporal context

**STAN cross-attended:** `(batch, 128)`
- Temporal features enhanced with spatial context

---

## Fusion Network

### Purpose
Integrate all four representations into a unified embedding for classification.

### Concatenation

```python
fused = torch.cat([
    gnn_proj,      # Projected GNN (batch, 128)
    stan_proj,     # Projected STAN (batch, 128)
    gnn_cross,     # GNN attended to STAN (batch, 128)
    stan_cross     # STAN attended to GNN (batch, 128)
], dim=1)

Shape: (batch, 512)
```

**Four Components:**
1. **gnn_proj:** Pure spatial structure
2. **stan_proj:** Pure temporal dynamics
3. **gnn_cross:** Spatially-aware temporal features
4. **stan_cross:** Temporally-aware spatial features

**Information Richness:**
- Captures both modalities independently
- Captures cross-modal interactions
- Provides redundancy for robustness
- Enables multi-scale feature learning

### Fusion Layer 1

```python
fc1 = nn.Linear(512, 256)
```

**Computation:**
```
x = fc1(fused)
x = W1 @ fused + b1

where:
W1: (256, 512)  - 131,072 parameters
b1: (256,)      - 256 parameters

Output: (batch, 256)
```

**Example:**
```
fused = [gnn_proj(128) || stan_proj(128) || gnn_cross(128) || stan_cross(128)]
      = [0.42, ..., 0.18, ..., 0.23, ..., 0.31, ...]  (512 values)

x[0] = W1[0,:] @ fused + b1[0]
     = W1[0,0]×0.42 + W1[0,1]×... + W1[0,511]×0.31 + b1[0]
     = 0.38

x = [0.38, 0.52, -0.14, ..., 0.27]  (256 values)
```

### ReLU Activation

```python
x = F.relu(x)
```

**ReLU Function:**
```
relu(x) = max(0, x)

Example:
Before: [0.38, 0.52, -0.14, ..., 0.27]
After:  [0.38, 0.52, 0.0, ..., 0.27]

Negative values zeroed, positive unchanged
```

**Purpose:** Non-linearity, sparse activations

### Dropout

```python
x = self.dropout(x)
```

**Dropout (p=0.3):**
```
Training:
- Randomly zero 30% of activations
- Scale remaining by 1/0.7

Example:
Before: [0.38, 0.52, 0.0, 0.27, ...]
Mask:   [1,    0,    1,   1,    ...]
After:  [0.54, 0.0,  0.0, 0.39, ...]  (scaled)

Inference: No dropout
```

### Batch Normalization

```python
x = self.batch_norm1(x)
```

**Normalization per feature:**
```
For each of 256 features:
μ = (1/batch) Σ x[b,f]
σ² = (1/batch) Σ (x[b,f] - μ)²

normalized[b,f] = (x[b,f] - μ) / sqrt(σ² + ε)
output[b,f] = γ[f] × normalized[b,f] + β[f]

where γ, β are learnable per-feature parameters
```

**Purpose:**
- Stabilize training
- Reduce internal covariate shift
- Allow higher learning rates

### Fusion Layer 2

```python
fc2 = nn.Linear(256, 128)
```

**Computation:**
```
x = fc2(x)
x = W2 @ x + b2

where:
W2: (128, 256)  - 32,768 parameters
b2: (128,)      - 128 parameters

Output: (batch, 128)
```

### Second ReLU and Dropout

```python
x = F.relu(x)
x = self.dropout(x)
```

**Same operations as Layer 1:**
- ReLU: Sparsify activations
- Dropout: Regularization (p=0.3)

### Output

**Fused Embedding:**
- **Shape:** `(batch, 128)`
- **Content:** Integrated multimodal representation
  - Spatial structure from GNN
  - Temporal dynamics from STAN
  - Cross-modal interactions from attention
  - Hierarchically refined features

**Typical Value Range:** -2 to +3 (after batch norm, dropout)

---

## Classifier Architecture

### Purpose
Transform fused embedding into binary ADHD vs Control prediction.

### Layer Structure

```python
self.classifier = nn.Sequential(
    nn.Linear(128, 64),           # First layer
    nn.ReLU(),
    nn.Dropout(0.5),              # High dropout for regularization
    nn.BatchNorm1d(64),
    nn.Linear(64, 2)              # Output layer (2 classes)
)
```

### Classifier Layer 1

```python
fc1 = nn.Linear(128, 64)
```

**Computation:**
```
h = W_cls1 @ fused_embedding + b_cls1

where:
W_cls1: (64, 128)  - 8,192 parameters
b_cls1: (64,)      - 64 parameters

Output: (batch, 64)
```

**Example:**
```
fused_embedding = [0.42, 0.18, -0.23, ..., 0.31]  (128 values)

h[0] = W[0,:] @ fused_embedding + b[0]
     = 0.58

h = [0.58, -0.22, 0.41, ..., 0.19]  (64 values)
```

### ReLU

```python
h = F.relu(h)
```

```
After: [0.58, 0.0, 0.41, ..., 0.19]  (negatives zeroed)
```

### High Dropout (0.5)

```python
h = self.dropout(h)
```

**Dropout (p=0.5):**
```
Training: Zero 50% of activations, scale by 1/0.5 = 2

Example:
Before: [0.58, 0.0, 0.41, 0.19, ...]
Mask:   [1,    0,   1,    0,    ...]
After:  [1.16, 0.0, 0.82, 0.0,  ...]  (scaled by 2)

Inference: No dropout
```

**Rationale for high dropout:**
- Classifier most prone to overfitting
- Forces redundant representations
- Improves generalization to new sites

### Batch Normalization

```python
h = self.batch_norm(h)
```

**Per-feature normalization (64 features)**

### Output Layer

```python
logits = nn.Linear(64, 2)(h)
```

**Computation:**
```
logits = W_out @ h + b_out

where:
W_out: (2, 64)  - 128 parameters
b_out: (2,)     - 2 parameters

Output: (batch, 2)
```

**Logit Interpretation:**
```
logits[0]: Score for class 0 (Control)
logits[1]: Score for class 1 (ADHD)

Example:
logits = [2.3, -1.5]
  → Model strongly predicts Control (class 0)

logits = [-0.8, 1.9]
  → Model strongly predicts ADHD (class 1)

logits = [0.1, 0.2]
  → Model slightly favors ADHD but uncertain
```

### Softmax Conversion to Probabilities

```python
probs = F.softmax(logits, dim=1)
```

**Softmax Formula:**
```
probs[i] = exp(logits[i]) / Σ exp(logits[j])

Example:
logits = [2.3, -1.5]

exp(logits) = [exp(2.3), exp(-1.5)] = [9.97, 0.22]
sum = 9.97 + 0.22 = 10.19

probs = [9.97/10.19, 0.22/10.19] = [0.978, 0.022]
  → 97.8% Control, 2.2% ADHD

Σ probs = 0.978 + 0.022 = 1.0 ✓
```

### Prediction

```python
predictions = torch.argmax(logits, dim=1)
```

**Argmax:**
```
predictions[i] = argmax(logits[i])
               = class with highest logit

Example:
logits = [2.3, -1.5]
argmax = 0  → Control

logits = [-0.8, 1.9]
argmax = 1  → ADHD
```

### Classifier Output Summary

**Three output formats:**

1. **Logits:** `(batch, 2)` - Raw scores
   - Used for loss computation
   - Unbounded values

2. **Probabilities:** `(batch, 2)` - Softmax probabilities
   - Used for confidence estimates
   - Sum to 1.0, range [0, 1]

3. **Predictions:** `(batch,)` - Binary labels
   - Used for accuracy
   - Values: 0 (Control) or 1 (ADHD)

---

## Loss Functions

### Cross-Entropy Loss

**Standard loss for classification:**

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
```

**Formula:**
```
L = -(1/N) Σ log(softmax(logits[i])[targets[i]])

For binary classification:
L = -(1/N) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]

where:
y_i: true label (0 or 1)
p_i: predicted probability for class 1
```

**Example:**
```
Subject 0:
True label: 1 (ADHD)
Logits: [-0.8, 1.9]
Probs: [0.215, 0.785]

Loss = -log(0.785) = 0.242

Subject 1:
True label: 0 (Control)
Logits: [2.3, -1.5]
Probs: [0.978, 0.022]

Loss = -log(0.978) = 0.022

Batch loss = (0.242 + 0.022) / 2 = 0.132
```

### Focal Loss

**Addresses class imbalance:**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        self.alpha = alpha    # Class weight
        self.gamma = gamma    # Focusing parameter
```

**Formula:**
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

where:
p_t = {  p      if y = 1 (positive class)
      {  1-p    if y = 0 (negative class)

α_t = {  α      if y = 1
      {  1-α    if y = 0
```

**Components:**

**1. Modulating Factor: `(1 - p_t)^γ`**
```
High confidence (p_t ≈ 1):
(1 - 0.95)^2 = 0.05^2 = 0.0025  → Very small weight (easy example)

Low confidence (p_t ≈ 0.5):
(1 - 0.5)^2 = 0.5^2 = 0.25      → Moderate weight

Very low confidence (p_t ≈ 0.1):
(1 - 0.1)^2 = 0.9^2 = 0.81      → High weight (hard example)

Effect: Focus training on hard examples, down-weight easy examples
```

**2. Alpha Balancing: `α_t`**
```
α = 0.8 (typical for ADHD dataset)

Class 1 (ADHD): α = 0.8    (more weight, rarer class)
Class 0 (Control): α = 0.2 (less weight, more common)

Purpose: Balance gradient contributions from imbalanced classes
```

**Complete Example:**
```
Subject: ADHD (y=1)
Predicted prob: p=0.85 (fairly confident)

Standard CE Loss:
L_CE = -log(0.85) = 0.163

Focal Loss (α=0.8, γ=2):
p_t = p = 0.85
α_t = α = 0.8
FL = -0.8 × (1-0.85)^2 × log(0.85)
   = -0.8 × 0.15^2 × (-0.163)
   = -0.8 × 0.0225 × (-0.163)
   = 0.00293

Reduction: 0.00293 / 0.163 = 1.8% of CE loss
(Easy example down-weighted)

---

Subject: ADHD (y=1)
Predicted prob: p=0.50 (uncertain)

Standard CE Loss:
L_CE = -log(0.50) = 0.693

Focal Loss:
FL = -0.8 × (1-0.50)^2 × log(0.50)
   = -0.8 × 0.50^2 × (-0.693)
   = -0.8 × 0.25 × (-0.693)
   = 0.139

Reduction: 0.139 / 0.693 = 20% of CE loss
(Hard example still heavily weighted)
```

**Focal Loss Implementation:**
```python
def forward(self, inputs, targets):
    # inputs: logits (batch, 2)
    # targets: labels (batch,) with values 0 or 1
    
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    # ce_loss: (batch,)
    
    pt = torch.exp(-ce_loss)  # p_t
    # pt close to 1 for confident correct predictions
    # pt close to 0 for wrong predictions
    
    alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
    # alpha_t = 0.8 for ADHD (targets=1)
    # alpha_t = 0.2 for Control (targets=0)
    
    focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
    
    return focal_loss.mean()
```

**When to use Focal Loss:**
- Class imbalance (ADHD 60%, Control 40%)
- Hard negatives (difficult Control subjects misclassified as ADHD)
- Fine-tuning stage (after initial training with CE loss)

**Hyperparameters:**
- `α = 0.8`: Favors ADHD class (typically minority)
- `γ = 2.0`: Standard focusing parameter
  - `γ = 0`: Reduces to CE loss
  - `γ = 2`: Moderate down-weighting of easy examples
  - `γ = 5`: Strong focus on hard examples

---

## Training Strategy

### Optimizer

**AdamW (Adam with Weight Decay):**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,                   # Learning rate
    weight_decay=0.01           # L2 regularization
)
```

**Adam Update Rule:**
```
m_t = β1 × m_{t-1} + (1 - β1) × g_t          # First moment (momentum)
v_t = β2 × v_{t-1} + (1 - β2) × g_t²         # Second moment (variance)

m̂_t = m_t / (1 - β1^t)                      # Bias correction
v̂_t = v_t / (1 - β2^t)

θ_t = θ_{t-1} - lr × m̂_t / (sqrt(v̂_t) + ε)  # Parameter update

where:
g_t: gradient at time t
β1 = 0.9 (default)
β2 = 0.999 (default)
ε = 1e-8
```

**Weight Decay (L2 Regularization):**
```
θ_t = θ_{t-1} × (1 - lr × weight_decay) - lr × gradient

Effect: Penalizes large weights, encourages simpler models
```

### Learning Rate Scheduler

**ReduceLROnPlateau:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # Monitor validation loss
    factor=0.5,           # Reduce by half
    patience=5,           # Wait 5 epochs
    verbose=True,
    min_lr=1e-6          # Minimum learning rate
)
```

**Schedule:**
```
Initial LR: 0.001

After 5 epochs without improvement:
LR → 0.0005

After 5 more epochs without improvement:
LR → 0.00025

...continues until min_lr = 0.000001
```

**Purpose:**
- Adaptive learning: Fast initial progress, fine-tuning later
- Escape plateaus: Smaller steps when stuck
- Improved convergence: Precise final adjustments

### Early Stopping

**Implementation:**
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False
```

**Example:**
```
Epoch 10: val_loss = 0.650 → best_loss = 0.650, counter = 0
Epoch 11: val_loss = 0.645 → best_loss = 0.645, counter = 0 (improved)
Epoch 12: val_loss = 0.648 → counter = 1 (no improvement)
Epoch 13: val_loss = 0.646 → counter = 2
...
Epoch 21: val_loss = 0.651 → counter = 10 → STOP (patience exhausted)
```

**Purpose:**
- Prevent overfitting
- Save computation time
- Automatic hyperparameter (when to stop)

### Gradient Clipping

**Prevent exploding gradients:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**Formula:**
```
total_norm = sqrt(Σ ||g_p||²)  for all parameters p

if total_norm > max_norm:
    scale = max_norm / total_norm
    g_p = g_p × scale  for all parameters p
```

**Example:**
```
Gradients: [1.2, 3.5, -2.1, 8.3, ...]
Total norm = sqrt(1.2² + 3.5² + 2.1² + 8.3² + ...) = 12.7

max_norm = 5.0
scale = 5.0 / 12.7 = 0.394

Clipped gradients: [0.47, 1.38, -0.83, 3.27, ...]
New norm = 5.0 ✓
```

**Purpose:**
- Stabilize LSTM training (prone to exploding gradients)
- Prevent NaN losses
- Enable higher learning rates

### Training Loop

**One epoch:**
```python
for batch in train_loader:
    # Forward pass
    fc_matrices = batch['fc_matrix']
    roi_timeseries = batch['roi_timeseries']
    labels = batch['label']
    
    logits = model(fc_matrices, roi_timeseries)
    loss = loss_fn(logits, labels)
    
    # Backward pass
    optimizer.zero_grad()           # Clear gradients
    loss.backward()                 # Compute gradients
    clip_grad_norm_(...)            # Clip gradients
    optimizer.step()                # Update parameters
    
    # Metrics
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean()
```

**Validation:**
```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        logits = model(...)
        loss = loss_fn(logits, labels)
        
        # Accumulate metrics
        ...

# Update learning rate
scheduler.step(val_loss)

# Check early stopping
if early_stopping(val_loss):
    break
```

### Cross-Validation

**K-Fold Cross-Validation (K=5):**
```
Fold 1: Train[2,3,4,5], Val[1]
Fold 2: Train[1,3,4,5], Val[2]
Fold 3: Train[1,2,4,5], Val[3]
Fold 4: Train[1,2,3,5], Val[4]
Fold 5: Train[1,2,3,4], Val[5]

Final performance: Average of 5 folds
```

**Leave-One-Site-Out (LOSO):**
```
Fold 1: Train[NYU, OHSU, Peking, ...], Val[KKI]
Fold 2: Train[KKI, OHSU, Peking, ...], Val[NYU]
...

Test generalization to new sites
```

### Batch Balancing

**Class balancing within batches:**
```python
adhd_subjects = data[data['label'] == 1]
control_subjects = data[data['label'] == 0]

batch_size = 32
n_adhd = 16  # Half batch
n_control = 16

batch = pd.concat([
    adhd_subjects.sample(n_adhd),
    control_subjects.sample(n_control)
])
```

**Purpose:**
- Equal class representation per batch
- Stable gradient estimates
- Prevents batch-specific biases

---

## Mathematical Formulations

### Complete Forward Pass

**GNN Branch:**
```
FC Matrix (200×200) → Node Features (4) → Edge Index →
GATConv × 3 + TopK Pooling × 2 → Global Pool →
GNN Embedding (32)
```

**STAN Branch:**
```
ROI Timeseries (200×time) → BiLSTM → Multi-Head Attention →
Temporal Conv → Global Pool → STAN Embedding (128)
```

**Fusion:**
```
Z_gnn ∈ ℝ^32
Z_stan ∈ ℝ^128

Projections:
Z_gnn_proj = W_gnn Z_gnn + b_gnn      (32 → 128)
Z_stan_proj = W_stan Z_stan + b_stan  (128 → 128)

Cross-Attention:
Z_gnn_cross = CrossAttn(Q=Z_gnn_proj, K,V=Z_stan_proj)
Z_stan_cross = CrossAttn(Q=Z_stan_proj, K,V=Z_gnn_proj)

Concatenation:
Z_concat = [Z_gnn_proj || Z_stan_proj || Z_gnn_cross || Z_stan_cross]
         ∈ ℝ^512

Fusion Network:
Z_fused = Dropout(ReLU(BN(Dropout(ReLU(W2 @ (W1 Z_concat + b1) + b2)))))
        ∈ ℝ^128
```

**Classifier:**
```
H = Dropout(ReLU(W_cls1 Z_fused + b_cls1))  ∈ ℝ^64
Logits = W_out H + b_out                     ∈ ℝ^2

Probabilities = Softmax(Logits)
Prediction = argmax(Logits)
```

### Loss Function

**Cross-Entropy:**
```
L_CE = -(1/N) Σ [y_i log(p_i,1) + (1-y_i) log(p_i,0)]

where:
y_i ∈ {0, 1}: true label
p_i,0, p_i,1: predicted probabilities for classes 0, 1
```

**Focal Loss:**
```
L_FL = -(1/N) Σ α_i (1 - p_i,y_i)^γ log(p_i,y_i)

where:
α_i = α if y_i=1, else (1-α)
p_i,y_i: predicted probability for true class
γ = 2.0 (focusing parameter)
```

### Gradient Descent

**Parameter Update:**
```
θ^(t+1) = θ^(t) - η ∇L(θ^(t))

where:
θ: model parameters
η: learning rate
∇L: gradient of loss
```

**Adam Optimizer:**
```
m_t = β1 m_{t-1} + (1-β1) g_t
v_t = β2 v_{t-1} + (1-β2) g_t²

θ_t = θ_{t-1} - η × m̂_t / (sqrt(v̂_t) + ε)
```

---

## Implementation Details

### Model Parameters

**Total Parameters:**
```
GNN Branch:        ~150,000
STAN Branch:       ~350,000
Fusion Layer:      ~165,000
Classifier:        ~8,500
─────────────────────────────
Total:             ~673,500 parameters
```

**Parameter Breakdown:**

**Fusion Layer:**
- GNN projection: 32×128 + 128 = 4,224
- STAN projection: 128×128 + 128 = 16,512
- GNN cross-attn: 128×128×4 (Q,K,V,O) = 65,536
- STAN cross-attn: 128×128×4 = 65,536
- Fusion FC1: 512×256 + 256 = 131,328
- Fusion FC2: 256×128 + 128 = 32,896
- **Total:** ~316,032 parameters

**Classifier:**
- FC1: 128×64 + 64 = 8,256
- FC2: 64×2 + 2 = 130
- **Total:** 8,386 parameters

### Memory Usage

**Per subject (single forward pass):**
```
GNN Input: 200×200×4 bytes = 160 KB
STAN Input: 200×176×4 bytes = 140 KB
Intermediate: ~500 KB
Gradients: ~2.7 MB
─────────────────────────────
Total: ~3.5 MB per subject
```

**Batch of 32:**
```
Forward: 32 × 3.5 MB = 112 MB
Backward: 32 × 2.7 MB = 86 MB
Optimizer states: ~11 MB
─────────────────────────────
Total: ~210 MB per batch
```

**GPU Memory (RTX 3090 24GB):**
- Model parameters: ~3 GB
- Batch processing: ~0.2 GB
- Available: ~20 GB for caching/multi-batches

### Training Time

**Per epoch (800 subjects, batch=32):**
```
Forward pass: 25 batches × 30ms = 0.75s
Backward pass: 25 batches × 50ms = 1.25s
Validation: ~0.5s
─────────────────────────────
Total: ~2.5s per epoch

100 epochs: ~4 minutes
5-fold CV: ~20 minutes total
```

### Hyperparameter Summary

```
Training:
- Batch size: 32
- Epochs: 100 (with early stopping)
- Learning rate: 0.001 (initial)
- Weight decay: 0.01
- Gradient clip: 5.0

Architecture:
- GNN output: 32
- STAN output: 128
- Fusion dim: 128
- Classifier hidden: 64

Regularization:
- Dropout (fusion): 0.3
- Dropout (classifier): 0.5
- Batch normalization: Yes

Attention:
- Cross-attention heads: 4
- Attention dropout: 0.1

Loss:
- Focal loss: α=0.8, γ=2.0
- Or CrossEntropyLoss

Scheduler:
- Type: ReduceLROnPlateau
- Factor: 0.5
- Patience: 5 epochs

Early stopping:
- Patience: 10 epochs
- Min delta: 0.001
```

### Validation Metrics

**Computed per epoch:**
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'auc': roc_auc_score(y_true, y_probs[:,1])
}
```

**Confusion Matrix:**
```
                Predicted
              Control  ADHD
Actual Control   TN     FP
       ADHD      FN     TP

Metrics:
Precision = TP / (TP + FP)  - Of predicted ADHD, how many correct?
Recall = TP / (TP + FN)     - Of actual ADHD, how many found?
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Output Interpretation

**For a single subject:**
```python
logits = model(fc_matrix, roi_timeseries)
# logits: [2.3, -1.5]

probs = F.softmax(logits, dim=0)
# probs: [0.978, 0.022]

prediction = torch.argmax(logits)
# prediction: 0 (Control)

confidence = probs[prediction]
# confidence: 0.978 (97.8% confident)
```

**Clinical Interpretation:**
```
High confidence Control (>90%): Likely neurotypical
Low confidence Control (50-70%): Borderline, subclinical traits
Low confidence ADHD (50-70%): Borderline, mild symptoms
High confidence ADHD (>90%): Strong ADHD indicators

Note: Not diagnostic tool, research purposes only
```

---

This documentation covers every computation, variable, architectural component, and training detail from fused embeddings to final ADHD predictions.

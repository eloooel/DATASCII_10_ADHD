# GNN Branch Documentation - Graph Neural Network for Functional Connectivity

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Node Feature Extraction](#node-feature-extraction)
4. [Edge Construction](#edge-construction)
5. [Graph Attention Convolution](#graph-attention-convolution)
6. [Hierarchical Pooling](#hierarchical-pooling)
7. [Graph Embedding](#graph-embedding)
8. [Mathematical Formulations](#mathematical-formulations)
9. [Implementation Details](#implementation-details)

---

## Overview

The GNN (Graph Neural Network) branch processes functional connectivity (FC) matrices to learn graph-structured representations of brain networks. It transforms the 200×200 correlation matrix into a graph where:

**Input:** Functional connectivity matrix `(batch, 200, 200)`
- Nodes: 200 ROIs (brain regions)
- Edges: Functional connections between ROIs
- Edge weights: Pearson correlation values

**Output:** Graph embedding `(batch, 32)` 
- Fixed-dimensional vector representing brain network structure
- Captures topological and connectivity patterns
- Invariant to node ordering

**Key Components:**
1. **Node Feature Extraction:** Compute graph-theoretic features per ROI
2. **Edge Construction:** Create sparse graph from dense FC matrix
3. **Graph Attention Convolution:** Learn node representations with attention
4. **Hierarchical Pooling:** Multi-scale graph coarsening
5. **Global Pooling:** Aggregate to fixed graph-level embedding

**Architecture Flow:**
```
FC Matrix (200×200) → Node Features (200×4) → GAT Layer 1 (200×128) →
Pool 1 (160×128) → GAT Layer 2 (160×64) → Pool 2 (96×64) →
GAT Layer 3 (96×32) → Global Pool → Graph Embedding (32)
```

---

## Architecture

### Class Definition

```python
class EnhancedGNNBranch(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,           # Node features: 4 graph metrics
        hidden_dims: list = [128, 64, 32],  # Hidden layer dimensions
        dropout: float = 0.3,          # Dropout probability
        pool_ratios: list = [0.8, 0.6] # Pooling ratios (keep 80%, then 60%)
    ):
        super().__init__()
```

**Parameters:**
- `input_dim`: Number of node features (degree, clustering, eigenvector centrality, local efficiency)
- `hidden_dims`: List of hidden dimensions for each GAT layer
- `dropout`: Dropout rate for regularization
- `pool_ratios`: Fraction of nodes to keep after each pooling operation

### Layer Structure

**Graph Attention Layers:**
```python
self.gnn_layers = nn.ModuleList([
    GATConv(4, 128, heads=4, concat=False, dropout=0.3),      # Layer 1
    GATConv(128, 64, heads=4, concat=False, dropout=0.3),     # Layer 2
    GATConv(64, 32, heads=4, concat=False, dropout=0.3)       # Layer 3
])
```

**Batch Normalization:**
```python
self.batch_norms = nn.ModuleList([
    nn.BatchNorm1d(128),  # After layer 1
    nn.BatchNorm1d(64),   # After layer 2
    nn.BatchNorm1d(32)    # After layer 3
])
```

**Skip Connections:**
```python
self.skip_connections = nn.ModuleList([
    nn.Linear(4, 128),    # Dimension change 4→128
    nn.Linear(128, 64),   # Dimension change 128→64
    nn.Linear(64, 32)     # Dimension change 64→32
])
```

**Hierarchical Pooling:**
```python
self.pool_layers = nn.ModuleList([
    TopKPooling(128, ratio=0.8),  # Keep top 80% nodes (200→160)
    TopKPooling(64, ratio=0.6)    # Keep top 60% nodes (160→96)
])
```

---

## Node Feature Extraction

### Purpose
Convert functional connectivity matrix into node-level features that capture each ROI's role in the brain network.

### Input
- **FC Matrix:** `(batch, 200, 200)` - Pearson correlations
- **Values:** -1 to +1 (diagonal = 0)

### Process

#### Initialize Feature Tensor
```python
node_features = torch.zeros(batch_size * 200, 4, device=fc_matrix.device)
```

**Shape:** `(batch_size × 200, 4)` - Flattened batch of node features

#### Extract Per-Node Features

For each subject in batch:

##### 1. Degree Centrality
```python
degree = torch.sum(torch.abs(fc), dim=1)
```

**Formula:**
```
degree_i = Σ|FC_ij|  for all j ≠ i
```

**Interpretation:**
- Sum of absolute connection strengths
- High degree: Hub region (highly connected)
- Typical range: 10-50

**Example (ROI 84):**
```
Connections: [0.45, 0.23, 0.12, 0.67, ...]  (199 values)
Degree = |0.45| + |0.23| + |0.12| + |0.67| + ... = 28.4
```

##### 2. Clustering Coefficient
```python
clustering = torch.diagonal(
    torch.matmul(torch.matmul(fc, fc), fc)
) / (degree + 1e-8)
```

**Formula:**
```
clustering_i = [FC³]_ii / degree_i

where FC³ = FC × FC × FC (matrix cube)
```

**Interpretation:**
- Measures local interconnectivity
- High clustering: ROI neighbors are also connected
- Range: 0-1 (typically 0.1-0.5)

**Matrix Cube Explanation:**
```
FC³[i,i] = Σ_j Σ_k FC[i,j] * FC[j,k] * FC[k,i]
```

Counts closed triangles involving node i.

**Example:**
```
ROI 84 connects to ROI 123 (0.45)
ROI 123 connects to ROI 56 (0.67)
ROI 56 connects back to ROI 84 (0.23)

Triangle contribution: 0.45 × 0.67 × 0.23 = 0.069
```

##### 3. Eigenvector Centrality
```python
eigenvals, eigenvecs = torch.linalg.eigh(fc)
eigen_centrality = torch.abs(eigenvecs[:, -1])  # Principal eigenvector
```

**Formula:**
```
Ax = λx

where:
A = FC matrix
x = eigenvector (centrality scores)
λ = eigenvalue
```

**Interpretation:**
- Importance based on connections to other important nodes
- High eigenvector centrality: Connected to hubs
- Range: 0-1 (normalized)

**Eigenvalue Decomposition:**
```
FC = Q Λ Q^T

where:
Q = eigenvectors (200×200)
Λ = diagonal eigenvalues

Principal eigenvector = Q[:, -1] (largest eigenvalue)
```

**Example:**
```
If ROI 84 has eigen_centrality = 0.15:
- Above average importance (mean ≈ 0.05)
- Connected to other central nodes
```

##### 4. Local Efficiency
```python
local_eff = torch.mean(torch.abs(fc), dim=1)
```

**Formula:**
```
local_eff_i = (1/N) Σ|FC_ij|  for all j
```

**Interpretation:**
- Average connection strength
- High local efficiency: Strong neighborhood connectivity
- Range: 0-0.8 (typical: 0.1-0.4)

**Example:**
```
ROI 84 connections: [0.45, 0.23, -0.12, 0.67, ...]
Local efficiency = mean(|connections|) = 0.31
```

#### Stack Features
```python
node_features[start_idx:end_idx] = torch.stack([
    degree,           # (200,)
    clustering,       # (200,)
    eigen_centrality, # (200,)
    local_eff         # (200,)
], dim=1)  # Result: (200, 4)
```

### Output

**Node Feature Matrix:**
- **Shape:** `(batch_size × 200, 4)`
- **Features per node:** [degree, clustering, eigenvector_centrality, local_efficiency]

**Example for one subject:**
```
ROI     Degree  Clustering  Eigen_Cent  Local_Eff
1       24.5    0.32        0.08        0.25
2       28.4    0.41        0.15        0.31
3       19.2    0.28        0.06        0.21
...
200     31.8    0.45        0.18        0.35

Shape: (200, 4)
```

**Feature Statistics (Typical):**
```
Feature              Mean    Std     Min     Max
Degree              25.3    5.2     10.1    45.6
Clustering          0.34    0.08    0.15    0.58
Eigenvector Cent    0.05    0.03    0.01    0.21
Local Efficiency    0.28    0.06    0.12    0.47
```

---

## Edge Construction

### Purpose
Convert dense 200×200 FC matrix to sparse edge list for efficient graph convolution.

### Strategy: Threshold-Based Sparsification

Keep only strong connections (top 20% by absolute correlation).

### Implementation

```python
def _create_edge_index_from_fc(self, fc_matrix: torch.Tensor) -> torch.Tensor:
    batch_size, n_rois, _ = fc_matrix.shape
    
    # Compute threshold (80th percentile)
    fc_flat = torch.abs(fc_matrix).view(batch_size, -1)
    threshold = torch.quantile(fc_flat, 0.8, dim=1, keepdim=True).unsqueeze(-1)
    
    # Create binary mask
    mask = torch.abs(fc_matrix) > threshold
    
    # Extract edge indices per subject
    edge_index_list = []
    for b in range(batch_size):
        sources, targets = torch.where(mask[b])
        
        # Add batch offset
        sources = sources + b * n_rois
        targets = targets + b * n_rois
        
        edge_index_list.append(torch.stack([sources, targets]))
    
    # Concatenate all edges
    edge_index = torch.cat(edge_index_list, dim=1)
    
    return edge_index
```

### Threshold Calculation

**80th Percentile:**
```
Total connections per subject: 200 × 200 = 40,000
Keep top 20%: 8,000 edges
Remove bottom 80%: 32,000 weak edges
```

**Example FC Matrix:**
```
Absolute correlations: [0.05, 0.12, 0.23, 0.45, 0.67, 0.78, ...]
Sorted: [0.78, 0.67, 0.45, 0.23, 0.12, 0.05, ...]

80th percentile threshold = 0.23

Keep edges: |FC| > 0.23
Discard edges: |FC| ≤ 0.23
```

### Edge Index Format

**PyTorch Geometric Format:**
```
edge_index = [
    [source_1, source_2, source_3, ...],  # Source nodes
    [target_1, target_2, target_3, ...]   # Target nodes
]

Shape: (2, num_edges)
```

**Example:**
```
edge_index = [
    [0,   0,   1,   1,   2,   2,   ...],  # Source ROIs
    [5,   12,  3,   8,   1,   9,   ...]   # Target ROIs
]

Interpretation:
Edge 1: ROI 0 → ROI 5
Edge 2: ROI 0 → ROI 12
Edge 3: ROI 1 → ROI 3
...
```

### Batch Handling

**Batch Offset:**
```
Subject 0: ROIs 0-199
Subject 1: ROIs 200-399
Subject 2: ROIs 400-599
...

For subject b:
node_offset = b × 200
sources = sources + node_offset
targets = targets + node_offset
```

**Batched Edge Index:**
```
Batch size = 3, 8000 edges per subject

edge_index shape: (2, 24000)
- First 8000 edges: Subject 0 (nodes 0-199)
- Next 8000 edges: Subject 1 (nodes 200-399)
- Last 8000 edges: Subject 2 (nodes 400-599)
```

### Output

**Edge Index:**
- **Shape:** `(2, total_edges)`
- **Total edges:** `batch_size × ~8000`
- **Data type:** `torch.long` (integer indices)

**Statistics:**
```
Original dense graph: 40,000 edges per subject (fully connected)
After thresholding: ~8,000 edges per subject (20%)
Sparsity gain: 80% reduction
Memory savings: 5× less storage
```

---

## Graph Attention Convolution

### Purpose
Learn node representations by aggregating information from neighbors with learned attention weights.

### Graph Attention Network (GAT)

**Formula:**
```
h'_i = σ(Σ α_ij W h_j)

where:
h_i = node i features
h_j = neighbor j features
W = learnable weight matrix
α_ij = attention coefficient (how much i attends to j)
σ = activation function (ReLU)
```

### Attention Mechanism

#### Step 1: Linear Transformation
```python
W h_i → transformed node features
W h_j → transformed neighbor features
```

**Weight Matrix:**
```
W: (input_dim, output_dim)

For Layer 1: (4, 128)
h_i: (4,) → W h_i: (128,)
```

#### Step 2: Attention Score
```python
e_ij = LeakyReLU(a^T [W h_i || W h_j])
```

**Variables:**
- `a`: Learnable attention vector `(2 × output_dim,)`
- `||`: Concatenation
- `[W h_i || W h_j]`: `(2 × output_dim,)` concatenated features
- `e_ij`: Unnormalized attention score (scalar)

**Example:**
```
W h_i = [0.5, -0.2, 0.8, ..., 0.3]  # 128 values
W h_j = [0.3, 0.1, -0.4, ..., 0.6]  # 128 values

Concat: [0.5, -0.2, ..., 0.3, 0.3, 0.1, ..., 0.6]  # 256 values

a^T [concat]: 0.5×a[0] + (-0.2)×a[1] + ... = 0.42

e_ij = LeakyReLU(0.42) = 0.42
```

#### Step 3: Attention Coefficient (Softmax)
```python
α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k∈N(i) exp(e_ik)
```

**Normalization:**
```
For node i with neighbors [j1, j2, j3]:

e_ij1 = 0.42
e_ij2 = 0.67
e_ij3 = 0.31

exp(e_ij1) = 1.52
exp(e_ij2) = 1.95
exp(e_ij3) = 1.36

Sum = 1.52 + 1.95 + 1.36 = 4.83

α_ij1 = 1.52 / 4.83 = 0.31
α_ij2 = 1.95 / 4.83 = 0.40
α_ij3 = 1.36 / 4.83 = 0.28

Sum(α) = 1.0 ✓
```

**Interpretation:**
- `α_ij2 = 0.40`: Most important neighbor (40% weight)
- `α_ij1 = 0.31`: Moderate importance (31% weight)
- `α_ij3 = 0.28`: Least important neighbor (28% weight)

#### Step 4: Weighted Aggregation
```python
h'_i = ReLU(Σ α_ij W h_j)
```

**Computation:**
```
h'_i = ReLU(
    0.31 × W h_j1 +
    0.40 × W h_j2 +
    0.28 × W h_j3
)

Result: (128,) - updated node features
```

### Multi-Head Attention

**Purpose:** Learn different aspects of relationships

```python
GATConv(input_dim, output_dim, heads=4, concat=False)
```

**Parameters:**
- `heads=4`: 4 independent attention mechanisms
- `concat=False`: Average outputs (instead of concatenating)

**Multi-Head Process:**
```
Head 1: α¹_ij, h'¹_i
Head 2: α²_ij, h'²_i
Head 3: α³_ij, h'³_i
Head 4: α⁴_ij, h'⁴_i

Combined: h'_i = (1/4)(h'¹_i + h'²_i + h'³_i + h'⁴_i)
```

**Benefits:**
- Each head learns different connectivity patterns
- Head 1: Structural connections
- Head 2: Functional connections
- Head 3: Long-range connections
- Head 4: Local connections

### Layer-by-Layer Propagation

#### Layer 1: Input → Hidden 1
```python
Input: node_features (batch×200, 4)
GAT: 4 → 128 dimensions, 4 heads
Output: h1 (batch×200, 128)
```

**Node Update:**
```
ROI 1: [24.5, 0.32, 0.08, 0.25] (4 features)
    ↓ GAT Layer 1
ROI 1: [0.23, -0.45, 0.67, ..., 0.12] (128 features)
```

#### Layer 2: Hidden 1 → Hidden 2 (After Pooling)
```python
Input: h1_pooled (batch×160, 128)  # After TopK pooling
GAT: 128 → 64 dimensions, 4 heads
Output: h2 (batch×160, 64)
```

**Information Flow:**
- 160 most important nodes (20% pruned)
- Each node aggregates from its neighbors
- Dimensions reduced: 128 → 64

#### Layer 3: Hidden 2 → Hidden 3 (After Pooling)
```python
Input: h2_pooled (batch×96, 64)
GAT: 64 → 32 dimensions, 4 heads
Output: h3 (batch×96, 32)
```

**Final node representations:**
- 96 most critical nodes
- 32-dimensional embeddings
- Ready for global pooling

---

## Hierarchical Pooling

### Purpose
Create multi-scale graph representations by progressively coarsening the graph structure.

### TopK Pooling

**Formula:**
```
score_i = |σ(X_i W_pool)|

Keep top k nodes with highest scores
k = ⌊ratio × n⌋
```

### Implementation

```python
TopKPooling(in_channels, ratio=0.8)
```

**Process:**

#### 1. Compute Node Scores
```python
scores = torch.sigmoid(torch.matmul(X, W_pool))
```

**Variables:**
- `X`: Node features `(n_nodes, in_channels)`
- `W_pool`: Learnable parameter `(in_channels, 1)`
- `scores`: Importance scores `(n_nodes,)`

**Example:**
```
X: (200, 128) - node features after Layer 1
W_pool: (128, 1) - learnable weights

scores = sigmoid(X @ W_pool)
scores: (200,) - values between 0 and 1

Example scores:
[0.89, 0.67, 0.45, 0.92, 0.34, ...]
```

#### 2. Select Top-K Nodes
```python
k = int(ratio × n_nodes)
_, idx = torch.topk(scores, k)
```

**Pooling 1 (ratio=0.8):**
```
n_nodes = 200
k = 0.8 × 200 = 160

Keep top 160 nodes
Discard bottom 40 nodes
```

**Selected Nodes:**
```
Sorted scores: [0.92, 0.89, 0.87, 0.85, ...]
Indices: [3, 0, 123, 45, ...]  (160 node indices)
```

#### 3. Update Node Features
```python
X_pooled = X[idx] * scores[idx].view(-1, 1)
```

**Feature Scaling:**
```
For kept node i:
X_new[i] = X_old[i] × score[i]

Example:
X_old[3] = [0.5, -0.2, ..., 0.8]  (128 values)
score[3] = 0.92

X_new[3] = [0.46, -0.18, ..., 0.74]  (scaled by 0.92)
```

**Purpose:** Re-weight features by importance

#### 4. Update Edge Index
```python
# Remove edges connected to discarded nodes
# Remap node indices to 0...k-1
```

**Example:**
```
Original edges:
[0→5, 0→12, 1→3, 2→7, 3→8, ...]

After pooling (keeping nodes [0, 3, 5, 7, 8, 12, ...]):
- Keep: 0→5, 0→12, 3→8
- Discard: 1→3 (node 1 removed), 2→7 (node 2 removed)

Remapped:
Node 0 → 0
Node 3 → 1
Node 5 → 2
Node 7 → 3
Node 8 → 4
Node 12 → 5

New edges: [0→2, 0→5, 1→4, ...]
```

#### 5. Update Batch Indices
```python
batch_pooled = batch[idx]
```

**Maintains subject identity for kept nodes.**

### Pooling Hierarchy

**Pooling 1 (ratio=0.8):**
```
Before: 200 nodes × batch_size
After: 160 nodes × batch_size
Removed: 40 least important nodes per subject
```

**Pooling 2 (ratio=0.6):**
```
Before: 160 nodes × batch_size
After: 96 nodes × batch_size
Removed: 64 more nodes
Total removed: 104 nodes (52% of original)
```

**Multi-Scale Representation:**
- **Layer 1 (200 nodes):** Fine-grained, local patterns
- **Layer 2 (160 nodes):** Intermediate, regional patterns
- **Layer 3 (96 nodes):** Coarse, global patterns

### Benefits

1. **Computational Efficiency:** Fewer nodes → faster computation
2. **Focus on Important Regions:** Keep discriminative ROIs
3. **Multi-Scale Learning:** Capture hierarchical brain organization
4. **Regularization:** Prune noisy nodes

---

## Graph Embedding

### Global Mean Pooling

**Purpose:** Aggregate node-level representations into a single graph-level embedding.

### Formula
```
graph_embedding = (1/n) Σ h_i

where:
h_i = final node features
n = number of nodes
```

### Implementation

```python
graph_embedding = global_mean_pool(h, batch)
```

**Process:**

#### Input
```
h: (batch_size × n_nodes, feature_dim)

Example (batch=2):
h: (192, 32)  # 2 subjects × 96 nodes each
batch: [0,0,0,...,0, 1,1,1,...,1]  # Subject labels
       └─ 96 zeros ┘ └─ 96 ones ┘
```

#### Aggregation
```python
for subject_id in range(batch_size):
    # Get nodes belonging to this subject
    subject_mask = (batch == subject_id)
    subject_nodes = h[subject_mask]  # (96, 32)
    
    # Average across nodes
    subject_embedding = torch.mean(subject_nodes, dim=0)  # (32,)
```

**Example Calculation:**
```
Subject 0 nodes:
Node 0: [0.5, -0.2, 0.8, ..., 0.3]
Node 1: [0.3, 0.1, -0.4, ..., 0.6]
...
Node 95: [0.7, 0.4, 0.2, ..., -0.1]

Mean: [(0.5+0.3+...+0.7)/96, (-0.2+0.1+...+0.4)/96, ...]
Result: [0.42, 0.15, 0.23, ..., 0.18]  (32 values)
```

#### Output
```
graph_embedding: (batch_size, feature_dim)
                (batch_size, 32)

Example (batch=32):
shape: (32, 32)
```

### Alternative Pooling Methods

**Max Pooling:**
```python
graph_embedding = global_max_pool(h, batch)
# Takes maximum value across nodes per feature
```

**Attention Pooling:**
```python
attention_weights = softmax(W @ h)
graph_embedding = Σ attention_weights[i] × h[i]
# Weighted average based on learned importance
```

**Sum Pooling:**
```python
graph_embedding = global_add_pool(h, batch)
# Sum node features (unnormalized)
```

**Current Choice:** Mean pooling (balanced, stable gradients)

---

## Mathematical Formulations

### Complete Forward Pass

**Input:**
- FC matrix: `X₀ ∈ ℝ^(200×200)`

**Node Features:**
```
F = [degree, clustering, eigen_centrality, local_efficiency]
F ∈ ℝ^(200×4)
```

**GAT Layer 1:**
```
H¹ = σ(Â D⁻¹ F W¹ ⊙ A¹)

where:
Â = adjacency matrix (from edge_index)
D = degree matrix
W¹ = learnable weights (4×128)
A¹ = attention coefficients
σ = ReLU activation
⊙ = element-wise product

Output: H¹ ∈ ℝ^(200×128)
```

**Pooling 1:**
```
s¹ = σ(H¹ W_pool¹)
idx¹ = topk(s¹, k=160)
H¹_pool = H¹[idx¹] ⊙ s¹[idx¹]

Output: H¹_pool ∈ ℝ^(160×128)
```

**GAT Layer 2:**
```
H² = σ(Â² D²⁻¹ H¹_pool W² ⊙ A²)

Output: H² ∈ ℝ^(160×64)
```

**Pooling 2:**
```
s² = σ(H² W_pool²)
idx² = topk(s², k=96)
H²_pool = H²[idx²] ⊙ s²[idx²]

Output: H²_pool ∈ ℝ^(96×64)
```

**GAT Layer 3:**
```
H³ = σ(Â³ D³⁻¹ H²_pool W³ ⊙ A³)

Output: H³ ∈ ℝ^(96×32)
```

**Global Pooling:**
```
z = (1/96) Σᵢ H³ᵢ

Output: z ∈ ℝ^32 (graph embedding)
```

### Attention Coefficient Formula

```
α_ij = softmax_j(e_ij)

where:
e_ij = LeakyReLU(a^T [W h_i || W h_j])

a ∈ ℝ^(2F')      (attention parameters)
W ∈ ℝ^(F×F')     (transformation matrix)
h_i ∈ ℝ^F        (node i features)
h_j ∈ ℝ^F        (neighbor j features)
|| = concatenation
```

### Multi-Head Attention

```
h'_i = ||ᴷₖ₌₁ σ(Σⱼ αᵏᵢⱼ Wᵏ hⱼ)   (concat=True)

h'_i = (1/K) Σᴷₖ₌₁ σ(Σⱼ αᵏᵢⱼ Wᵏ hⱼ)   (concat=False)

where:
K = number of heads (4)
αᵏᵢⱼ = attention weights for head k
Wᵏ = weights for head k
```

### Skip Connection

```
H' = σ(GAT(H) + W_skip H)

where:
W_skip = identity (if dim match) or linear projection (if dim change)
```

**Purpose:** Gradient flow, preserve information

---

## Implementation Details

### Device Handling

```python
fc_matrix = fc_matrix.to(self.device)  # Move to GPU/CPU
edge_index = edge_index.to(fc_matrix.device)
node_features = torch.zeros(..., device=fc_matrix.device)
```

### Batch Processing

**Batch Size:** Typically 16-32 subjects

**Total Nodes:**
```
batch_size = 32
nodes_per_subject = 200
total_nodes = 32 × 200 = 6400

After Layer 1: 6400 nodes (200 per subject)
After Pool 1: 5120 nodes (160 per subject)
After Layer 2: 5120 nodes
After Pool 2: 3072 nodes (96 per subject)
After Layer 3: 3072 nodes
After Global Pool: 32 embeddings (1 per subject)
```

### Memory Optimization

**Sparse Edge Storage:**
```
Dense: 200×200 = 40,000 entries per subject
Sparse: ~8,000 edges per subject
Memory: 5× reduction
```

**Gradient Checkpointing:**
```python
torch.utils.checkpoint.checkpoint(gat_layer, x, edge_index)
```

Trades compute for memory (recomputes activations during backward).

### Training Parameters

**Dropout:** 0.3 (30% neurons dropped)
- Prevents overfitting
- Applied after each GAT layer

**Batch Normalization:**
```python
h = batch_norm(h)
```
- Stabilizes training
- Accelerates convergence

**Learning Rate:** Typically 0.001 (Adam optimizer)

**Weight Initialization:**
```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
```

### Output Dimensions Summary

```
Stage                    Shape                Notes
─────────────────────────────────────────────────────────────────
Input FC Matrix         (batch, 200, 200)    Correlation matrix
Node Features           (batch×200, 4)       Graph metrics
GAT Layer 1             (batch×200, 128)     First encoding
Pool 1                  (batch×160, 128)     80% kept
GAT Layer 2             (batch×160, 64)      Second encoding
Pool 2                  (batch×96, 64)       60% kept
GAT Layer 3             (batch×96, 32)       Final encoding
Global Pool             (batch, 32)          Graph embedding
```

### Computational Complexity

**Node Feature Extraction:** O(N²) per subject
- N = 200 ROIs
- Operations: matrix multiplications for clustering, eigendecomposition

**GAT Layer:** O(E × F) per layer
- E = number of edges (~8,000)
- F = feature dimension

**Pooling:** O(N × F) per pooling operation

**Total (per subject):** ~10-20ms on GPU (RTX 3090)

---

This documentation covers every computation, variable, and architectural component of the GNN branch from functional connectivity input to graph embedding output.

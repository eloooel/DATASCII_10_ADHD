# STAN Branch Documentation - Spatio-Temporal Attention Network for ROI Timeseries

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Bidirectional LSTM Encoding](#bidirectional-lstm-encoding)
4. [Multi-Head Spatio-Temporal Attention](#multi-head-spatio-temporal-attention)
5. [Temporal Convolution](#temporal-convolution)
6. [Global Temporal Pooling](#global-temporal-pooling)
7. [Mathematical Formulations](#mathematical-formulations)
8. [Implementation Details](#implementation-details)

---

## Overview

The STAN (Spatio-Temporal Attention Network) branch processes ROI timeseries data to learn temporal dynamics and spatial relationships across brain regions over time.

**Input:** ROI timeseries `(batch, 200, n_timepoints)`
- 200 ROIs (brain regions)
- Variable timepoints per subject (140-220 TRs)
- Each value: Average BOLD signal in that ROI at that timepoint

**Output:** Temporal embedding `(batch, 128)`
- Fixed-dimensional vector representing temporal brain dynamics
- Captures both spatial (across ROIs) and temporal (across time) patterns
- Invariant to sequence length

**Key Components:**
1. **Bidirectional LSTM:** Encode temporal sequences in both directions
2. **Multi-Head Attention:** Learn which timepoints/patterns are important
3. **Temporal Convolution:** Extract local temporal patterns
4. **Global Pooling:** Aggregate to fixed temporal embedding

**Architecture Flow:**
```
ROI Timeseries (batch, 200, time) → Transpose (batch, time, 200) →
BiLSTM (batch, time, 256) → Multi-Head Attention (batch, time, 256) →
Temporal Conv (batch, 128, time) → Global Avg Pool → Embedding (batch, 128)
```

---

## Architecture

### Class Definition

```python
class EnhancedSTANBranch(nn.Module):
    def __init__(
        self,
        input_dim: int = 200,         # Number of ROIs
        hidden_dim: int = 128,         # LSTM hidden dimension
        num_layers: int = 2,           # Number of LSTM layers
        dropout: float = 0.3           # Dropout probability
    ):
        super().__init__()
```

**Parameters:**
- `input_dim`: Number of ROIs (200 for Schaefer-200)
- `hidden_dim`: LSTM hidden state dimension (128)
- `num_layers`: Number of stacked LSTM layers (2)
- `dropout`: Dropout rate for regularization (0.3)

### Layer Structure

**Bidirectional LSTM:**
```python
self.temporal_encoder = nn.LSTM(
    input_size=200,           # 200 ROIs
    hidden_size=128,          # 128 hidden units
    num_layers=2,             # 2 stacked layers
    batch_first=True,         # Input: (batch, time, features)
    bidirectional=True,       # Forward + backward
    dropout=0.3               # Between LSTM layers
)
```

**Output dimension:** 128 × 2 = 256 (bidirectional)

**Multi-Head Attention:**
```python
self.attention = MultiHeadSpatioTemporalAttention(
    d_model=256,              # Input dimension (BiLSTM output)
    n_heads=8,                # 8 attention heads
    dropout=0.3
)
```

**Temporal Convolution:**
```python
self.temp_conv = nn.Conv1d(
    in_channels=256,          # BiLSTM + attention output
    out_channels=128,         # Reduced dimension
    kernel_size=3,            # 3-timepoint window
    padding=1                 # Preserve length
)
```

**Batch Normalization:**
```python
self.temp_norm = nn.BatchNorm1d(128)
```

---

## Bidirectional LSTM Encoding

### Purpose
Encode temporal sequences in both forward and backward directions to capture long-range temporal dependencies.

### Input Preparation

**Original Shape:** `(batch, n_rois, n_timepoints)`
```python
# Input: (32, 200, 176)
x = x.transpose(1, 2)  # (32, 176, 200)
# Now: (batch, time, features)
```

**Transpose Reasoning:**
- LSTM expects: `(batch, sequence_length, input_size)`
- Sequence: Timepoints (temporal ordering)
- Features: ROI values at each timepoint

### LSTM Architecture

**Stacked Bidirectional LSTM:**
```
Layer 1 (Forward):  →→→→→→→
Input (200) → Hidden (128)

Layer 1 (Backward): ←←←←←←←
Input (200) → Hidden (128)

Concatenate: (256,)

Layer 2 (Forward):  →→→→→→→
Input (256) → Hidden (128)

Layer 2 (Backward): ←←←←←←←
Input (256) → Hidden (128)

Final Output: (256,) per timepoint
```

### LSTM Cell Operations

**Forward Pass Equations:**
```
Input Gate:     i_t = σ(W_i x_t + U_i h_{t-1} + b_i)
Forget Gate:    f_t = σ(W_f x_t + U_f h_{t-1} + b_f)
Cell Update:    g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)
Output Gate:    o_t = σ(W_o x_t + U_o h_{t-1} + b_o)

Cell State:     c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden State:   h_t = o_t ⊙ tanh(c_t)
```

**Variables:**
- `x_t`: Input at time t `(200,)` - ROI values
- `h_{t-1}`: Previous hidden state `(128,)`
- `c_{t-1}`: Previous cell state `(128,)`
- `W, U`: Weight matrices
- `b`: Bias vectors
- `σ`: Sigmoid activation (0 to 1)
- `tanh`: Hyperbolic tangent (-1 to 1)
- `⊙`: Element-wise multiplication

**Gate Interpretations:**

**Input Gate (`i_t`):**
- Controls how much new information to add
- High value: Accept new input
- Low value: Ignore new input

**Forget Gate (`f_t`):**
- Controls what to forget from cell state
- High value: Keep old memory
- Low value: Discard old memory

**Output Gate (`o_t`):**
- Controls how much cell state to expose
- High value: Output current state
- Low value: Suppress output

### Bidirectional Processing

**Forward LSTM:**
```
Time:      t=0   t=1   t=2   t=3   ...  t=175
Forward:   →→→→→→→→→→→→→→→→→→→→→→→→→→

h_fwd[0] = LSTM_fwd(x[0])
h_fwd[1] = LSTM_fwd(x[1], h_fwd[0])
h_fwd[2] = LSTM_fwd(x[2], h_fwd[1])
...
```

**Backward LSTM:**
```
Time:      t=175 t=174 t=173 ...  t=1   t=0
Backward:  ←←←←←←←←←←←←←←←←←←←←←←←←←←

h_bwd[175] = LSTM_bwd(x[175])
h_bwd[174] = LSTM_bwd(x[174], h_bwd[175])
h_bwd[173] = LSTM_bwd(x[173], h_bwd[174])
...
```

**Concatenation:**
```
For each timepoint t:
h[t] = [h_fwd[t] || h_bwd[t]]

Example at t=50:
h_fwd[50]: (128,) - forward context (t=0 to t=50)
h_bwd[50]: (128,) - backward context (t=175 to t=50)
h[50]: (256,) - full bidirectional context
```

### Example Computation

**Input at t=50:**
```
x[50] = [0.23, -0.45, 0.67, ..., 0.12]  (200 ROI values)
```

**Forward LSTM Layer 1:**
```
Previous: h_fwd[49] = [0.5, -0.2, 0.8, ...]  (128 values)

Gates:
i[50] = σ(W_i @ x[50] + U_i @ h[49] + b_i) = [0.7, 0.3, ...]
f[50] = σ(W_f @ x[50] + U_f @ h[49] + b_f) = [0.9, 0.8, ...]
g[50] = tanh(W_g @ x[50] + U_g @ h[49] + b_g) = [0.4, -0.2, ...]
o[50] = σ(W_o @ x[50] + U_o @ h[49] + b_o) = [0.8, 0.6, ...]

Cell state:
c[50] = f[50] ⊙ c[49] + i[50] ⊙ g[50]
      = [0.9×0.6 + 0.7×0.4, ...] = [0.82, ...]

Hidden state:
h_fwd[50] = o[50] ⊙ tanh(c[50])
          = [0.8×tanh(0.82), ...] = [0.53, ...]  (128 values)
```

**Backward LSTM Layer 1:**
```
(Similar computation from t=175 to t=50)
h_bwd[50] = [0.31, -0.18, ...]  (128 values)
```

**Concatenated:**
```
h_layer1[50] = [h_fwd[50] || h_bwd[50]]
             = [0.53, ..., 0.31, -0.18, ...]  (256 values)
```

**Layer 2 (Forward & Backward):**
```
Input: h_layer1[50] (256,)
Output: h_layer2[50] (256,) - final encoding
```

### Output

**LSTM Output:**
- **Shape:** `(batch, n_timepoints, 256)`
- **Example:** `(32, 176, 256)` for batch of 32 subjects
- **Each timepoint:** 256-dimensional encoding capturing:
  - Past context (forward direction)
  - Future context (backward direction)
  - Temporal dependencies
  - Spatial patterns across ROIs

**Temporal Coverage:**
```
t=0:   Early activation patterns
t=1:   Transition dynamics
...
t=88:  Mid-scan patterns (resting state baseline)
...
t=175: Late activation patterns

Each encoded with full temporal context (past + future)
```

---

## Multi-Head Spatio-Temporal Attention

### Purpose
Learn to focus on the most relevant timepoints and spatial patterns for classification.

### Attention Mechanism

**Core Idea:** Not all timepoints are equally important for ADHD classification. Attention learns which moments contain discriminative information.

### Multi-Head Attention Architecture

```python
class MultiHeadSpatioTemporalAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        self.d_k = d_model // n_heads  # 256 / 8 = 32
        
        self.w_q = nn.Linear(256, 256)  # Query projection
        self.w_k = nn.Linear(256, 256)  # Key projection
        self.w_v = nn.Linear(256, 256)  # Value projection
        self.w_o = nn.Linear(256, 256)  # Output projection
```

**Head Dimension:** `d_k = 256 / 8 = 32` per head

### Attention Computation

#### Step 1: Linear Projections

```python
Q = self.w_q(x)  # Query: What am I looking for?
K = self.w_k(x)  # Key: What do I contain?
V = self.w_v(x)  # Value: What information do I have?
```

**Shapes:**
```
x: (batch, time, 256)
Q: (batch, time, 256)
K: (batch, time, 256)
V: (batch, time, 256)
```

**Reshape for Multi-Head:**
```python
Q = Q.view(batch, time, n_heads, d_k).transpose(1, 2)
K = K.view(batch, time, n_heads, d_k).transpose(1, 2)
V = V.view(batch, time, n_heads, d_k).transpose(1, 2)

Result shape: (batch, n_heads, time, d_k)
Example: (32, 8, 176, 32)
```

#### Step 2: Scaled Dot-Product Attention

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Score Computation:**
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

**Variables:**
- `Q`: `(batch, 8, 176, 32)` - Queries
- `K^T`: `(batch, 8, 32, 176)` - Keys transposed
- `scores`: `(batch, 8, 176, 176)` - Attention scores

**Score Matrix:**
```
scores[i,j] = similarity between timepoint i and timepoint j

Example for Head 0, Subject 0:
           t=0   t=1   t=2   ...  t=175
t=0     [  0.8   0.3   0.1   ...   0.2  ]
t=1     [  0.3   0.9   0.4   ...   0.1  ]
t=2     [  0.1   0.4   0.8   ...   0.3  ]
...
t=175   [  0.2   0.1   0.3   ...   0.9  ]

Shape: (176, 176)
```

**Scaling Factor `√d_k`:**
```
d_k = 32
√d_k = 5.66

Purpose: Prevent large dot products that cause saturated softmax
```

#### Step 3: Softmax Normalization

```python
attn_weights = F.softmax(scores, dim=-1)
```

**Per-Row Softmax:**
```
For timepoint t=50:
scores[50] = [0.8, 0.3, 0.1, ..., 0.2]

exp(scores[50]) = [2.23, 1.35, 1.11, ..., 1.22]
sum = 2.23 + 1.35 + ... + 1.22 = 176.5

attn_weights[50] = [2.23/176.5, 1.35/176.5, ...]
                 = [0.013, 0.008, ...]

Σ attn_weights[50] = 1.0 ✓
```

**Attention Weights Matrix:**
```
attn_weights: (batch, 8, 176, 176)

attn_weights[0, 0, 50, :] represents:
"When processing timepoint 50 in head 0,
 how much attention to pay to each other timepoint"

High values: Important related timepoints
Low values: Less relevant timepoints
```

#### Step 4: Apply Attention to Values

```python
context = torch.matmul(attn_weights, V)
```

**Computation:**
```
context[t] = Σ attn_weights[t, τ] × V[τ]

For t=50:
context[50] = 0.013 × V[0] + 0.008 × V[1] + ... + 0.012 × V[175]
```

**Shapes:**
```
attn_weights: (batch, 8, 176, 176)
V: (batch, 8, 176, 32)
context: (batch, 8, 176, 32)
```

**Interpretation:**
- Each output timepoint is a weighted combination of all input timepoints
- Weights learned to focus on relevant temporal patterns

#### Step 5: Concatenate Heads

```python
context = context.transpose(1, 2).contiguous()
context = context.view(batch, time, n_heads × d_k)

Shape: (batch, 176, 256)
```

**Multi-Head Combination:**
```
Head 0: Attends to early activation patterns
Head 1: Attends to sustained activity
Head 2: Attends to transient peaks
Head 3: Attends to baseline fluctuations
Head 4: Attends to network transitions
Head 5: Attends to late-stage patterns
Head 6: Attends to periodic oscillations
Head 7: Attends to inter-ROI synchrony

Combined output: Rich representation capturing diverse temporal patterns
```

#### Step 6: Output Projection

```python
output = self.w_o(context)
```

**Linear Transformation:**
```
W_o: (256, 256)
output: (batch, 176, 256)
```

#### Step 7: Residual Connection + Layer Norm

```python
output = self.layer_norm(output + residual)
```

**Residual Connection:**
```
output = LayerNorm(attention_output + original_input)

Purpose:
- Gradient flow (addresses vanishing gradients)
- Information preservation
- Training stability
```

**Layer Normalization:**
```
For each timepoint:
mean = (1/256) Σ output[i]
std = sqrt((1/256) Σ (output[i] - mean)²)

normalized = (output - mean) / (std + ε)
output_final = γ × normalized + β

where γ, β are learnable parameters
```

### Attention Patterns Example

**Subject with ADHD:**
```
High attention weights at:
- t=10-30: Initial task disengagement
- t=80-90: Mid-scan attention lapses
- t=150-170: Late-stage cognitive fatigue

Pattern: Fluctuating attention, frequent transitions
```

**Control Subject:**
```
High attention weights at:
- t=0-20: Stable baseline
- t=40-160: Sustained engagement
- t=165-175: Consistent performance

Pattern: Stable attention, fewer transitions
```

### Output

**Attended Features:**
- **Shape:** `(batch, n_timepoints, 256)`
- **Content:** Time-aware representations
  - Enhanced discriminative patterns
  - Suppressed irrelevant fluctuations
  - Context-dependent encoding

**Attention Weights (for interpretability):**
- **Shape:** `(batch, n_heads, n_timepoints, n_timepoints)`
- **Usage:** Visualize which timepoints are important
- **Example:** `(32, 8, 176, 176)`

---

## Temporal Convolution

### Purpose
Extract local temporal patterns using convolutional filters sliding over time.

### 1D Convolution Architecture

```python
self.temp_conv = nn.Conv1d(
    in_channels=256,      # Input: attended LSTM features
    out_channels=128,     # Output: reduced dimension
    kernel_size=3,        # 3-timepoint window
    padding=1             # Preserve temporal length
)
```

### Input Preparation

**Transpose for Conv1D:**
```python
# Before: (batch, time, features) = (32, 176, 256)
conv_input = attended_out.transpose(1, 2)
# After: (batch, features, time) = (32, 256, 176)
```

**Conv1D expects:** `(batch, channels, length)`
- Channels: Feature dimensions (256)
- Length: Temporal sequence (176)

### Convolution Operation

**Kernel Size 3:**
```
Window:  [t-1, t, t+1]

At t=50:
Window: [t=49, t=50, t=51]
```

**Convolution Formula:**
```
output[t] = Σ W[i] × input[t-1+i]  for i in [0, 1, 2]
          + bias

= W[0] × input[t-1] + W[1] × input[t] + W[2] × input[t+1] + b
```

**Weight Tensor:**
```
W: (out_channels, in_channels, kernel_size)
   (128, 256, 3)

Each output channel has:
- 256 × 3 = 768 weights
- 1 bias
Total: 128 × (768 + 1) = 98,432 parameters
```

### Detailed Example

**Input at t=50:**
```
input[49]: [0.5, -0.2, 0.8, ..., 0.3]  (256 values)
input[50]: [0.3, 0.1, -0.4, ..., 0.6]  (256 values)
input[51]: [0.7, 0.4, 0.2, ..., -0.1]  (256 values)
```

**For output channel 0:**
```
W[0]: (256, 3) weights

output[50, 0] = W[0,0,:] @ [input[49], input[50], input[51]] + b[0]

Expanded:
= W[0,0,0] × input[49,0] + W[0,0,1] × input[50,0] + W[0,0,2] × input[51,0]
+ W[0,1,0] × input[49,1] + W[0,1,1] × input[50,1] + W[0,1,2] × input[51,1]
+ ...
+ W[0,255,0] × input[49,255] + W[0,255,1] × input[50,255] + W[0,255,2] × input[51,255]
+ b[0]

Result: Single scalar value
```

**All 128 output channels:**
```
output[50] = [output[50,0], output[50,1], ..., output[50,127]]
           = [0.42, 0.15, 0.23, ..., 0.18]  (128 values)
```

### Padding

**Padding=1:** Add one zero on each side

```
Original:  [x[0], x[1], x[2], ..., x[175]]
Padded:    [0, x[0], x[1], x[2], ..., x[175], 0]

Boundary handling:
t=0:   Window [0, x[0], x[1]]      (left padded)
t=1:   Window [x[0], x[1], x[2]]   (normal)
...
t=175: Window [x[174], x[175], 0]  (right padded)

Output length = Input length = 176 (same)
```

### Learned Patterns

**Convolutional kernels learn to detect:**

**Kernel 1:** Rising activation
```
W = [-0.5, 0.0, 0.5]
Detects: signal increasing over 3 TRs
```

**Kernel 2:** Falling activation
```
W = [0.5, 0.0, -0.5]
Detects: signal decreasing
```

**Kernel 3:** Peak detection
```
W = [-0.25, 0.5, -0.25]
Detects: local maximum at center
```

**Kernel 4:** Sustained activity
```
W = [0.33, 0.33, 0.33]
Detects: stable signal (local average)
```

### Batch Normalization

```python
conv_out = self.temp_norm(conv_out)
```

**Per-Channel Normalization:**
```
For each of 128 channels:
mean = (1/(batch×time)) Σ conv_out[b,c,t]
var = (1/(batch×time)) Σ (conv_out[b,c,t] - mean)²

normalized[b,c,t] = (conv_out[b,c,t] - mean) / sqrt(var + ε)
output[b,c,t] = γ[c] × normalized[b,c,t] + β[c]

where γ, β are learnable per-channel parameters
```

### Activation

```python
conv_out = F.relu(conv_out)
```

**ReLU:**
```
relu(x) = max(0, x)

Negative values → 0
Positive values → unchanged

Purpose: Non-linearity, sparse activations
```

### Output

**Temporal Convolution Output:**
- **Shape:** `(batch, 128, n_timepoints)`
- **Example:** `(32, 128, 176)`
- **Content:** Local temporal features
  - Rising/falling trends
  - Peak/trough detection
  - Sustained activity patterns
  - Temporal transitions

---

## Global Temporal Pooling

### Purpose
Aggregate variable-length temporal sequences into fixed-dimensional embeddings.

### Average Pooling

```python
temporal_embedding = torch.mean(conv_out, dim=2)
```

**Computation:**
```
For each channel c:
embedding[c] = (1/time) Σ conv_out[:,c,t]  for all t

Example:
conv_out: (32, 128, 176)

Channel 0 values across time: [0.5, 0.3, 0.8, ..., 0.4]
embedding[0] = (0.5 + 0.3 + 0.8 + ... + 0.4) / 176 = 0.42

Result: 128 averaged values (one per channel)
```

**Shape Transformation:**
```
Before: (batch, 128, 176)
After:  (batch, 128)
```

### Dropout

```python
temporal_embedding = self.dropout(temporal_embedding)
```

**Dropout (p=0.3):**
```
During training:
- Randomly set 30% of values to 0
- Scale remaining by 1/0.7 to maintain expected value

Example:
Before: [0.42, 0.15, 0.23, 0.18, 0.31, ...]
Mask:   [1,    0,    1,    1,    0,    ...]  (random)
After:  [0.60, 0.0,  0.33, 0.26, 0.0,  ...]  (scaled by 1/0.7)

During inference: No dropout, use all values
```

**Purpose:** Regularization to prevent overfitting

### Output

**Final STAN Embedding:**
- **Shape:** `(batch, 128)`
- **Example:** `(32, 128)` for batch of 32 subjects
- **Content:** Temporal signature of brain activity
  - Aggregated temporal patterns
  - Discriminative features for ADHD vs Control
  - Fixed-length representation (regardless of original timepoints)

**Typical Value Range:** -2 to +2 (after batch norm, dropout)

---

## Mathematical Formulations

### Complete Forward Pass

**Input:**
```
X₀ ∈ ℝ^(batch × 200 × time)
```

**Transpose:**
```
X₁ = X₀^T ∈ ℝ^(batch × time × 200)
```

**Bidirectional LSTM:**
```
H_fwd, H_bwd = BiLSTM(X₁)

where:
H_fwd ∈ ℝ^(batch × time × 128)  (forward)
H_bwd ∈ ℝ^(batch × time × 128)  (backward)

H₂ = [H_fwd || H_bwd] ∈ ℝ^(batch × time × 256)
```

**Multi-Head Attention:**
```
Q = W_Q H₂
K = W_K H₂
V = W_V H₂

scores = softmax(QK^T / √d_k)
context = scores × V

H₃ = LayerNorm(W_O context + H₂)

Output: H₃ ∈ ℝ^(batch × time × 256)
```

**Temporal Convolution:**
```
H₃_T = H₃^T ∈ ℝ^(batch × 256 × time)

H₄ = ReLU(BatchNorm(Conv1D(H₃_T)))

where Conv1D uses kernel_size=3

Output: H₄ ∈ ℝ^(batch × 128 × time)
```

**Global Pooling:**
```
z = (1/time) Σ H₄[:, :, t]  for all t

Output: z ∈ ℝ^(batch × 128)  (final temporal embedding)
```

### LSTM Equations

**Forward LSTM:**
```
i_t = σ(W_i [x_t, h_{t-1}] + b_i)      Input gate
f_t = σ(W_f [x_t, h_{t-1}] + b_f)      Forget gate
g_t = tanh(W_g [x_t, h_{t-1}] + b_g)   Cell candidate
o_t = σ(W_o [x_t, h_{t-1}] + b_o)      Output gate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        Cell state
h_t = o_t ⊙ tanh(c_t)                   Hidden state
```

**Bidirectional Concatenation:**
```
h_t = [h⃗_t || h⃖_t]

where:
h⃗_t: forward hidden state
h⃖_t: backward hidden state
```

### Attention Equations

**Scaled Dot-Product:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
Q ∈ ℝ^(time × d_k)
K ∈ ℝ^(time × d_k)
V ∈ ℝ^(time × d_k)
d_k = 32 (per head)
```

**Multi-Head:**
```
MultiHead(Q, K, V) = Concat(head₁, ..., head₈) W_O

where:
head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

### Convolution Equation

**1D Convolution:**
```
y[t] = Σ w[k] × x[t + k] + b

where:
k ∈ [-1, 0, 1]  (kernel_size=3, centered)
w: learnable weights
b: learnable bias
```

---

## Implementation Details

### Variable Sequence Lengths

**Padding Strategy:**
```python
# Subjects may have different timepoints (140-220)
# Padding applied in data loader to match batch max length

Original lengths: [176, 145, 198, 162, ...]
Batch max: 198
Padded lengths: [198, 198, 198, 198, ...]

Padding: Zeros appended to end
```

**Attention Mask (optional):**
```python
# Mask padded positions in attention
mask = (sequence != 0)  # Binary mask
scores.masked_fill_(mask == 0, -1e9)  # Set to -infinity before softmax
```

### Memory Management

**Gradient Checkpointing:**
```python
lstm_out = torch.utils.checkpoint.checkpoint(
    self.temporal_encoder, x
)
```

Saves memory by recomputing activations during backward pass.

### Training Parameters

**LSTM Dropout:** 0.3
- Applied between LSTM layers (num_layers > 1)
- Not applied to output layer

**Attention Dropout:** 0.1
- Applied to attention weights after softmax
- Prevents overfitting to specific timepoints

**Final Dropout:** 0.3
- Applied to temporal embedding before fusion

**Learning Rate:** Typically 0.001 (Adam optimizer)

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

Prevents exploding gradients in LSTM.

### Computational Complexity

**BiLSTM:** O(time × d² × layers)
- time: sequence length (~176)
- d: hidden dimension (128)
- layers: number of LSTM layers (2)

**Multi-Head Attention:** O(time² × d)
- time²: pairwise attention scores
- d: model dimension (256)

**Temporal Convolution:** O(time × d² × k)
- k: kernel size (3)

**Total (per subject):** ~15-25ms on GPU (RTX 3090)

### Output Dimensions Summary

```
Stage                    Shape                   Notes
─────────────────────────────────────────────────────────────────
Input Timeseries        (batch, 200, time)      ROI signals
Transpose               (batch, time, 200)      Prepared for LSTM
BiLSTM                  (batch, time, 256)      Temporal encoding
Multi-Head Attention    (batch, time, 256)      Attended features
Transpose for Conv      (batch, 256, time)      Channel-first
Temporal Conv           (batch, 128, time)      Local patterns
Global Pool             (batch, 128)            Fixed embedding
```

### Typical Value Ranges

**After each stage:**
```
BiLSTM output:          -1 to +1 (tanh activation)
Attention output:       -2 to +2 (layer norm)
Conv output (pre-ReLU): -3 to +3 (batch norm)
Conv output (post-ReLU): 0 to +3 (ReLU)
Global pool:            -1 to +2 (averaged)
Final dropout:          -1.5 to +3 (scaled)
```

---

This documentation covers every computation, variable, and architectural component of the STAN branch from ROI timeseries input to temporal embedding output.

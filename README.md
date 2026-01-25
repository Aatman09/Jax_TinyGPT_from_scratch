# Phase 1: From-Scratch Transformer (Mathematical Grounding)

**Status:** Educational / Conceptual  
**Frameworks:** Pure JAX, Python (No Flax, Haiku, or NN modules)

This phase implements a complete GPT-style Transformer from first principles. The goal is **conceptual grounding**: deriving every component manually to understand tensor shapes, data flow, and mathematical operations before moving to production abstractions.

## Objectives
* **De-abstraction:** Implement every layer manually (Linear, Norm, Attention) to see "what is really happening."
* **Shape Verification:** Explicitly track $(B, T, D)$ flow through the network.
* **Math Validation:** Re-derive the mechanics of Self-Attention and SwiGLU.

---

## 1. Architectural Components

### RMSNorm
Root Mean Square Layer Normalization. Used in LLaMA/Qwen for training stability.

$$\text{RMS}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

* **Shape:** $(B, T, D) \to (B, T, D)$

### Linear Layer & SiLU
* **Linear:** Standard affine transformation $y = xW + b$. Used for projections ($Q, K, V$) and the vocabulary head.
* **SiLU (Swish):** Smooth nonlinearity used in modern LLMs: $f(x) = x \cdot \sigma(x)$.

### SwiGLU MLP
The modern standard for Feed-Forward networks (replacing standard ReLU MLPs).

$$\text{SwiGLU}(x) = \text{Down}(\text{SiLU}(\text{Gate}(x)) \odot \text{Up}(x))$$

* **Flow:** Split input $\to$ Gate/Up projection $\to$ Element-wise mult $\to$ Down projection.
* **Hidden Dim:** Expands to $4D$ internally.

### Causal Self-Attention
Single-head implementation with causal masking.
1. **Project:** $Q, K, V$ via linear layers.
2. **Score:** $S = \frac{QK^T}{\sqrt{D_{head}}}$
3. **Mask:** Apply causal mask (set future positions to $-\infty$).
4. **Attend:** $A = \text{softmax}(S) \cdot V$
5. **Output:** Final linear projection $W_o$.

### Transformer Block (Pre-Norm)
Standard GPT-2/LLaMA architecture using residual connections.

```python
x = x + attention(rmsnorm(x))
x = x + swiglu(rmsnorm(x))
```


## 2. The TinyGPT Model Structure

**Input:** Token Indices `(B, T)`  
**Output:** Logits `(B, T, V)`

### Forward Pass:
- **Embedding:** Look up token vectors.  
- **Stack:** Pass through `N` Transformer Blocks.  
- **Final Norm:** RMSNorm.  
- **Head:** Linear projection to vocabulary size.

---

## 3. Training & Optimization Strategy

### Objective
Next-Token Prediction (Self-Supervised Learning).  

Given context `t[0:i]`, predict `t[i+1]`.

- **Inputs:** `tokens[:, :-1]`  
- **Targets:** `tokens[:, 1:]`  
- **Loss:** Cross Entropy.

### Gradient Descent (Conceptual)

θ_{t+1} = θ_t - α ∇_θ L

Implemented via `jax.value_and_grad`.

---

## ⚠️ Important Note: Why this Phase is NOT Trainable

### Current State:
The model is built using standard Python classes and objects to make the code readable and explicit.

### The Limitation:
JAX transformations (`jit`, `grad`, and libraries like Optax) require models to be PyTrees (nested arrays/dictionaries), not stateful Python class instances.

### Next Step (Phase 2):
Refactor this logic into functional PyTrees (or use Equinox/Flax) to enable JIT compilation and actual training loops.  

Phase 1 is strictly for validation and understanding.

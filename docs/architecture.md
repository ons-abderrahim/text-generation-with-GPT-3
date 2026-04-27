# Transformer Architecture — GPT-Style Models

## Overview

GPT models are **decoder-only transformers** trained with a **causal language modeling** (CLM) objective: predict the next token given all previous tokens.

This document covers the key architectural components and how they relate to fine-tuning.

---

## 1. Input Representation

### Token Embeddings

Raw text is first converted to integer token IDs via a **Byte Pair Encoding (BPE)** tokenizer, then mapped to dense vectors via a learned embedding matrix `E ∈ ℝ^{V × d}` where `V` is the vocabulary size and `d` is the model dimension.

### Positional Embeddings

Since attention is permutation-invariant, position information must be injected explicitly.

- **GPT-2**: Learned absolute positional embeddings `P ∈ ℝ^{L × d}` (max 1024 positions)
- **GPT-Neo**: Uses **Rotary Position Embeddings (RoPE)** — relative and length-extrapolatable
- **LLaMA**: Also uses RoPE with grouped-query attention

The input to the first transformer block is: `X₀ = E[token_ids] + P[positions]`

---

## 2. Transformer Block

Each of the `N` transformer blocks has two sub-layers:

### 2a. Causal Multi-Head Self-Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**Causal masking**: The attention score matrix is masked with `-inf` above the diagonal, preventing any token from attending to future tokens:

```
Mask[i, j] = 0    if j ≤ i   (can attend)
           = -inf  if j > i   (masked)
```

Multi-head attention runs `h` attention heads in parallel:
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_o

where headᵢ = Attention(X·W_Qᵢ, X·W_Kᵢ, X·W_Vᵢ)
```

### 2b. Feed-Forward Network (MLP)

After attention, a position-wise feed-forward network expands and contracts the representation:

```
FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂
```

The hidden dimension is typically `4 × d_model`.

### 2c. Residual Connections & Layer Normalization

GPT-2 uses **Pre-LN** (layer norm before each sub-layer):

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-LN is more training-stable than the original Post-LN formulation.

---

## 3. Language Model Head

After `N` transformer blocks, a final layer norm and linear projection map the hidden states to vocabulary logits:

```
logits = LayerNorm(x_N) · W_lm   where W_lm ∈ ℝ^{d × V}
```

The weight matrix `W_lm` is often **tied** with the token embedding matrix `E` to reduce parameters.

---

## 4. Training Objective

The model is trained to minimize **cross-entropy loss** over all positions:

```
L = -1/T · Σₜ log P(token_t | token_{<t})
```

During fine-tuning, this exact objective is used — we simply continue training on the domain-specific corpus.

---

## 5. Fine-Tuning Considerations

| Aspect | Full Fine-Tuning | LoRA (PEFT) |
|--------|-----------------|-------------|
| Trainable params | 100% | ~0.1–1% |
| VRAM usage | High | Low |
| Training speed | Moderate | Faster |
| Catastrophic forgetting risk | Higher | Lower |
| Best for | Small–medium models | Large models |

### LoRA (Low-Rank Adaptation)

LoRA decomposes weight updates into low-rank matrices:

```
W' = W + ΔW = W + B · A

where A ∈ ℝ^{r × d}, B ∈ ℝ^{d × r}, r << d
```

Only `A` and `B` are trained, dramatically reducing memory requirements.

---

## 6. Inference & Decoding

At inference time, tokens are generated autoregressively:

```
for t in range(max_new_tokens):
    logits = model(input_ids)[:, -1, :]   # Last token's logits
    probs = softmax(logits / temperature)
    next_token = sample(probs)             # Or argmax for greedy
    input_ids = concat(input_ids, next_token)
```

See [metrics_explained.md](metrics_explained.md) for details on evaluating generated output quality.

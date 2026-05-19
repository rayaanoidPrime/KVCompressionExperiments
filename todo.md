# KV Cache Compression Framework — Integration Roadmap & Spec

## Overview

This document is the authoritative build spec for a modular KV cache compression research framework targeting x86 CPU inference with HuggingFace Transformers as the research backend and llama.cpp as the deployment validation backend. It covers architecture, per-method implementation specs, evaluation protocol, and a phased build order.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Core Infrastructure](#2-core-infrastructure)
3. [Hook Manager](#3-hook-manager)
4. [Method Specs](#4-method-specs)
5. [Evaluation Protocol](#5-evaluation-protocol)
6. [Build Phases](#6-build-phases)
7. [File Layout](#7-file-layout)
8. [Testing Checklist](#8-testing-checklist)
9. [Known Constraints & Gotchas](#9-known-constraints--gotchas)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        model.generate()                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │        HookManager          │
              │  (registered before call)   │
              └──┬──────────────────────┬───┘
                 │                      │
    ┌────────────▼──────┐   ┌───────────▼───────────┐
    │  AttentionHook    │   │     QueryHook          │
    │  (post self_attn) │   │  (post q_proj)         │
    └────────────┬──────┘   └───────────┬────────────┘
                 │                      │
                 └──────────┬───────────┘
                            │
              ┌─────────────▼─────────────────┐
              │      BaseCompressedCache       │
              │      (DynamicCache subclass)   │
              │                               │
              │  receive_attn()               │
              │  receive_queries()            │
              │  receive_pre_rope_keys()      │  ← KVQuant only
              │  receive_metadata()           │
              │                               │
              │  update()                     │
              │    └── compress()             │
              │          └── [dispatch]       │
              │               ├── keepkv      │
              │               ├── h2o         │
              │               ├── snapkv      │
              │               ├── token_merge │
              │               ├── kivi        │
              │               ├── kvquant     │
              │               └── turboquant  │
              └───────────────────────────────┘
```

### Design Principles

- **Single entry point**: all methods are accessed via the same `cache = SomeCache(...)` + `HookManager(model, cache)` pattern. `model.generate()` call signature never changes.
- **Non-destructive by default**: methods return de-quantized / de-compressed tensors so the rest of the HuggingFace pipeline consumes normal tensors. Compressed representations are stored internally for analysis.
- **Budget-first**: every method accepts a `budget` (max KV entries to retain per layer) as its primary constraint. Method-specific hyperparameters are secondary.
- **CPU-first**: no Triton, no CUDA-only ops. Everything runs on pure PyTorch with `torch.compile`-safe patterns.
- **Composable**: quantization methods (KIVI, KVQuant, TurboQuant) can be stacked on top of eviction methods (H2O, SnapKV, KeepKV) since eviction reduces length and quantization reduces bit-width independently.

---

## 2. Core Infrastructure

### 2.1 `BaseCompressedCache`

**File:** `cache/base.py`

**Inherits from:** `transformers.cache_utils.DynamicCache`

#### State

```python
self.budget: int                          # max KV entries per layer
self.n_layers: int                        # set on first update() call
self._attn_weights: dict[int, Tensor]     # layer_idx → [B, H, q, kv]
self._query_vectors: dict[int, Tensor]    # layer_idx → [B, H, q, d]
self._pre_rope_keys: dict[int, Tensor]    # layer_idx → [B, H, q, d]  (KVQuant)
self._metadata: dict[int, dict]           # layer_idx → arbitrary dict
self.compression_stats: dict[int, dict]   # populated after each compress()
```

#### Interface

```python
class BaseCompressedCache(DynamicCache):

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Called by HuggingFace on every forward pass per layer.
        Runs compress() if budget is exceeded, then delegates to super().
        """

    def compress(self, layer_idx, key_states, value_states) -> tuple[Tensor, Tensor]:
        """
        Override per method. Receives raw KV tensors.
        Must return tensors of same dtype and shape[:-2] as input
        (seq dimension may shrink, head/dim dimensions must be preserved).
        """
        raise NotImplementedError

    def should_compress(self, layer_idx) -> bool:
        """True if current seq length exceeds budget."""

    def receive_attn(self, layer_idx, attn_weights: Tensor):
        """Called by AttentionHook. Stores raw weights."""

    def receive_queries(self, layer_idx, query_states: Tensor):
        """Called by QueryHook. Stores raw query vectors."""

    def receive_pre_rope_keys(self, layer_idx, key_states: Tensor):
        """Called by PreRopeKeyHook. Stores keys before RoPE. KVQuant only."""

    def receive_metadata(self, layer_idx: int, metadata: dict):
        """
        Called manually before generate().
        layer_idx=-1 for global metadata.
        """

    def record_stats(self, layer_idx, original_len, compressed_len):
        """Updates self.compression_stats."""
```

#### Contracts

- `compress()` is only called when `should_compress()` is `True`.
- `compress()` must not modify `self.key_cache` or `self.value_cache` directly — it returns new tensors and `update()` handles storage.
- `receive_*` methods are no-ops on base class — safe to call even if a subclass doesn't use that signal.

---

### 2.2 Composable Cache

**File:** `cache/composed.py`

Allows stacking an eviction method with a quantization method:

```python
class ComposedCache(BaseCompressedCache):
    """
    eviction_cache: handles length reduction (H2O, SnapKV, KeepKV)
    quant_cache:    handles bit-width reduction (KIVI, KVQuant, TurboQuant)
    Order: evict first (reduce seq length), then quantize (reduce bit-width).
    """
    def __init__(self, eviction_cache, quant_cache):
        ...

    def compress(self, layer_idx, K, V):
        K, V = self.eviction_cache.compress(layer_idx, K, V)
        K, V = self.quant_cache.compress(layer_idx, K, V)
        return K, V
```

---

## 3. Hook Manager

**File:** `hooks/manager.py`

### 3.1 Hook Types

Hook	Attach point	Fires	Calls
`AttentionWeightHook`	`layer.self_attn` (post)	After attention computed	`cache.receive_attn()`
`QueryVectorHook`	`layer.self_attn.q_proj` (post)	After Q projection	`cache.receive_queries()`
`PreRopeKeyHook`	`layer.self_attn.k_proj` (post)	After K projection, before RoPE	`cache.receive_pre_rope_keys()`


### 3.2 Interface

```python
class HookManager:
    def __init__(self, model, cache: BaseCompressedCache):
        ...

    def register(self,
                 attn_weights: bool = False,
                 query_vectors: bool = False,
                 pre_rope_keys: bool = False):
        """Register only the hooks the method actually needs."""

    def remove(self):
        """Remove all registered hooks. Always call this after generate()."""

    def __enter__(self): return self
    def __exit__(self, *_): self.remove()
```

### 3.3 Usage Pattern

```python
cache = KeepKVCache(budget=128)
cache.receive_metadata(-1, {"ema_alpha": 0.1})

with HookManager(model, cache) as hm:
    hm.register(attn_weights=True)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            past_key_values=cache,
            output_attentions=True,
            max_new_tokens=128,
        )
```

### 3.4 Hook Timing Guarantee

```
Layer N forward:
  k_proj.forward()  → PreRopeKeyHook  → receive_pre_rope_keys(N, K_raw)
  q_proj.forward()  → QueryVectorHook → receive_queries(N, Q)
  self_attn.forward() → AttentionWeightHook → receive_attn(N, A)
  DynamicCache.update() → compress() reads all stored signals  ✅
```

All `receive_*` calls are guaranteed to complete before `compress()` runs within the same layer's forward pass.

---

## 4. Method Specs

---

### 4.1 H2O (Heavy Hitter Oracle)

**File:** `cache/methods/h2o.py`

**Hooks needed:** `attn_weights=True`

**Core idea:** Tokens that receive high cumulative attention are "heavy hitters" and are kept; the rest are evicted.

#### Algorithm

```
1. Accumulate attention scores across decode steps:
   self._scores[layer_idx] += attn_weights.mean(dim=1)  # mean over heads

2. When should_compress() is True:
   topk_indices = scores.topk(budget).indices
   K = K[:, :, topk_indices, :]
   V = V[:, :, topk_indices, :]
   scores = scores[topk_indices]   # prune scores to match
```

#### Hyperparameters

Param	Default	Notes
`budget`	required	Max tokens to retain
`recent_window`	`16`	Always keep the last N tokens regardless of score


#### Notes

- `recent_window` prevents eviction of the most recent tokens, which are always needed for next-token prediction.
- Score accumulation should be **reset** when the cache is cleared between sequences.

---

### 4.2 SnapKV

**File:** `cache/methods/snapkv.py`

**Hooks needed:** `attn_weights=True`

**Core idea:** Rather than accumulating across all decode steps, SnapKV uses the **prefill attention pattern** to decide which tokens to keep — then fixes the selection for the entire generation. This is cheaper and empirically comparable.

#### Algorithm

```
1. During prefill (seq_len > 1), record the attention pattern:
   self._prefill_scores[layer_idx] = attn_weights.mean(dim=(0,1))  # [kv_len]

2. On first compress() call, select topk by prefill score:
   keep_indices = prefill_scores.topk(budget).indices
   K = K[:, :, keep_indices, :]
   V = V[:, :, keep_indices, :]
   self._selection_fixed[layer_idx] = True

3. On subsequent compress() calls:
   if self._selection_fixed[layer_idx]: return K, V unchanged
```

#### Hyperparameters

Param	Default	Notes
`budget`	required	
`pooling_kernel`	`5`	Optional local average pooling on scores before topk


#### Notes

- The `is_prefill` detection is: `key_states.shape[-2] > 1`.
- SnapKV is cheaper than H2O at decode time because no per-step score accumulation is needed.

---

### 4.3 KeepKV

**File:** `cache/methods/keepkv.py`

**Hooks needed:** `attn_weights=True`

**Core idea:** EMA (exponential moving average) of attention scores, giving a smooth importance estimate that weights recent attention more heavily than H2O's raw accumulation.

#### Algorithm

```
1. On each decode step, update EMA:
   if layer_idx not in self._ema_scores:
       self._ema_scores[layer_idx] = attn_weights.mean(dim=1)
   else:
       self._ema_scores[layer_idx] = (
           alpha * attn_weights.mean(dim=1) +
           (1 - alpha) * self._ema_scores[layer_idx]
       )

2. When should_compress() is True:
   keep = ema_scores.topk(budget).indices
   K, V = K[..., keep, :], V[..., keep, :]
```

#### Hyperparameters

Param	Default	Notes
`budget`	required	
`ema_alpha`	`0.1`	Higher = more weight on recent attention


---

### 4.4 Token Merging

**File:** `cache/methods/token_merge.py`

**Hooks needed:** `attn_weights=True`

**Core idea:** Instead of evicting low-importance tokens, **merge** similar adjacent tokens into a single token (weighted average). Reduces length without total information loss.

#### Algorithm

```
1. Compute pairwise cosine similarity between adjacent key vectors:
   sim = F.cosine_similarity(K[..., :-1, :], K[..., 1:, :], dim=-1)

2. Find pairs with similarity above threshold:
   merge_mask = sim > threshold

3. For each mergeable pair (i, i+1):
   w_i = attn_score[i] / (attn_score[i] + attn_score[i+1])
   K_merged = w_i * K[i] + (1 - w_i) * K[i+1]
   V_merged = w_i * V[i] + (1 - w_i) * V[i+1]

4. Replace pair with single merged token, reduce seq length by n_merges.
```

#### Hyperparameters

Param	Default	Notes
`budget`	required	Controls max merges
`similarity_threshold`	`0.9`	Cosine similarity above which tokens are mergeable


#### Notes

- Merging is **not** equivalent to eviction — perplexity degradation is typically lower.
- Requires `attn_weights` for the importance weighting step.

---

### 4.5 VLCache

**File:** `cache/methods/vlcache.py`

**Hooks needed:** `attn_weights=True`

**Core idea:** Vision-language specific method. Assigns different eviction budgets per layer (layers that attend more to image tokens get larger budgets) and applies a **modality dampening factor** that reduces the importance score of image tokens.

#### Algorithm

```
1. receive_metadata(-1, {"image_positions": set(...)})

2. On each layer, compute scores as in H2O.

3. Apply modality dampening to image token scores:
   for pos in image_positions:
       scores[pos] *= dampening_factor   # < 1.0

4. Compute per-layer budget:
   image_attn_ratio = attn_weights[..., image_positions].mean()
   layer_budget = base_budget * (1 + gamma * image_attn_ratio)

5. Evict by adjusted scores with layer-specific budget.
```

#### Hyperparameters

Param	Default	Notes
`base_budget`	required	Global budget floor
`dampening_factor`	`0.5`	Score multiplier for image tokens
`gamma`	`1.0`	Controls how aggressively image-heavy layers get more budget


---

### 4.6 KIVI

**File:** `cache/methods/kivi.py`

**Hooks needed:** None

**Core idea:** Match the quantization axis to the outlier axis. Keys have per-channel outliers → quantize per-channel. Values have per-token outliers → quantize per-token.

#### Algorithm

```
Keys (per-channel quantization):
  k_scale = K.abs().max(dim=-2).values       # [B, H, D] — max over seq dim
  K_q = round(K / k_scale) clamped to int8

Values (per-token quantization):
  v_scale = V.abs().max(dim=-1).values       # [B, H, S] — max over dim axis
  V_q = round(V / v_scale) clamped to int8

Store: K_q, k_scale, V_q, v_scale per layer
Return: dequantized K and V (same dtype as input)
```

#### Implementation

```python
def _kivi_compress(self, layer_idx, K, V):
    bits = self.bits   # default 8, target 2 (int8 storage, 2-bit effective)

    # Keys: per-channel
    k_scale = K.abs().max(dim=-2, keepdim=True).values.clamp(min=1e-8)
    K_norm = (K / k_scale).clamp(-1, 1)
    K_q = (K_norm * (2**(bits-1) - 1)).round().to(torch.int8)

    # Values: per-token
    v_scale = V.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    V_norm = (V / v_scale).clamp(-1, 1)
    V_q = (V_norm * (2**(bits-1) - 1)).round().to(torch.int8)

    self._kv_quant[layer_idx] = {
        "K_q": K_q, "k_scale": k_scale,
        "V_q": V_q, "v_scale": v_scale,
    }

    K_deq = K_q.to(torch.float32) / (2**(bits-1) - 1) * k_scale
    V_deq = V_q.to(torch.float32) / (2**(bits-1) - 1) * v_scale
    return K_deq.to(K.dtype), V_deq.to(V.dtype)
```

#### Hyperparameters

Param	Default	Notes
`bits`	`8`	Storage bit-width. 2-bit is the paper's target
`residual_length`	`32`	Last N tokens kept in fp16 (they haven't been seen enough to quantize safely)


---

### 4.7 KVQuant

**File:** `cache/methods/kvquant.py`

**Hooks needed:** `pre_rope_keys=True`

**Core idea:** KIVI + three additional improvements: pre-RoPE quantization of keys, non-uniform quantization bins fitted to activation distributions, and dense+sparse decomposition to isolate outliers.

#### Calibration (offline, one-time per model)

```python
def calibrate(self, model, calibration_texts, tokenizer, n_samples=128):
    """
    Run n_samples forward passes, collect pre-RoPE key distributions
    and value distributions per layer. Fit Lloyd-Max quantizers.
    Stores self._key_bins[layer] and self._value_bins[layer].
    """
```

#### Algorithm

```
Pre-RoPE keys (from receive_pre_rope_keys()):
  1. Identify outlier channels: abs(K) > threshold (e.g. 99th percentile)
  2. K_sparse = zeros_like(K); K_sparse[outlier_mask] = K[outlier_mask]
  3. K_dense = K.clone(); K_dense[outlier_mask] = 0
  4. Quantize K_dense using non-uniform bins (Lloyd-Max): K_dense_q

Post-RoPE passthrough:
  - K returned to model = dequant(K_dense_q) + K_sparse  (reconstructed)

Values (same as KIVI: per-token, but with non-uniform bins):
  5. Quantize V with non-uniform per-token bins
```

#### Hyperparameters

Param	Default	Notes
`bits`	`4`	Target bit-width
`outlier_threshold_pct`	`99`	Percentile above which a value is treated as outlier
`sparse_ratio`	`0.01`	Fraction of entries kept in fp16 sparse component
`n_calibration_samples`	`128`	Samples for Lloyd-Max bin fitting


#### Notes

- **Hardest method to integrate** because `pre_rope_keys` requires hooking inside `self_attn` before positional encoding — this is model-architecture specific.
- For Llama-based models: RoPE is applied in `self_attn.forward()` after `k_proj`. The hook on `k_proj` output is the right place.
- For non-Llama models: verify RoPE application location before registering this hook.
- Implement and validate KIVI first. KVQuant is a direct extension.

---

### 4.8 TurboQuant

**File:** `cache/methods/turboquant.py`

**Hooks needed:** None

**Core idea:** Apply a random rotation to spread information uniformly across all coordinates, enabling near-optimal per-coordinate scalar quantization without any calibration data. A 1-bit JL sketch on residuals corrects inner-product bias.

#### Initialization (once per model, not per forward pass)

```python
def _init_rotation(self, dim: int):
    torch.manual_seed(self.seed)   # deterministic, reproducible
    G = torch.randn(dim, dim)
    self._R, _ = torch.linalg.qr(G)              # random orthogonal matrix
    S = (torch.randint(0, 2, (self.sketch_dim, dim)) * 2 - 1).float()
    self._S = S / math.sqrt(self.sketch_dim)      # normalized JL matrix
```

#### Algorithm

```
Step 1: Rotate
  K_rot = K @ R.T          # R is the fixed random orthogonal matrix
  V_rot = V @ R.T          # same R for queries at attention time

Step 2: Per-coordinate scalar quantization
  scale = K_rot.abs().max(dim=-1, keepdim=True).values
  K_q = round(K_rot / scale * 127).clamp(-127, 127).to(int8)

Step 3: JL residual sketch (corrects inner product bias)
  K_deq = K_q.float() / 127 * scale
  residual = K_rot - K_deq
  K_sketch = sign(residual @ S.T)    # [B, H, seq, sketch_dim]

Store: K_q, scale, K_sketch per layer
Return: (K_deq @ R).to(original_dtype)   — un-rotate before returning
```

#### Inner product correction at attention time

> ⚠️ **Note:** Full residual correction requires modifying how attention scores are computed — the sketch correction term `Q_rot @ S.T @ K_sketch.T` must be added to `Q @ K.T`. This is not possible in standard HuggingFace eager mode without patching `self_attn.forward()`.

**Practical options:**

Option	Fidelity	Effort
Store sketch, skip correction (baseline)	Lossy but functional	Lowest
Patch `self_attn.forward()` per model	Full correction	Medium
Port to llama.cpp	Full correction + real latency	Highest


**Recommended:** Implement without correction first. Measure PPL. Add correction patch if degradation is unacceptable.

#### Hyperparameters

Param	Default	Notes
`bits`	`8`	Scalar quantization bit-width
`sketch_dim`	`64`	Dimension of JL residual sketch
`seed`	`42`	Random seed for rotation matrix


---

### 4.9 Attention Matching

**File:** `cache/methods/attn_match.py`

**Hooks needed:** `query_vectors=True`, `attn_weights=True`

**Core idea:** Solve an optimization problem to find compressed KV pairs `(Ck, Cv)` such that the local attention output and attention mass are matched as closely as possible. A bias term `β` corrects for mass mismatch when `t < T`.

#### Algorithm

```
Given: Q_ref (from calibration), current attention output A_out, attention mass M

Solve: min over (Ck, Cv, β) of:
    || softmax(Q_ref @ Ck.T / sqrt(d) + β) @ Cv - A_out ||²
    + λ * || mass(Q_ref, Ck, β) - M ||²

β corrects for the fact that compressed cache has fewer tokens
(mass is lower), which would shift attention distribution.
```

#### Practical HuggingFace Approximation

Full β injection requires a custom attention kernel. In HF eager mode, approximate via **key scaling**:

```python
# Scale keys up to compensate for reduced mass
scale_factor = math.sqrt(original_seq_len / compressed_seq_len)
Ck = Ck * scale_factor
```

#### Calibration

```python
# Collect reference queries from representative prompts
def calibrate(self, model, prompts, tokenizer):
    """Stores self._Q_ref[layer_idx] for use in the optimization."""
```

#### Hyperparameters

Param	Default	Notes
`budget`	required	
`lambda_mass`	`0.1`	Weight of mass-matching term
`n_calibration_samples`	`32`	Number of prompts for Q_ref collection
`n_optim_steps`	`20`	Gradient steps for the inner optimization


#### Notes

- **Implement last.** Most complex method, requires calibration, inner optimization loop, and approximate β correction.
- Lower priority on CPU due to optimization overhead per decode step.

---

## 5. Evaluation Protocol

### 5.1 Metrics

Metric	Measures	How to compute
**PPL (WikiText-2)**	Quality degradation	`evaluate.load("perplexity")` on 2048-token chunks
**Compression ratio**	Length reduction	`compressed_len / original_len` per layer
**Bits per KV entry**	Memory reduction	`(bits * dim * 2) / (32 * dim * 2)` for quant methods
**Decode latency (ms/token)**	Speed	Median of `n_reps` timed decode steps, CPU only
**Memory (MB)**	RAM usage	`torch.cuda.memory_allocated()` or `psutil.Process().memory_info()` on CPU


> ⚠️ **Note:** "Bits per KV entry" not "bits per weight" — this framework compresses activations, not model weights.

### 5.2 Benchmark Harness

**File:** `eval/harness.py`

```python
@torch.inference_mode()
def benchmark(
    model,
    tokenizer,
    cache_factory,          # callable → BaseCompressedCache instance
    dataset="wikitext-2",
    n_ppl_samples=100,
    n_latency_reps=5,
    n_warmup=2,
    max_gen_len=128,
    device="cpu",
):
    results = {
        "ppl": run_ppl(model, tokenizer, cache_factory, n_ppl_samples),
        "latency_ms": run_latency(model, tokenizer, cache_factory,
                                  n_latency_reps, n_warmup, max_gen_len),
        "compression": aggregate_compression_stats(cache_factory),
    }
    return results
```

### 5.3 Baseline

Always run and record a **baseline** (no compression, standard `DynamicCache`) before benchmarking any method. All metrics are reported relative to baseline.

```python
baseline = benchmark(model, tokenizer, cache_factory=DynamicCache)
```

### 5.4 Comparison Table Format

Method	PPL (↓)	ΔPPL vs baseline	Comp. ratio (↓)	Bits/KV (↓)	Latency ms/tok (↓)
Baseline	X.XX	—	1.00	32	X.X
H2O				32	
SnapKV				32	
KeepKV				32	
KIVI (8-bit)			1.00	8	
KIVI (2-bit)			1.00	2	
KVQuant (4-bit)			1.00	4	
TurboQuant (8-bit)			1.00	8	
H2O + KIVI				8	


---

## 6. Build Phases

### Phase 1 — Core Infrastructure ✅ Build first

**Goal:** Get a working end-to-end loop before any compression logic.

Task	File	Done?
Implement `BaseCompressedCache` with no-op `compress()`	`cache/base.py`	
Implement `HookManager` with `register()` / `remove()` / context manager	`hooks/manager.py`	
Verify baseline PPL matches `DynamicCache` exactly	`tests/test_baseline.py`	
Implement benchmark harness skeleton	`eval/harness.py`	


**Exit criterion:** Running `model.generate()` with `BaseCompressedCache` produces identical output to `DynamicCache`.

---

### Phase 2 — Eviction Methods

**Goal:** Validate that the hook → `receive_attn` → `compress` → evict pipeline works correctly.

**Order:** H2O → SnapKV → KeepKV → Token Merging → VLCache (last, needs multimodal model)

Task	File	Done?
H2O with cumulative score + recent_window	`cache/methods/h2o.py`	
SnapKV with prefill-fixed selection	`cache/methods/snapkv.py`	
KeepKV with EMA scoring	`cache/methods/keepkv.py`	
Token Merging with cosine similarity	`cache/methods/token_merge.py`	
VLCache with modality dampening	`cache/methods/vlcache.py`	
PPL + latency benchmarks for all eviction methods	`eval/run_eviction.py`	


**Exit criterion:** H2O at `budget=128` on a 512-token input shows < 0.5 PPL increase vs baseline.

---

### Phase 3 — Quantization Methods

**Goal:** Implement and validate the three quantization approaches.

**Order:** KIVI → TurboQuant → KVQuant (last, needs pre-RoPE hook)

Task	File	Done?
KIVI: per-channel keys, per-token values	`cache/methods/kivi.py`	
KIVI calibration-free 8-bit smoke test	`tests/test_kivi.py`	
TurboQuant: rotation init + scalar quant + JL sketch	`cache/methods/turboquant.py`	
TurboQuant without residual correction (baseline)		
KVQuant: calibration pass + Lloyd-Max bins	`cache/methods/kvquant.py`	
KVQuant: pre-RoPE hook (Llama-specific)	`hooks/manager.py`	
KVQuant: dense+sparse decomposition		
PPL + bits/KV benchmarks for all quant methods	`eval/run_quant.py`	


**Exit criterion:** KIVI at 8-bit shows < 0.1 PPL increase vs baseline. TurboQuant within 0.3 PPL of baseline.

---

### Phase 4 — Composition

**Goal:** Stack eviction + quantization and verify they compose correctly.

Task	File	Done?
Implement `ComposedCache`	`cache/composed.py`	
Test H2O + KIVI composition	`tests/test_composed.py`	
Run full comparison table (all methods + compositions)	`eval/run_all.py`	


---

### Phase 5 — llama.cpp Deployment Validation

**Goal:** Validate the best 2–3 methods from Phase 4 under real x86 latency conditions.

Task	Notes	Done?
Convert model to GGUF	`llama.cpp/convert.py`	
Baseline: `--cache-type-k fp16` (fastest on x86)		
Implement eviction method equivalent in llama.cpp C++	Method with best PPL/ratio tradeoff from Phase 4	
Implement KIVI equivalent: per-channel keys, per-token values	`--cache-type-k q8_0` is close; custom impl for 2-bit	
Record real tokens/sec on target x86 hardware		


---

### Phase 6 — Attention Matching (Optional)

**Goal:** Implement the most complex method last when the framework is stable.

Task	File	Done?
Calibration pass for Q_ref collection	`cache/methods/attn_match.py`	
Inner optimization loop (20 steps per decode)		
Key scaling as β approximation		
PPL benchmark vs other methods		


---

## 7. File Layout

```
kv_compress/
│
├── cache/
│   ├── base.py               # BaseCompressedCache
│   ├── composed.py           # ComposedCache (eviction + quant stacking)
│   └── methods/
│       ├── h2o.py
│       ├── snapkv.py
│       ├── keepkv.py
│       ├── token_merge.py
│       ├── vlcache.py
│       ├── kivi.py
│       ├── kvquant.py
│       ├── turboquant.py
│       └── attn_match.py
│
├── hooks/
│   └── manager.py            # HookManager + all hook types
│
├── eval/
│   ├── harness.py            # benchmark() entrypoint
│   ├── run_eviction.py       # Phase 2 benchmark script
│   ├── run_quant.py          # Phase 3 benchmark script
│   └── run_all.py            # Phase 4 full comparison table
│
├── tests/
│   ├── test_baseline.py      # BaseCompressedCache == DynamicCache
│   ├── test_kivi.py          # KIVI 8-bit PPL smoke test
│   ├── test_composed.py      # ComposedCache correctness
│   └── test_hooks.py         # Hook timing and signal correctness
│
├── scripts/
│   └── calibrate_kvquant.py  # Offline calibration for KVQuant
│
└── README.md
```

---

## 8. Testing Checklist

### Per-method tests (run for each new method before benchmarking)

- [ ] **Shape test**: output `K`, `V` have same `batch`, `heads`, `dim` as input. Seq dimension ≤ input.
- [ ] **Dtype test**: output dtype matches input dtype (no silent fp32 upcast left in return value).
- [ ] **Budget test**: after compression, `K.shape[-2] <= budget`.
- [ ] **Dequant roundtrip** (quant methods): `(K_deq - K).abs().max() < tolerance` at 8-bit (tolerance = 0.05).
- [ ] **Logit equivalence** (quant methods, 32-bit): `compress()` at 32-bit should reproduce baseline logits exactly.
- [ ] **Hook cleanup**: after `hm.remove()`, no hooks remain on the model. Verify with `len(model._forward_hooks)`.
- [ ] **Stat recording**: `cache.compression_stats[layer_idx]` is populated after a forward pass.
- [ ] **Multi-sequence reset**: cache stats and scores reset correctly between independent sequences.

### Integration tests

- [ ] `BaseCompressedCache` with no-op `compress()` produces identical logits to `DynamicCache`.
- [ ] `ComposedCache(H2O, KIVI)` output is within tolerance of running H2O alone (then KIVI alone).
- [ ] `HookManager` context manager leaves zero hooks on model after `__exit__`.

---

## 9. Known Constraints & Gotchas

Issue	Detail	Mitigation
**`output_attentions=True` required**	Without this flag, `attn_weights` is `None` and all eviction methods silently receive nothing.	Add an assertion in `receive_attn()`: `assert attn_weights is not None`. Add a check in `HookManager.register()` that warns if `attn_weights=True` is set but `output_attentions` is not.
**RoPE location is model-specific**	Llama 3.x applies RoPE inside `self_attn.forward()`. Other architectures differ.	Add a model-family check in `HookManager._register_pre_rope_hook()`. Document supported architectures.
**`torch.compile` + DynamicCache**	`torch.compile` with `inductor` backend crashes on `DynamicCache`'s dynamic Python ops.	Do not compile the model when using any cache subclass. Use `torch.set_num_threads()` for CPU speedup instead.
**Recent tokens and eviction**	Evicting the most recently generated tokens breaks generation coherence.	Always keep the last `recent_window` (default 16) tokens in any eviction method.
**Prefill vs decode detection**	`should_compress()` and SnapKV's prefill detection depend on `key_states.shape[-2] > 1`. This can misfire on batched prefills.	Use `cache_kwargs.get("cache_position")` to detect prefill more reliably when available.
**Score tensor growth**	Cumulative score tensors (H2O, KeepKV) grow with sequence length before first eviction.	Pre-allocate with `torch.zeros(budget * 2)` and use a circular buffer if memory is a concern.
**TurboQuant un-rotation**	The random rotation `R` must be applied to queries at attention time for inner products to be correct. In HF eager mode, queries are never exposed post-rotation.	Either patch `self_attn.forward()` or accept approximate inner products without residual correction in Phase 3.
**KVQuant calibration stale**	If calibration was run on a different task distribution, Lloyd-Max bins may not be optimal.	Document calibration dataset and re-run if switching domains.
**CPU latency = relative indicator only**	Absolute CPU latency numbers are not publication-quality.	Use HF benchmarks for PPL and compression ratio reporting. Use llama.cpp for latency claims.
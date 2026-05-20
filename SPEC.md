# KV Cache Compression Framework — Build Spec

## Overview

This document is the authoritative build spec for a modular KV cache compression research
framework targeting x86 CPU inference. The research backend is HuggingFace Transformers;
the deployment validation backend is llama.cpp.

The framework is built around a **post-forward hook on `self_attn`** (`BasePress`) rather than
a `DynamicCache` subclass. This design choice eliminates the causal-mask mismatch bug that
affects eviction-based `DynamicCache` subclasses in HF's eager and SDPA attention backends,
while providing access to attention weights, hidden states, and the full cache in a single
hook with no inter-hook coordination.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Core Infrastructure](#2-core-infrastructure)
3. [Method Specs](#3-method-specs)
4. [Evaluation Protocol](#4-evaluation-protocol)
5. [Build Phases](#5-build-phases)
6. [File Layout](#6-file-layout)
7. [Testing Checklist](#7-testing-checklist)
8. [Known Constraints & Gotchas](#8-known-constraints--gotchas)

---

## 1. System Architecture

```
model.generate()
      │
      ▼
self_attn.forward()          ← runs at full sequence length, mask is correct
      │
      │  produces: output = (hidden_out, attn_weights, ...)
      │            kwargs  = {hidden_states, past_key_values, cache_position, ...}
      ▼
BasePress.forward_hook()     ← registered as post-forward hook on self_attn
      │
      ├── should_compress()?      ← prefill detection via cache_position
      │
      ├── extract_keys_and_values()  ← handles DynamicCache and QuantizedCache
      │
      ├── compress()              ← dispatches to method subclass
      │     ├── H2OPress
      │     ├── SnapKVPress
      │     ├── KeepKVPress
      │     ├── TokenMergePress
      │     ├── VLCachePress
      │     ├── KIVIPress
      │     ├── KVQuantPress      ← also registers opt-in k_proj hook
      │     ├── TurboQuantPress
      │     └── AttnMatchPress
      │
      └── _write_to_cache()       ← writes back; dispatches DynamicCache / QuantizedCache
```

### Why post-forward hook, not DynamicCache subclass

| Issue | DynamicCache subclass | BasePress hook |
|---|---|---|
| Causal mask after eviction | ❌ Crashes in eager/SDPA — mask shape lags behind shortened KV | ✅ Hook fires after attention; mask never sees compressed tensor |
| Hook coordination | ⚠️ Three hooks (k_proj, q_proj, self_attn) with ordering contracts | ✅ Single hook, all signals available at call time |
| `QuantizedCache` support | ❌ Not addressed | ✅ `_write_to_cache` dispatches correctly |
| Prefill detection | ⚠️ `key_states.shape[-2] > 1` — breaks with speculative decoding | ✅ `cache_position[-1] <= q_len` — matches HF's own logic |
| Hidden states in compress | ❌ Not available | ✅ Available via `kwargs["hidden_states"]` |
| HF API stability | ❌ `key_cache`/`value_cache` are stale; current API uses `CacheLayer` objects | ✅ Only touches `cache.layers[layer_idx]` which is stable |

### Design principles

- **Single hook per layer.** All signals (attention weights, keys, values, hidden states,
  cache position, position embeddings) are available in one `forward_hook` call.
  No inter-hook timing contracts needed.
- **Post-forward, pre-write.** The hook fires after `self_attn.forward()` completes.
  Attention was computed at full length with a valid causal mask. Compression only
  affects what is written back for the next step.
- **Non-destructive return.** Quantization methods dequantize before writing back to the
  dense cache layer, preserving dtype. Compressed representations are stored internally
  for analysis.
- **Budget-first.** Every method accepts `budget` (max KV entries per layer) as its
  primary constraint. Method-specific hyperparameters are secondary.
- **CPU-first.** No Triton, no CUDA-only ops. Pure PyTorch.
- **Composable.** `ComposedPress` stacks an eviction press with a quantization press at
  the press level, not the cache level.

---

## 2. Core Infrastructure

### 2.1 `BasePress`

**File:** `press/base.py`

```python
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import torch
from torch import nn
from transformers import Cache, PreTrainedModel, QuantizedCache

from utils import extract_keys_and_values

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.

    Compression is applied via a post-forward hook on each attention layer.
    Subclasses implement compress() to define their specific logic.

    The hook fires after self_attn.forward() completes, meaning:
    - Attention was computed at full length with a correct causal mask.
    - Keys and values in the cache reflect the full sequence.
    - Compression only affects what is stored for the next decode step.
    """

    decoding: bool = False  # apply compression on decode steps too (not just prefill)

    def __post_init__(self) -> None:
        if self.decoding:
            logger.warning(
                "Decoding compression enabled. Compression will run on every "
                "decode step, which may significantly increase latency."
            )

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        """
        Optional hook called once, just before hooks are registered.
        Override to initialise anything requiring model access:
        hidden_size, num_heads, num_kv_heads, head_dim, etc.
        """
        pass

    def is_prefilling(self, hidden_states: torch.Tensor, kwargs: dict) -> bool:
        """
        True if we are in the prefill phase.

        Uses cache_position to detect phase:
          - Prefill:  cache_position[-1] <= q_len  (all positions are new)
          - Decode:   cache_position[-1] >  q_len  (one new position appended)

        Falls back to True (assume prefill) if cache_position is absent.
        """
        q_len = hidden_states.shape[1]
        cache_position = kwargs.get("cache_position", None)
        if cache_position is None:
            return True
        return cache_position[-1] <= q_len

    def should_compress(self, hidden_states: torch.Tensor, kwargs: dict) -> bool:
        if self.is_prefilling(hidden_states, kwargs):
            return True
        return self.decoding

    @staticmethod
    def _write_to_cache(
        cache: Cache,
        cache_layer: Any,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Write compressed keys and values back to the cache layer.
        Handles both DynamicCache layers and QuantizedCache layers.
        """
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys   = cache_layer._quantize(keys,   axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys   = torch.zeros(0, dtype=keys.dtype,   device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys   = keys
            cache_layer.values = values

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Core compression logic. Must be overridden by each subclass.

        Parameters
        ----------
        module : nn.Module
            The attention layer (self_attn). Provides layer_idx, num_heads,
            num_key_value_heads, head_dim, etc.
        hidden_states : torch.Tensor
            Input to the attention layer. Shape: [batch, seq_len, hidden_dim].
            For decode steps with decoding=True, shape is [batch, 1, hidden_dim].
        keys : torch.Tensor
            Full KV cache keys for this layer.
            Shape: [batch, num_kv_heads, seq_len, head_dim].
        values : torch.Tensor
            Full KV cache values for this layer.
            Shape: [batch, num_kv_heads, seq_len, head_dim].
        attentions : torch.Tensor | None
            Attention weight matrix from this layer's forward pass.
            Shape: [batch, num_heads, q_len, kv_len].
            None if the model was not run with output_attentions=True.
        kwargs : dict
            Full kwargs dict from the attention layer's forward call.
            Contains: cache_position, position_embeddings, attention_mask, etc.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Compressed (keys, values). Eviction methods return shorter seq dim.
            Quantization methods return dequantized tensors at same seq length,
            storing compressed form internally.
        """
        raise NotImplementedError

    def forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict,
        output: list,
    ):
        """
        Post-forward hook registered on each self_attn layer.

        Execution order within one layer's forward pass:
          k_proj.forward()
          q_proj.forward()
          self_attn.forward()  ← attention computed at full length, mask valid
            └── forward_hook() ← fires here; compress and write back
        """
        hidden_states = kwargs["hidden_states"]
        cache         = kwargs.get("past_key_values")

        if cache is None:
            return output

        if not self.should_compress(hidden_states, kwargs):
            return output

        cache_layer = cache.layers[module.layer_idx]
        keys, values = extract_keys_and_values(cache, module.layer_idx)

        # output[1] is attn weights when output_attentions=True; else None
        attentions = output[1] if len(output) > 1 else None

        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

        self._write_to_cache(cache, cache_layer, keys, values)

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager. Registers hooks on all attention layers before the
        generate() call and removes them cleanly on exit, even if an exception
        is raised inside the block.

        Usage
        -----
        press = H2OPress(budget=128)
        with press(model):
            outputs = model.generate(**inputs, past_key_values=cache)
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(
                f"Model type {type(model).__name__} is not in the tested list. "
                f"Supported: {[m.__name__ for m in SUPPORTED_MODELS]}"
            )

        self.post_init_from_model(model)
        hooks = []

        try:
            lm = (
                model.model.language_model
                if hasattr(model.model, "language_model")
                else model.model
            )

            for layer in lm.layers:
                # Propagate the shared rotary embedding to the attention layer
                # only when the layer doesn't already have its own instance.
                if (
                    not hasattr(layer.self_attn, "rotary_emb")
                    or layer.self_attn.rotary_emb is None
                ):
                    layer.self_attn.rotary_emb = lm.rotary_emb

                hooks.append(
                    layer.self_attn.register_forward_hook(
                        self.forward_hook, with_kwargs=True
                    )
                )

                # Opt-in pre-RoPE key hook for KVQuantPress only
                if getattr(self, "_needs_pre_rope_keys", False):
                    hooks.append(
                        layer.self_attn.k_proj.register_forward_hook(
                            self._pre_rope_key_hook
                        )
                    )

            yield

        finally:
            for h in hooks:
                h.remove()
```

#### Contracts

- `compress()` receives the full, unmodified KV cache for the layer. It returns
  `(keys, values)` of any sequence length ≤ input.
- `compress()` must preserve `batch`, `num_kv_heads`, and `head_dim` dimensions.
- `compress()` must return tensors in the same dtype as its inputs (no silent fp32 upcasts).
- `_write_to_cache()` is the only method that modifies `cache.layers[layer_idx]`.
  `compress()` must not touch the cache directly.
- `should_compress()` must be checked before `extract_keys_and_values()` is called.

---

### 2.2 GQA / MQA head handling

Llama 3.x, Mistral, Qwen2, and Gemma 2 all use grouped-query attention (GQA) where
`num_heads != num_kv_heads`. Attention weights have shape `[B, num_heads, q, kv]` but
keys and values have shape `[B, num_kv_heads, kv, d]`.

Every eviction method that uses attention weights to select KV positions must map head
indices correctly:

```python
def _importance_from_attn(
    attn_weights: torch.Tensor,   # [B, num_heads, q, kv]
    module: nn.Module,
) -> torch.Tensor:
    """
    Aggregate attention weights to KV-head granularity.

    For GQA: average the num_heads/num_kv_heads query heads that share
    each KV head, then take mean over query positions.

    Returns: [B, num_kv_heads, kv_len]
    """
    B, H, q, kv = attn_weights.shape
    num_kv_heads = module.num_key_value_heads
    groups = H // num_kv_heads  # heads per KV head

    # [B, num_kv_heads, groups, q, kv] → mean over groups and q
    scores = attn_weights.view(B, num_kv_heads, groups, q, kv)
    return scores.mean(dim=(2, 3))  # [B, num_kv_heads, kv]
```

Call `_importance_from_attn` in every eviction method. Do **not** call
`attn_weights.mean(dim=1)` directly — that averages across all heads before accounting
for GQA grouping, producing incorrect importance scores.

---

### 2.3 `ComposedPress`

**File:** `press/composed.py`

Stacks an eviction press with a quantization press. Eviction reduces sequence length;
quantization reduces bit-width. Correct order is always evict-then-quantize: quantizing
tokens that will be dropped wastes computation.

```python
@dataclass
class ComposedPress(BasePress):
    """
    Compose an eviction press with a quantization press.

    order: evict (shorten seq) → quantize (reduce bit-width).
    """
    eviction:     BasePress = None
    quantization: BasePress = None

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        self.eviction.post_init_from_model(model)
        self.quantization.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        keys, values = self.eviction.compress(
            module, hidden_states, keys, values, attentions, kwargs
        )
        keys, values = self.quantization.compress(
            module, hidden_states, keys, values, attentions, kwargs
        )
        return keys, values
```

#### Usage

```python
press = ComposedPress(
    eviction=H2OPress(budget=128),
    quantization=KIVIPress(bits=8),
)
with press(model):
    output = model.generate(**inputs, past_key_values=cache, max_new_tokens=128)
```

#### Composable combinations

| Eviction | Quantization | Notes |
|---|---|---|
| `H2OPress` | `KIVIPress` | Recommended first composition to validate |
| `SnapKVPress` | `KIVIPress` | Cheaper prefill selection + quant |
| `H2OPress` | `TurboQuantPress` | Calibration-free quant |
| `H2OPress` | `KVQuantPress` | Highest fidelity, needs calibration |
| `KeepKVPress` | `KIVIPress` | EMA scoring + quant |

Composing two eviction methods or two quantization methods is not supported.

---

### 2.4 `extract_keys_and_values`

**File:** `utils.py`

Must handle both `DynamicCache` layers (dense tensors in `cache_layer.keys / .values`)
and `QuantizedCache` layers (dequantize from `_quantized_keys / _quantized_values`).

```python
def extract_keys_and_values(
    cache: Cache, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the full, dequantized keys and values for the given layer.

    For QuantizedCache, dequantizes the stored int tensors before returning.
    For DynamicCache, returns the dense tensors directly (no copy).
    """
    cache_layer = cache.layers[layer_idx]
    if isinstance(cache, QuantizedCache):
        keys   = cache_layer._dequantize(cache_layer._quantized_keys,   axis=cache_layer.axis_key)
        values = cache_layer._dequantize(cache_layer._quantized_values, axis=cache_layer.axis_value)
    else:
        keys   = cache_layer.keys
        values = cache_layer.values
    return keys, values
```

---

## 3. Method Specs

All method files live in `press/methods/`.

Signal availability in `compress()` without any extra hooks:

| Signal | Source | Always available? |
|---|---|---|
| `keys` | `extract_keys_and_values()` | ✅ |
| `values` | `extract_keys_and_values()` | ✅ |
| `hidden_states` | `kwargs["hidden_states"]` | ✅ |
| `attentions` | `output[1]` | Only when `output_attentions=True` |
| `cache_position` | `kwargs["cache_position"]` | ✅ (HF always passes this) |
| `position_embeddings` | `kwargs["position_embeddings"]` | ✅ on Llama/Qwen2 |
| Pre-RoPE keys | opt-in `k_proj` hook | KVQuant only |

---

### 3.1 H2OPress — Heavy Hitter Oracle

**File:** `press/methods/h2o.py`
**Requires:** `output_attentions=True`
**Core idea:** Accumulate attention scores across decode steps. Tokens receiving the
highest cumulative attention are "heavy hitters" and are kept; the rest are evicted.

```python
@dataclass
class H2OPress(BasePress):
    budget:        int   = 128   # max KV entries to retain per layer
    recent_window: int   = 16    # always keep the last N tokens
    head_agg:      str   = "max" # "max" | "mean" | "sum" over KV-grouped heads

    def __post_init__(self):
        super().__post_init__()
        self._scores: dict[int, torch.Tensor] = {}

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        assert attentions is not None, (
            "H2OPress requires output_attentions=True in model.generate()"
        )

        layer_idx = module.layer_idx
        scores = _importance_from_attn(attentions, module)  # [B, kv_heads, kv]

        if head_agg == "max":
            scores = scores.max(dim=1).values   # [B, kv]
        elif head_agg == "mean":
            scores = scores.mean(dim=1)
        else:
            scores = scores.sum(dim=1)

        scores = scores.mean(dim=0)  # [kv] — average over batch

        # Accumulate
        if layer_idx not in self._scores:
            self._scores[layer_idx] = scores
        else:
            # Pad/truncate to current seq length before accumulating
            prev = self._scores[layer_idx]
            kv_len = scores.shape[-1]
            if prev.shape[-1] < kv_len:
                prev = torch.cat(
                    [prev, torch.zeros(kv_len - prev.shape[-1], device=prev.device)]
                )
            self._scores[layer_idx] = prev[:kv_len] + scores

        seq_len = keys.shape[2]
        if seq_len <= self.budget:
            return keys, values

        acc = self._scores[layer_idx]  # [kv_len]

        # Always keep recent_window tokens regardless of score
        protected = torch.zeros(seq_len, dtype=torch.bool, device=keys.device)
        protected[-self.recent_window:] = True

        # Score unprotected positions and select topk
        scorable_budget = self.budget - self.recent_window
        unprotected_scores = acc.clone()
        unprotected_scores[-self.recent_window:] = -float("inf")
        keep_idx = unprotected_scores.topk(scorable_budget).indices

        all_idx = torch.cat([
            keep_idx,
            torch.arange(seq_len - self.recent_window, seq_len, device=keys.device)
        ]).sort().values

        # Prune accumulated scores to match
        self._scores[layer_idx] = acc[all_idx]

        return keys[:, :, all_idx, :], values[:, :, all_idx, :]

    def reset(self) -> None:
        """Call between independent sequences."""
        self._scores.clear()
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `budget` | required | Max tokens to retain |
| `recent_window` | `16` | Always-kept tail tokens |
| `head_agg` | `"max"` | How to reduce GQA heads before accumulation |

---

### 3.2 SnapKVPress

**File:** `press/methods/snapkv.py`
**Requires:** `output_attentions=True` (prefill only)
**Core idea:** Use the prefill attention pattern to fix the KV selection for the entire
generation. No per-step score accumulation needed at decode time.

```python
@dataclass
class SnapKVPress(BasePress):
    budget:          int = 128
    pooling_kernel:  int = 5     # local average pooling before topk (0 = off)

    def __post_init__(self):
        super().__post_init__()
        self._keep_idx: dict[int, torch.Tensor] = {}  # fixed per layer after prefill

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer_idx = module.layer_idx

        # Decode step: selection already fixed
        if layer_idx in self._keep_idx:
            idx = self._keep_idx[layer_idx]
            return keys[:, :, idx, :], values[:, :, idx, :]

        # Prefill: compute and fix selection
        assert attentions is not None, (
            "SnapKVPress requires output_attentions=True during prefill"
        )

        scores = _importance_from_attn(attentions, module)   # [B, kv_heads, kv]
        scores = scores.mean(dim=(0, 1))                     # [kv]

        if self.pooling_kernel > 0:
            scores = torch.nn.functional.avg_pool1d(
                scores.unsqueeze(0).unsqueeze(0),
                kernel_size=self.pooling_kernel,
                stride=1,
                padding=self.pooling_kernel // 2,
            ).squeeze()

        seq_len = keys.shape[2]
        if seq_len <= self.budget:
            return keys, values

        idx = scores.topk(self.budget).indices.sort().values
        self._keep_idx[layer_idx] = idx
        return keys[:, :, idx, :], values[:, :, idx, :]

    def reset(self) -> None:
        self._keep_idx.clear()
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `budget` | required | |
| `pooling_kernel` | `5` | Local average pooling on scores before topk; `0` disables |

---

### 3.3 KeepKVPress

**File:** `press/methods/keepkv.py`
**Requires:** `output_attentions=True`
**Core idea:** EMA of attention scores across decode steps. Gives a smooth importance
estimate that weights recent attention more heavily than H2O's raw accumulation.

```python
@dataclass
class KeepKVPress(BasePress):
    budget:     int   = 128
    ema_alpha:  float = 0.1   # higher = more weight on recent attention
    head_agg:   str   = "max"

    def __post_init__(self):
        super().__post_init__()
        self._ema: dict[int, torch.Tensor] = {}

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        assert attentions is not None
        layer_idx = module.layer_idx

        scores = _importance_from_attn(attentions, module)
        if self.head_agg == "max":
            scores = scores.max(dim=1).values
        elif self.head_agg == "mean":
            scores = scores.mean(dim=1)
        else:
            scores = scores.sum(dim=1)
        scores = scores.mean(dim=0)  # [kv]

        if layer_idx not in self._ema:
            self._ema[layer_idx] = scores
        else:
            prev = self._ema[layer_idx]
            kv_len = scores.shape[-1]
            if prev.shape[-1] < kv_len:
                prev = torch.cat(
                    [prev, torch.zeros(kv_len - prev.shape[-1], device=prev.device)]
                )
            self._ema[layer_idx] = (
                self.ema_alpha * scores +
                (1 - self.ema_alpha) * prev[:kv_len]
            )

        seq_len = keys.shape[2]
        if seq_len <= self.budget:
            return keys, values

        ema = self._ema[layer_idx]
        # Protect recent tokens
        ema_masked = ema.clone()
        ema_masked[-16:] = float("inf")
        idx = ema_masked.topk(self.budget).indices.sort().values
        self._ema[layer_idx] = ema[idx]

        return keys[:, :, idx, :], values[:, :, idx, :]

    def reset(self) -> None:
        self._ema.clear()
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `budget` | required | |
| `ema_alpha` | `0.1` | Higher → more weight on recent attention |
| `head_agg` | `"max"` | Same as H2OPress |

---

### 3.4 TokenMergePress

**File:** `press/methods/token_merge.py`
**Requires:** `output_attentions=True`
**Core idea:** Instead of dropping low-importance tokens, *merge* adjacent similar tokens
into a single weighted-average token. Reduces length without total information loss; PPL
degradation is typically lower than eviction at equal compression ratios.

```python
@dataclass
class TokenMergePress(BasePress):
    budget:               int   = 128
    similarity_threshold: float = 0.9   # cosine sim above which tokens are mergeable

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        seq_len = keys.shape[2]
        n_merges = max(0, seq_len - self.budget)
        if n_merges == 0:
            return keys, values

        # Cosine similarity between adjacent key vectors (mean over kv heads)
        k = keys.mean(dim=1)  # [B, seq, d]
        sim = torch.nn.functional.cosine_similarity(
            k[:, :-1, :], k[:, 1:, :], dim=-1
        ).mean(dim=0)  # [seq-1]

        # Importance weights from attention (GQA-aware)
        if attentions is not None:
            importance = _importance_from_attn(attentions, module).mean(dim=(0, 1))
        else:
            importance = torch.ones(seq_len, device=keys.device)

        # Greedily merge the n_merges most-similar adjacent pairs
        merged_k = list(keys.unbind(dim=2))
        merged_v = list(values.unbind(dim=2))
        imp       = list(importance)

        for _ in range(n_merges):
            if len(merged_k) < 2:
                break
            current_sim = torch.stack([
                torch.nn.functional.cosine_similarity(
                    merged_k[i].reshape(-1), merged_k[i+1].reshape(-1), dim=0
                )
                for i in range(len(merged_k) - 1)
            ])
            if current_sim.max() < self.similarity_threshold:
                break  # no more mergeable pairs
            best = current_sim.argmax().item()
            wi = imp[best] / (imp[best] + imp[best+1] + 1e-8)
            merged_k[best] = wi * merged_k[best] + (1 - wi) * merged_k[best+1]
            merged_v[best] = wi * merged_v[best] + (1 - wi) * merged_v[best+1]
            imp[best] = imp[best] + imp[best+1]
            del merged_k[best+1]
            del merged_v[best+1]
            del imp[best+1]

        keys_out   = torch.stack(merged_k, dim=2)
        values_out = torch.stack(merged_v, dim=2)
        return keys_out, values_out
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `budget` | required | Maximum post-merge length |
| `similarity_threshold` | `0.9` | Stop merging if best remaining pair is below this |

#### Notes

- The greedy loop merges at most `n_merges` times, so budget is always respected even if
  the similarity threshold is never reached.
- For long sequences, the O(S) similarity recomputation per step can be expensive on CPU.
  Pre-compute the similarity vector once and update only the affected indices after each merge.

---

### 3.5 VLCachePress

**File:** `press/methods/vlcache.py`
**Requires:** `output_attentions=True`
**Core idea:** Vision-language specific. Per-layer adaptive budget based on how much each
layer attends to image tokens, plus a dampening factor that deprioritises image tokens in
the importance ranking.

```python
@dataclass
class VLCachePress(BasePress):
    base_budget:       int   = 128
    dampening_factor:  float = 0.5   # score multiplier for image tokens (< 1.0)
    gamma:             float = 1.0   # budget scaling sensitivity to image attention
    image_positions:   set   = field(default_factory=set)  # token positions of image patch tokens
    head_agg:          str   = "max"

    def __post_init__(self):
        super().__post_init__()
        self._scores: dict[int, torch.Tensor] = {}

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        assert attentions is not None
        layer_idx = module.layer_idx
        seq_len   = keys.shape[2]

        scores = _importance_from_attn(attentions, module)
        if self.head_agg == "max":
            scores = scores.max(dim=1).values
        else:
            scores = scores.mean(dim=1)
        scores = scores.mean(dim=0)  # [kv]

        # Apply modality dampening to image token scores
        if self.image_positions:
            img_idx = torch.tensor(
                [p for p in self.image_positions if p < seq_len],
                dtype=torch.long, device=scores.device
            )
            scores[img_idx] *= self.dampening_factor

        # Accumulate
        if layer_idx not in self._scores:
            self._scores[layer_idx] = scores
        else:
            prev = self._scores[layer_idx]
            self._scores[layer_idx] = prev + scores[:prev.shape[-1]]

        # Per-layer adaptive budget
        if self.image_positions:
            img_attn = attentions[..., list(self.image_positions)].mean().item()
        else:
            img_attn = 0.0
        layer_budget = int(self.base_budget * (1 + self.gamma * img_attn))

        if seq_len <= layer_budget:
            return keys, values

        acc = self._scores[layer_idx]
        idx = acc.topk(layer_budget).indices.sort().values
        self._scores[layer_idx] = acc[idx]

        return keys[:, :, idx, :], values[:, :, idx, :]

    def reset(self) -> None:
        self._scores.clear()
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `base_budget` | required | Global budget floor |
| `dampening_factor` | `0.5` | Score multiplier for image tokens |
| `gamma` | `1.0` | Controls budget expansion for image-heavy layers |
| `image_positions` | `set()` | Set of token positions corresponding to image patches |

#### Setup

Pass `image_positions` to the constructor before the context manager:

```python
image_positions = set(range(img_start_idx, img_end_idx))
press = VLCachePress(base_budget=128, image_positions=image_positions)
with press(model):
    output = model.generate(...)
```

---

### 3.6 KIVIPress

**File:** `press/methods/kivi.py`
**Requires:** No extra hooks
**Core idea:** Asymmetric quantization axes matching the outlier structure of each tensor.
Keys have per-channel outliers → quantize per-channel (over the sequence dimension).
Values have per-token outliers → quantize per-token (over the feature dimension).

> **Note:** HuggingFace ships `QuantizedCacheConfig` with `backend="quanto"` which
> supports 4-bit and 8-bit KV quantization. Consider using it as the baseline for the
> evaluation comparison column. Implement `KIVIPress` for 2-bit and for the per-axis
> asymmetry not supported by `QuantizedCacheConfig`.

```python
@dataclass
class KIVIPress(BasePress):
    bits:             int = 8    # storage bit-width; paper target is 2
    residual_length:  int = 32   # last N tokens kept in fp16 (too new to quantize safely)
    q_group_size:     int = 64   # group size along head_dim for scale computation

    def __post_init__(self):
        super().__post_init__()
        self._stored: dict[int, dict] = {}

    def _quantize_per_channel(self, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Keys: per-channel quantization.
        Channels = head_dim (last axis). Scale computed over the sequence axis.
        Groups of q_group_size along head_dim share a scale.

        K shape: [B, kv_heads, seq, head_dim]
        """
        B, H, S, D = K.shape
        K_groups = K.reshape(B, H, S, D // self.q_group_size, self.q_group_size)
        scale = K_groups.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # [B, H, 1, groups, g]
        K_norm = (K_groups / scale).clamp(-1, 1)
        max_val = 2 ** (self.bits - 1) - 1
        K_q = (K_norm * max_val).round().to(torch.int8)
        return K_q.reshape(B, H, S, D), scale.reshape(B, H, 1, D)

    def _quantize_per_token(self, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Values: per-token quantization.
        Scale computed over head_dim (last axis).

        V shape: [B, kv_heads, seq, head_dim]
        """
        scale = V.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, H, S, 1]
        V_norm = (V / scale).clamp(-1, 1)
        max_val = 2 ** (self.bits - 1) - 1
        V_q = (V_norm * max_val).round().to(torch.int8)
        return V_q, scale

    def _dequantize(self, X_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        max_val = 2 ** (self.bits - 1) - 1
        return (X_q.float() / max_val) * scale

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer_idx = module.layer_idx
        seq_len   = keys.shape[2]

        # Split into bulk (to quantize) and residual (keep in fp16)
        if seq_len <= self.residual_length:
            return keys, values

        bulk_k = keys[:, :, :-self.residual_length, :]
        bulk_v = values[:, :, :-self.residual_length, :]
        res_k  = keys[:, :, -self.residual_length:, :]
        res_v  = values[:, :, -self.residual_length:, :]

        K_q, k_scale = self._quantize_per_channel(bulk_k)
        V_q, v_scale = self._quantize_per_token(bulk_v)

        self._stored[layer_idx] = {
            "K_q": K_q, "k_scale": k_scale,
            "V_q": V_q, "v_scale": v_scale,
        }

        bulk_k_deq = self._dequantize(K_q, k_scale).to(keys.dtype)
        bulk_v_deq = self._dequantize(V_q, v_scale).to(values.dtype)

        keys_out   = torch.cat([bulk_k_deq, res_k],   dim=2)
        values_out = torch.cat([bulk_v_deq, res_v],   dim=2)

        return keys_out, values_out
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `bits` | `8` | Storage bit-width; 2-bit is the paper's target |
| `residual_length` | `32` | Last N tokens kept in fp16 |
| `q_group_size` | `64` | Head-dim group size for scale computation |

---

### 3.7 KVQuantPress

**File:** `press/methods/kvquant.py`
**Requires:** Opt-in `k_proj` pre-RoPE hook; calibration pass
**Core idea:** Three improvements over KIVI: quantize keys before RoPE is applied
(outlier structure is cleaner pre-rotation), fit non-uniform quantization bins via
Lloyd-Max to each layer's empirical distribution, and isolate outlier channels into a
sparse fp16 component.

#### Pre-RoPE key hook

`KVQuantPress` sets `self._needs_pre_rope_keys = True` in `__post_init__`. `BasePress.__call__`
detects this flag and registers an additional hook on `k_proj`:

```python
def _pre_rope_key_hook(self, module, input, output):
    """
    Fires after k_proj.forward(), before RoPE is applied in self_attn.forward().
    Stores the raw key projection for quantization.
    """
    # output shape: [B, seq, num_kv_heads * head_dim]
    self._pre_rope_keys[self._current_layer_idx] = output.detach()
```

`self._current_layer_idx` must be set at the start of `forward_hook` before `compress()`
is called. Add this to `BasePress.forward_hook`:

```python
self._current_layer_idx = module.layer_idx
```

#### Calibration

```python
@torch.inference_mode()
def calibrate(
    self,
    model: PreTrainedModel,
    tokenizer,
    calibration_texts: list[str],
    n_samples: int = 128,
) -> None:
    """
    Run n_samples forward passes to collect pre-RoPE key and value
    distributions per layer. Fit Lloyd-Max non-uniform quantization bins.
    Stores:
        self._key_bins[layer_idx]:   list of bin edges
        self._value_bins[layer_idx]: list of bin edges
    """
    key_samples   = defaultdict(list)
    value_samples = defaultdict(list)

    with self(model):   # registers the pre-RoPE hook
        for text in calibration_texts[:n_samples]:
            inputs = tokenizer(text, return_tensors="pt")
            model(**inputs)
            for layer_idx, k in self._pre_rope_keys.items():
                key_samples[layer_idx].append(k.cpu())
            for layer_idx, (k, v) in self._stored.items():
                value_samples[layer_idx].append(v.cpu())

    for layer_idx in key_samples:
        all_k = torch.cat(key_samples[layer_idx], dim=0).flatten()
        self._key_bins[layer_idx] = _lloyd_max(all_k, n_bins=2**self.bits)

    for layer_idx in value_samples:
        all_v = torch.cat(value_samples[layer_idx], dim=0).flatten()
        self._value_bins[layer_idx] = _lloyd_max(all_v, n_bins=2**self.bits)
```

#### Compression algorithm

```
Pre-RoPE keys (from _pre_rope_keys[layer_idx]):
  1. Identify outlier channels: abs(K) > outlier_threshold_pct percentile
  2. K_sparse = zeros_like(K); K_sparse[outlier_mask] = K[outlier_mask]
  3. K_dense  = K.clone(); K_dense[outlier_mask] = 0.0
  4. Quantize K_dense with non-uniform Lloyd-Max bins → K_dense_q
  5. Return dequant(K_dense_q) + K_sparse (full-precision outliers add back in)

Values (post-RoPE, same as in compress() kwargs):
  6. Quantize V per-token with non-uniform bins → V_q
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `bits` | `4` | Target bit-width |
| `outlier_threshold_pct` | `99` | Percentile above which a channel value is an outlier |
| `sparse_ratio` | `0.01` | Fraction of entries kept in fp16 sparse component |
| `n_calibration_samples` | `128` | Samples for Lloyd-Max fitting |

#### Implementation order

Implement and validate `KIVIPress` first. `KVQuantPress` is a direct extension: Lloyd-Max
bins replace the linear scale, pre-RoPE keys replace post-RoPE keys, and the dense+sparse
decomposition wraps the quantization step.

---

### 3.8 TurboQuantPress

**File:** `press/methods/turboquant.py`
**Requires:** No extra hooks; no calibration data
**Core idea:** A random orthogonal rotation spreads information uniformly across all
coordinates, enabling near-optimal per-coordinate scalar quantization without calibration.
A 1-bit JL sketch on residuals corrects inner-product bias.

#### Rotation initialisation (once per model)

```python
def post_init_from_model(self, model: PreTrainedModel) -> None:
    # Determine head_dim from the model config
    self.head_dim = model.config.hidden_size // model.config.num_attention_heads
    torch.manual_seed(self.seed)
    G = torch.randn(self.head_dim, self.head_dim)
    self._R, _ = torch.linalg.qr(G)          # random orthogonal [D, D]
    S = (torch.randint(0, 2, (self.sketch_dim, self.head_dim)) * 2 - 1).float()
    self._S = S / (self.sketch_dim ** 0.5)   # JL matrix [sketch_dim, D]
```

#### Compression algorithm

```python
def compress(self, module, hidden_states, keys, values, attentions, kwargs):
    layer_idx = module.layer_idx

    R = self._R.to(keys.device)

    # Step 1: Rotate into the isotropic basis
    K_rot = keys   @ R.T    # [B, kv_heads, seq, D]
    V_rot = values @ R.T

    # Step 2: Per-coordinate scalar quantization
    max_val = 2 ** (self.bits - 1) - 1
    k_scale = K_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    K_q     = (K_rot / k_scale * max_val).round().clamp(-max_val, max_val).to(torch.int8)

    v_scale = V_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    V_q     = (V_rot / v_scale * max_val).round().clamp(-max_val, max_val).to(torch.int8)

    # Step 3: JL residual sketch
    K_deq     = K_q.float() / max_val * k_scale
    residual  = K_rot - K_deq
    S         = self._S.to(keys.device)
    K_sketch  = (residual @ S.T).sign()           # [B, kv_heads, seq, sketch_dim]

    self._stored[layer_idx] = {
        "K_q": K_q, "k_scale": k_scale, "K_sketch": K_sketch,
        "V_q": V_q, "v_scale": v_scale,
    }

    # Un-rotate before writing back so standard attention still works
    K_deq_out = (K_deq @ R).to(keys.dtype)
    V_deq_out = (V_rot.float() / max_val * v_scale @ R).to(values.dtype)

    return K_deq_out, V_deq_out
```

> The JL sketch corrects inner-product bias but full application requires adding
> `Q_rot @ S.T @ K_sketch.T` to the attention logits, which means patching
> `self_attn.forward()`. Implement without residual correction first and measure PPL
> degradation. Add the correction patch if degradation exceeds the exit criterion.

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `bits` | `8` | Scalar quantization bit-width |
| `sketch_dim` | `64` | JL sketch dimension |
| `seed` | `42` | Random seed for the rotation matrix |

---

### 3.9 AttnMatchPress — Attention Matching

**File:** `press/methods/attn_match.py`
**Requires:** `output_attentions=True`; calibration pass for reference queries
**Core idea:** Solve a small optimisation problem to find compressed KV pairs
`(Ck, Cv)` such that the local attention output and attention mass are matched as closely
as possible. A key-scaling approximation stands in for the full β injection.
**Implement last.** Most complex method; requires calibration and a per-decode optimisation
loop, which is expensive on CPU.

#### Calibration

```python
def calibrate(self, model, tokenizer, prompts, n_samples=32):
    """Collect reference query vectors per layer for the optimisation."""
    with self(model):
        for prompt in prompts[:n_samples]:
            inputs = tokenizer(prompt, return_tensors="pt")
            model(**inputs, output_attentions=True)
    # self._Q_ref[layer_idx] is populated by forward_hook
```

#### Compression algorithm (approximate)

```python
def compress(self, module, hidden_states, keys, values, attentions, kwargs):
    layer_idx = module.layer_idx
    seq_len   = keys.shape[2]

    if seq_len <= self.budget:
        return keys, values

    Q_ref = self._Q_ref.get(layer_idx)   # [B, H, q, D]
    A_out = (attentions @ values)        # reference attention output

    # Initialise compressed KV as topk by attention score
    scores = _importance_from_attn(attentions, module).mean(dim=(0, 1))
    idx    = scores.topk(self.budget).indices.sort().values
    Ck = keys[:, :, idx, :].clone().requires_grad_(True)
    Cv = values[:, :, idx, :].clone().requires_grad_(True)

    opt = torch.optim.Adam([Ck, Cv], lr=1e-3)
    d   = Ck.shape[-1]

    for _ in range(self.n_optim_steps):
        logits = (Q_ref @ Ck.transpose(-1, -2)) / (d ** 0.5)
        A_hat  = torch.softmax(logits, dim=-1) @ Cv
        loss   = (A_hat - A_out.detach()).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # Key scaling as β approximation for reduced attention mass
    scale_factor = (seq_len / self.budget) ** 0.5
    Ck_out = Ck.detach() * scale_factor
    Cv_out = Cv.detach()

    return Ck_out, Cv_out
```

#### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `budget` | required | |
| `lambda_mass` | `0.1` | Weight of mass-matching regularisation (extend if needed) |
| `n_calibration_samples` | `32` | Prompts for Q_ref collection |
| `n_optim_steps` | `20` | Gradient steps per decode |

---

## 4. Evaluation Protocol

### 4.1 Metrics

| Metric | Measures | How to compute |
|---|---|---|
| PPL (WikiText-2) | Quality degradation | `evaluate.load("perplexity")` on 2048-token chunks |
| Compression ratio | Length reduction | `compressed_len / original_len` per layer, averaged |
| Bits per KV entry | Memory reduction (quant methods) | `bits * 2 / 32` (keys + values relative to fp16) |
| Peak memory (MB) | RAM footprint | `tracemalloc` peak or `torch.profiler` with `profile_memory=True` |
| Decode latency (ms/tok) | Speed | Median of `n_reps` timed decode steps; CPU only |

> **On latency and `output_attentions=True`:** enabling this flag forces eager attention
> mode, materialises the full `[B, H, S, S]` weight tensor on every decode step, and
> roughly doubles CPU latency for eviction methods. Latency numbers for H2O, SnapKV,
> KeepKV, TokenMerge, and VLCache must be reported in two rows: one with attention
> weight extraction enabled (research mode) and one without (production mode where
> eviction is prefill-only). Do not mix the two in the comparison table.
>
> **On CPU memory:** `torch.cuda.memory_allocated()` is meaningless for CPU runs.
> Use `tracemalloc.get_traced_memory()` or `psutil.Process().memory_info().rss` as a
> coarse indicator. Report the delta from baseline, not the absolute RSS.

### 4.2 Benchmark harness

**File:** `eval/harness.py`

```python
@torch.inference_mode()
def benchmark(
    model,
    tokenizer,
    press_factory,              # callable → BasePress instance
    dataset: str = "wikitext-2",
    n_ppl_samples: int = 100,
    n_latency_reps: int = 5,
    n_warmup: int = 2,
    max_gen_len: int = 128,
) -> dict:
    """
    press_factory is called once per sample so each sample gets a fresh press
    instance with clean state (no accumulated scores carried across sequences).
    """
    press = press_factory()
    return {
        "ppl":          run_ppl(model, tokenizer, press_factory, n_ppl_samples),
        "latency_ms":   run_latency(model, tokenizer, press_factory,
                                    n_latency_reps, n_warmup, max_gen_len),
        "compression":  press.compression_stats if hasattr(press, "compression_stats") else {},
    }
```

### 4.3 Baseline

Always record a baseline (no compression) before benchmarking any press. All metrics
are reported as absolute values and as deltas from baseline.

```python
# Baseline: standard DynamicCache(config=model.config), no press
baseline = benchmark(model, tokenizer, press_factory=lambda: NoOpPress())
```

`NoOpPress` is a `BasePress` subclass whose `compress()` returns `(keys, values)` unchanged.

### 4.4 Comparison table format

| Method | PPL (↓) | ΔPPL | Comp. ratio (↓) | Bits/KV (↓) | Latency ms/tok (↓) |
|---|---|---|---|---|---|
| Baseline | X.XX | — | 1.00 | 32 | X.X |
| H2OPress (budget=128) | | | | 32 | |
| SnapKVPress (budget=128) | | | | 32 | |
| KeepKVPress (budget=128) | | | | 32 | |
| TokenMergePress | | | | 32 | |
| KIVIPress (8-bit) | | | 1.00 | 8 | |
| KIVIPress (2-bit) | | | 1.00 | 2 | |
| KVQuantPress (4-bit) | | | 1.00 | 4 | |
| TurboQuantPress (8-bit) | | | 1.00 | 8 | |
| H2OPress + KIVIPress | | | | 8 | |
| SnapKVPress + KIVIPress | | | | 8 | |

---

## 5. Build Phases

### Phase 1 — Core infrastructure ✅ Build first

**Goal:** End-to-end loop producing output identical to a plain `DynamicCache` run.

| Task | File |
|---|---|
| Implement `BasePress` with no-op `compress()` | `press/base.py` |
| Implement `extract_keys_and_values` (DynamicCache + QuantizedCache) | `utils.py` |
| Implement `_importance_from_attn` with GQA support | `utils.py` |
| Implement `NoOpPress` | `press/methods/noop.py` |
| Verify baseline PPL matches plain `DynamicCache` exactly | `tests/test_baseline.py` |
| Implement benchmark harness skeleton | `eval/harness.py` |

**Exit criterion:** `NoOpPress` produces logit-identical output to a plain
`DynamicCache(config=model.config)` run on a 5-token input.

---

### Phase 2 — Eviction methods

**Goal:** Validate the hook → `compress()` → `_write_to_cache()` pipeline.

**Order:** H2O → SnapKV → KeepKV → TokenMerge → VLCache (last; needs multimodal model)

| Task | File |
|---|---|
| `H2OPress` with GQA-aware head aggregation | `press/methods/h2o.py` |
| `SnapKVPress` with prefill-fixed selection | `press/methods/snapkv.py` |
| `KeepKVPress` with EMA scoring | `press/methods/keepkv.py` |
| `TokenMergePress` with cosine similarity + budget guard | `press/methods/token_merge.py` |
| `VLCachePress` with per-layer adaptive budget | `press/methods/vlcache.py` |
| PPL + latency benchmarks, both with and without `output_attentions=True` | `eval/run_eviction.py` |

**Exit criterion:** `H2OPress(budget=128)` on a 512-token input shows < 0.5 PPL
increase vs baseline.

---

### Phase 3 — Quantization methods

**Goal:** Implement and validate the three quantization approaches.

**Order:** KIVI → TurboQuant → KVQuant (last; needs pre-RoPE hook)

| Task | File |
|---|---|
| `KIVIPress`: per-channel keys, per-token values, group quantization | `press/methods/kivi.py` |
| KIVI 8-bit smoke test: PPL delta < 0.1 | `tests/test_kivi.py` |
| `TurboQuantPress`: rotation init + scalar quant + JL sketch | `press/methods/turboquant.py` |
| TurboQuant without residual correction (baseline); measure PPL | |
| `KVQuantPress`: calibration + Lloyd-Max bins | `press/methods/kvquant.py` |
| `KVQuantPress`: pre-RoPE hook via `_needs_pre_rope_keys` flag | `press/base.py` |
| `KVQuantPress`: dense+sparse decomposition | |
| PPL + bits/KV benchmarks for all quant methods | `eval/run_quant.py` |

**Exit criterion:** `KIVIPress(bits=8)` shows < 0.1 PPL increase vs baseline.
`TurboQuantPress` within 0.3 PPL of baseline.

---

### Phase 4 — Composition

**Goal:** Stack eviction + quantization and verify correct interaction.

| Task | File |
|---|---|
| Implement `ComposedPress` | `press/composed.py` |
| Test `ComposedPress(H2OPress, KIVIPress)` correctness | `tests/test_composed.py` |
| Verify evict-then-quantize order produces lower PPL than quantize-then-evict | `tests/test_composed.py` |
| Run full comparison table (all methods + compositions) | `eval/run_all.py` |

---

### Phase 5 — Multi-turn and state reset

**Goal:** Ensure scores, EMA, and fixed selections reset correctly between sequences
and persist correctly within a multi-turn conversation where intended.

| Task | File |
|---|---|
| Add `reset()` to all stateful presses | all method files |
| Test PPL does not drift across 5 independent sequences | `tests/test_reset.py` |
| Test multi-turn: KeepKV EMA persists correctly within a conversation | `tests/test_multiturn.py` |
| Document multi-turn policy (reset vs carry-over) per method | `README.md` |

**Multi-turn policy per method:**

| Method | Between turns | Rationale |
|---|---|---|
| H2OPress | Reset | Scores from prior turn distort new context |
| SnapKVPress | Reset (clear `_keep_idx`) | Selection is prefill-specific |
| KeepKVPress | Reset or carry-over | Carry-over is valid for same-topic conversations; expose `reset_between_turns: bool` param |
| TokenMergePress | N/A (stateless) | — |
| VLCachePress | Reset | Image positions change between turns |
| KIVIPress | N/A (stateless) | — |
| KVQuantPress | Retain bins (they're model-level) | Calibration is per-model, not per-sequence |
| TurboQuantPress | Retain R, S (model-level) | Rotation is fixed per model |

---

### Phase 6 — llama.cpp deployment validation

**Goal:** Validate the 2–3 best methods from Phase 4 under real x86 latency conditions.

| Task | Notes |
|---|---|
| Convert model to GGUF | `llama.cpp/convert.py` |
| Baseline: `--cache-type-k fp16` | Fastest on x86 |
| Eviction method equivalent in llama.cpp C++ | Method with best PPL/ratio tradeoff |
| KIVI equivalent: `--cache-type-k q8_0` is close; custom for 2-bit | `src/llama-kv-cache.cpp` |
| Record real tokens/sec on target x86 hardware | |

---

### Phase 7 — AttnMatchPress (optional, implement last)

**Goal:** Implement the most complex method once the framework is stable.

| Task | File |
|---|---|
| Calibration pass for Q_ref collection | `press/methods/attn_match.py` |
| Inner optimisation loop (20 steps per compress call) | |
| Key scaling as β approximation | |
| PPL benchmark vs other methods | `eval/run_all.py` |

---

## 6. File Layout

```
kv_compress/
│
├── press/
│   ├── base.py                # BasePress
│   ├── composed.py            # ComposedPress (eviction + quant stacking)
│   └── methods/
│       ├── noop.py            # NoOpPress (baseline)
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
├── utils.py                   # extract_keys_and_values, _importance_from_attn
│
├── eval/
│   ├── harness.py             # benchmark() entrypoint
│   ├── run_eviction.py        # Phase 2 benchmark script
│   ├── run_quant.py           # Phase 3 benchmark script
│   └── run_all.py             # Phase 4 full comparison table
│
├── tests/
│   ├── test_baseline.py       # NoOpPress == plain DynamicCache
│   ├── test_kivi.py           # KIVIPress 8-bit PPL smoke test
│   ├── test_composed.py       # ComposedPress correctness + order test
│   ├── test_reset.py          # Multi-sequence state isolation
│   ├── test_multiturn.py      # Multi-turn state carry-over
│   └── test_hooks.py          # Hook count before/after context manager
│
├── scripts/
│   └── calibrate_kvquant.py   # Offline calibration for KVQuantPress
│
└── README.md
```

---

## 7. Testing Checklist

### Per-method tests (run for each new method before benchmarking)

- [ ] **Shape test:** output `K`, `V` have same `batch`, `num_kv_heads`, `head_dim` as
  input. Seq dimension ≤ input seq dimension.
- [ ] **Dtype test:** output dtype matches input dtype. No silent fp32 upcasts in the
  return value.
- [ ] **Budget test:** for eviction methods, `K.shape[2] <= budget` after compression.
- [ ] **GQA test:** run on a model with `num_kv_heads != num_heads` (e.g. Llama 3.2);
  verify shapes and importance scores are correct.
- [ ] **Dequant roundtrip (quant methods):** `(K_deq - K).abs().max() < 0.05` at 8-bit.
- [ ] **32-bit logit equivalence (quant methods):** `compress()` at 32-bit should
  reproduce baseline logits exactly (no information loss at full precision).
- [ ] **Hook cleanup:** after context manager `__exit__`, verify
  `len(list(module.parameters())) == len(list(module._forward_hooks))` is zero
  for all attention layers.
- [ ] **Stat recording:** `compression_stats` (if implemented) is populated after one
  forward pass.
- [ ] **Multi-sequence reset:** call `reset()` between two independent sequences; verify
  scores/EMA from sequence 1 do not affect sequence 2's compression.

### Integration tests

- [ ] `NoOpPress` produces logit-identical output to plain `DynamicCache(config=model.config)`.
- [ ] `ComposedPress(H2OPress, KIVIPress)` output is within tolerance of running
  `H2OPress` alone followed by `KIVIPress` alone on the same input.
- [ ] Context manager leaves zero hooks on all attention layers after `__exit__`,
  even when an exception is raised inside the `with` block.
- [ ] `QuantizedCache` path: `_write_to_cache` correctly re-quantizes after compression
  (verify `cumulative_length` matches `keys.shape[2]`).

---

## 8. Known Constraints & Gotchas

| Issue | Detail | Mitigation |
|---|---|---|
| `output_attentions=True` performance cost | Forces eager attention mode; materialises full `[B, H, S, S]` tensor every decode step; roughly doubles latency on CPU. Affects all eviction methods. | Report latency separately for prefill-only (no `output_attentions`) vs full-decode compression modes. For production, prefer SnapKV (prefill-only selection) to avoid per-step cost. |
| GQA/MQA head mismatch | `attn_weights.shape[1] = num_heads`; `keys.shape[1] = num_kv_heads`. Direct indexing without grouping produces wrong importance scores. | Always use `_importance_from_attn()`. Never `attn_weights.mean(dim=1)` directly. |
| `cache_position` management after eviction | After evicting tokens, positions stored in the cache are non-contiguous. RoPE frequencies were computed for the original positions; re-indexing is not needed because positions are embedded at write time, but `cache_position` passed to the next decode step must still be contiguous (HF manages this in `generate()`). | Do not modify `cache_position` inside `compress()`. Position indices are handled by `generate()`. |
| `rotary_emb` mutation | Assigning `layer.self_attn.rotary_emb = lm.rotary_emb` mutates the model in-place and is not undone on exit. | Only assign when `layer.self_attn.rotary_emb is None` (already in spec). Consider saving and restoring the original value in `finally`. |
| `torch.compile` incompatible with DynamicCache | Dynamic shapes in `DynamicCache` cause recompilations (verified in HF issue #37908). | Do not compile the model when using any press. Use `torch.set_num_threads()` for CPU throughput instead. |
| Hook removal on exception | If `compress()` raises inside the context manager, `finally` still runs and hooks are removed. | Already handled by `try/finally` in `__call__`. Add a test that verifies hook count after a deliberate exception. |
| Calibration data distribution | `KVQuantPress` Lloyd-Max bins fitted on one domain (e.g. WikiText-2) may not be optimal for another (e.g. code, conversation). | Document calibration dataset in `scripts/calibrate_kvquant.py`. Re-run if switching task domains. |
| CPU latency is a relative indicator only | Absolute CPU numbers are not publication-quality. BLAS threading, turbo frequency, and Python GIL all introduce noise. | Use llama.cpp for latency claims. Use HF benchmarks for PPL and compression ratio. Report CPU latency as a relative speedup vs baseline, not as absolute ms/token. |
| TurboQuant residual correction | Full JL correction requires `Q_rot @ S.T @ K_sketch.T` added to attention logits, which means patching `self_attn.forward()`. | Implement without correction first. If PPL degradation > 0.3 vs baseline at 8-bit, add the `self_attn` patch. |
| SUPPORTED_MODELS list | `BasePress.__call__` logs a warning for untested models but does not block. The `rotary_emb` propagation and `cache.layers[layer_idx]` access are Llama/Qwen2-specific. | Before adding a new model family, verify `cache.layers` attribute layout and `rotary_emb` propagation pattern. Add to `SUPPORTED_MODELS` after validation. |

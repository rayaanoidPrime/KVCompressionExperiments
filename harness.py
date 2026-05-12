import logging
from pathlib import Path
import time

from matplotlib import pyplot as plt
import pandas as pd
from transformers import DynamicCache
from utils import tokenize
import torch


def measure_perplexity(
    model, tokenizer, context: str, press=None, max_length: int = 512
) -> dict:
    prefix_len = max_length // 2
    cont_len   = max_length // 2

    full_ids = tokenize(tokenizer, context, max_length=max_length)
    full_ids = full_ids[0, :max_length]   # (T,)
    n_full   = full_ids.shape[0]

    # Guard: need at least 2 tokens to have a prefix and a continuation
    if n_full < 2:
        return {
            "perplexity": float("nan"),
            "loss": float("nan"),
            "n_prefix_tokens": 0,
            "n_continuation_tokens": 0,
            "inference_ms": 0.0,
            "avg_kv_entries": 0,
        }

    actual_prefix = min(prefix_len, n_full - 1)   # leave at least 1 token for continuation
    prefix_ids = full_ids[:actual_prefix].unsqueeze(0)
    n_prefix   = prefix_ids.shape[1]

    cont_start = n_prefix
    cont_end   = min(cont_start + cont_len, n_full)
    cont_ids   = full_ids[cont_start:cont_end].unsqueeze(0)
    n_cont     = cont_ids.shape[1]

    # Guard: skip if continuation is empty
    if n_cont == 0:
        return {
            "perplexity": float("nan"),
            "loss": float("nan"),
            "n_prefix_tokens": n_prefix,
            "n_continuation_tokens": 0,
            "inference_ms": 0.0,
            "avg_kv_entries": 0,
        }

    t0 = time.perf_counter()

    with torch.no_grad():
        cache = DynamicCache()
        if press is not None:
            with press(model):
                outputs = model(
                    prefix_ids,
                    past_key_values  = cache,
                    cache_position   = torch.arange(n_prefix, device="cpu"),
                    use_cache        = True,
                )
        else:
            outputs = model(
                prefix_ids,
                past_key_values = cache,
                cache_position  = torch.arange(n_prefix, device="cpu"),
                use_cache       = True,
            )

        cont_cache_pos = torch.arange(n_prefix, n_prefix + n_cont, device="cpu")
        outputs = model(
            cont_ids,
            past_key_values = cache,
            cache_position  = cont_cache_pos,
            use_cache       = True,
            labels          = cont_ids,
        )
        loss = outputs.loss.item()

    dt = (time.perf_counter() - t0) * 1000.0

    avg_kv  = 0
    n_layers = 0
    for layer_cache in cache.layers:
        if hasattr(layer_cache, "keys") and isinstance(layer_cache.keys, torch.Tensor):
            avg_kv   += layer_cache.keys.shape[2]
            n_layers += 1
    avg_kv = avg_kv / max(n_layers, 1) if n_layers > 0 else 0

    import math
    ppl = math.exp(loss)

    return {
        "perplexity":              ppl,
        "loss":                    loss,
        "n_prefix_tokens":         n_prefix,
        "n_continuation_tokens":   n_cont,
        "inference_ms":            dt,
        "avg_kv_entries":          avg_kv,
    }


def measure_decoding_latency(
    model,
    tokenizer,
    context: str,
    seq_lengths: list[int],
    output_dir: Path,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    For each seq_len in seq_lengths:
      - Tokenise `context` to `seq_len` tokens as the prompt
      - Autoregressively decode one token at a time until `seq_len`
        additional tokens have been generated
      - Record per-step latency and KV cache size

    Saves one CSV and one PNG to `output_dir`.
    """
    log = logging.getLogger("decoding_latency")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    model = model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seq_len in seq_lengths:
        log.info("  seq_len=%d", seq_len)

        # ── Tokenise prompt ────────────────────────────────────────────────
        from utils import tokenize
        input_ids = tokenize(tokenizer, context, max_length=seq_len).to(device)
        n_prompt  = input_ids.shape[1]

        if n_prompt == 0:
            log.warning("Empty prompt for seq_len=%d - skipping.", seq_len)
            continue

        # ── Prefill ────────────────────────────────────────────────────────
        with torch.no_grad():
            cache = DynamicCache()
            outputs = model(
                input_ids,
                past_key_values = cache,
                cache_position  = torch.arange(n_prompt, device=device),
                use_cache       = True,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        n_cached   = n_prompt

        # ── Autoregressive decode — one token at a time ────────────────────
        for step in range(seq_len):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                outputs = model(
                    next_token,
                    past_key_values = cache,
                    cache_position  = torch.tensor([n_cached], device=device),
                    use_cache       = True,
                )

            if device == "cuda":
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            n_cached  += 1

            # KV cache size: sum key tensors across all layers
            kv_entries = sum(
                layer_kv[0].shape[2]          # (batch, heads, seq, head_dim)
                for layer_kv in cache.key_cache
                if layer_kv is not None
            )

            rows.append({
                "seq_len":    seq_len,
                "decode_step": step + 1,
                "kv_entries": kv_entries,
                "latency_ms": step_ms,
            })

    df = pd.DataFrame(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / f"decoding_latency_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV → %s", csv_path)

    return df

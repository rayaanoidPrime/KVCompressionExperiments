import contextlib
import statistics
import time

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

@torch.inference_mode()
def measure_latency(
    model,
    press:None,
    prompt_tokens: int,
    n_gen: int,
    n_reps: int,
    n_warmup: int,
    device: str = "cpu",
) -> dict:
    """
    Measure prefill, TTFT, and per-token decode latency.

    Args:
        model:         HuggingFace model (already on `device`)
        prompt_tokens: Number of prompt tokens to prefill
        n_gen:         Number of new tokens to generate after prefill
        n_reps:        Number of timed repetitions (mean/stddev reported)
        n_warmup:      Number of warmup passes (untimed)
        device:        Target device (currently CPU only)

    Returns:
        dict with keys:
            prefill_ms_mean, prefill_ms_std,
            ttft_ms_mean,    ttft_ms_std,
            decode_ms_mean,  decode_ms_std,   # per token
            total_ms_mean,   total_ms_std
    """

    assert device == "cpu", "Only CPU timing is supported."
    model.eval()
    model.to(device)

    # --- Build a fixed dummy prompt (shape: [1, prompt_tokens]) ---
    dummy_input = torch.zeros(1, prompt_tokens, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Helper: run one full prefill + n_gen decode steps, return timings
    # ------------------------------------------------------------------
    def _run_once(warmup:bool = False) -> tuple[float, float, float, float]:
        """Returns (prefill_ns, ttft_ns, decode_per_tok_ns, total_ns)."""

        ctx = press(model) if press is not None else contextlib.nullcontext()
        with ctx:

            cache = DynamicCache()  # fresh cache for this rep
            # ── 1. PREFILL ──────────────────────────────────────────
            cache_position = torch.arange(prompt_tokens, device=device)

            t0 = time.perf_counter_ns()
            out = model(
                input_ids=dummy_input,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
            t1 = time.perf_counter_ns()

            prefill_ns = t1 - t0
            past = out.past_key_values

            # ── 2. FIRST DECODE STEP (needed for TTFT) ──────────────
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache_position = torch.tensor([prompt_tokens], device=device)

            t2 = time.perf_counter_ns()
            out = model(
                input_ids=next_token,
                cache_position=cache_position,
                past_key_values=past,
                use_cache=True,
            )
            t3 = time.perf_counter_ns()

            first_decode_ns = t3 - t2
            ttft_ns = prefill_ns + first_decode_ns
            past = out.past_key_values

            if warmup:
                # If this is just a warmup pass, skip the remaining decode steps
                return prefill_ns, ttft_ns, 0.0, ttft_ns

            # ── 3. REMAINING DECODE STEPS ───────────────────────────
            decode_times = [first_decode_ns]

            for step in range(1, n_gen):
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache_position = torch.tensor([prompt_tokens + step], device=device)

                t4 = time.perf_counter_ns()
                out = model(
                    input_ids=next_token,
                    cache_position=cache_position,
                    past_key_values=past,
                    use_cache=True,
                )
                t5 = time.perf_counter_ns()

                decode_times.append(t5 - t4)
                past = out.past_key_values

            decode_per_tok_ns = statistics.mean(decode_times)
            total_ns = prefill_ns + sum(decode_times)

        return prefill_ns, ttft_ns, decode_per_tok_ns, total_ns

    # ------------------------------------------------------------------
    # Warmup passes (cache object is NOT the same across reps anyway,
    # but this primes the CPU caches / JIT / etc.)
    # ------------------------------------------------------------------
    for _ in range(n_warmup):
        _run_once(warmup=True)

    # ------------------------------------------------------------------
    # Timed repetitions
    # ------------------------------------------------------------------
    prefill_list, ttft_list, decode_list, total_list = [], [], [], []

    for _ in range(n_reps):
        p, tt, d, tot = _run_once()
        prefill_list.append(p / 1e6)   # ns → ms
        ttft_list.append(tt / 1e6)
        decode_list.append(d / 1e6)
        total_list.append(tot / 1e6)

    def _stats(lst):
        mean = statistics.mean(lst)
        std  = statistics.stdev(lst) if len(lst) > 1 else 0.0
        return mean, std

    pm, ps   = _stats(prefill_list)
    tm, ts   = _stats(ttft_list)
    dm, ds   = _stats(decode_list)
    totm, tots = _stats(total_list)

    return {
        "prefill_ms_mean":  pm,   "prefill_ms_std":  ps,
        "ttft_ms_mean":     tm,   "ttft_ms_std":     ts,
        "decode_ms_mean":   dm,   "decode_ms_std":   ds,   # per token
        "total_ms_mean":    totm, "total_ms_std":    tots,
    }
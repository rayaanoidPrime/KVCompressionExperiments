from time import time

from transformers import DynamicCache
from utils import tokenize
import torch


def measure_perplexity(
    model, tokenizer, context: str, press=None, max_length: int = 512
) -> dict:
    prefix_len = max_length // 2
    cont_len = max_length // 2

    full_ids = tokenize(tokenizer, context, max_length=max_length)
    full_ids = full_ids[0, :max_length]
    n_full = full_ids.shape[0]
    prefix_ids = full_ids[:prefix_len].unsqueeze(0)
    n_prefix = min(prefix_ids.shape[1], prefix_len)

    cont_start = n_prefix
    cont_end = min(cont_start + cont_len, n_full)
    cont_ids = full_ids[cont_start:cont_end].unsqueeze(0)
    n_cont = cont_ids.shape[1]

    t0 = time.perf_counter()

    with torch.no_grad():
        cache = DynamicCache()
        if press is not None:
            with press(model):
                outputs = model(
                    prefix_ids,
                    past_key_values=cache,
                    cache_position=torch.arange(n_prefix, device="cpu"),
                    use_cache=True,
                )
        else:
            outputs = model(
                prefix_ids,
                past_key_values=cache,
                cache_position=torch.arange(n_prefix, device="cpu"),
                use_cache=True,
            )

        cont_cache_pos = torch.arange(n_prefix, n_prefix + n_cont, device="cpu")
        outputs = model(
            cont_ids,
            past_key_values=cache,
            cache_position=cont_cache_pos,
            use_cache=True,
            labels=cont_ids,
        )
        loss = outputs.loss.item()

    dt = (time.perf_counter() - t0) * 1000.0

    avg_kv = 0
    n_layers = 0
    for layer_cache in cache.layers:
        if hasattr(layer_cache, "keys") and isinstance(layer_cache.keys, torch.Tensor):
            avg_kv += layer_cache.keys.shape[2]
            n_layers += 1
    avg_kv = avg_kv / max(n_layers, 1) if n_layers > 0 else 0

    import math

    ppl = math.exp(loss)

    return {
        "perplexity": ppl,
        "loss": loss,
        "n_prefix_tokens": n_prefix,
        "n_continuation_tokens": n_cont,
        "inference_ms": dt,
        "avg_kv_entries": avg_kv,
    }
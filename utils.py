import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, QuantizedCache, Qwen2ForCausalLM
from transformers.cache_utils import QuantizedLayer

# MODEL_PATHS = {
#     "TinyLlama": r".models\TinyLlama-1.1B-Chat-v1.0",
#     "Qwen": r".models\Qwen2.5-0.5B-Instruct",
# }


MODEL_ID = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}

SUPPORTED_CTX_TYPES = ("prose", "code")


def load_model(model_name, eager=False):
    model_id = MODEL_ID[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"torch_dtype": torch.float16, "device_map": "cpu"}
    if eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    model = model.to(torch.bfloat16)
    return model, tokenizer


def tokenize(tokenizer, text: str, max_length: int = 1024) -> torch.Tensor:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
    return torch.tensor(tokens, dtype=torch.long, device="cpu").unsqueeze(0)

def dequantize_layer(cache_layer: QuantizedLayer) -> tuple[torch.Tensor, torch.Tensor]:
    keys = cache_layer._dequantize(cache_layer._quantized_keys)
    values = cache_layer._dequantize(cache_layer._quantized_values)
    return keys, values


def extract_keys_and_values(cache: Cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the keys and values from a given cache layer,
    handling both quantized and unquantized caches.
    """
    if isinstance(cache, QuantizedCache):
        keys, values = dequantize_layer(cache.layers[layer_idx])
    else:
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
    return keys, values
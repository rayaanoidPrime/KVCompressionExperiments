from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, QuantizedCache, Qwen2ForCausalLM
from transformers.cache_utils import QuantizedLayer

WIKITEXT_PATH = Path("llamacpp_baseline_results") / "wikitext-2-test.txt"

MODEL_ID = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}

SUPPORTED_CTX_TYPES = ("prose", "code")


def load_model(model_name, device:str = None, eager=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = MODEL_ID[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16
    kwargs = {"torch_dtype": dtype}

    if device != "cpu":
        kwargs["device_map"] = device

    if eager:
        kwargs["attn_implementation"] = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    model = model.to(device)
        
    return model, tokenizer


def tokenize(tokenizer:AutoTokenizer, text: str, max_length: int = 1024, device:str= None) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

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


def load_wikitext(tokenizer: AutoTokenizer, seq_len, device:str):
    """
    Load wikitext-2 text, concatenate all the text, tokenize 
    and return the first n overlapping chunks
    """
    fulltext = WIKITEXT_PATH.read_text(encoding='utf-8')
    tokens = tokenizer(fulltext, return_tensors='pt',add_special_tokens=False,
                       trucation=True, max_length=seq_len)["input_ids"]
    
    if tokens.shape[1] < seq_len:
        raise ValueError(f"File only contains {tokens.shape[1]} tokens, requested {seq_len} tokens")
    if device:
        tokens = tokens.to(device)

    return tokens 

    

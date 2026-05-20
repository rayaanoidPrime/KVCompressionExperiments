import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from typing import Any
from transformers import (
    Cache,
    LlamaForCausalLM,
    PreTrainedModel,
    QuantizedCache,
    Qwen2ForCausalLM,
)


logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)
from utils import extract_keys_and_values

@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.

    This class provides the foundation for implementing various key-value cache compression
    techniques. Subclasses must implement the `compress` method to define their specific
    compression logic.

    The compression is applied during prefilling and (optionally) every decoding step.
    """
    decoding: bool = False  # whether to apply compression during decoding steps as well
    
    def __post_init__(self) -> None:
        if self.decoding:
            logger.warning("Decoding compression enabled")

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Optional hook called once, jsut before hook are registered.
        override to initialize anything that requires access to the model, 
        eg. hidden_size, num_heads, ...
        """
        pass

    def is_prefilling(self,hidden_states: torch.Tensor, kwargs: dict) -> bool:
        """
        Helper method to determine if we're in the prefilling phase.
        During prefilling, cache_position is less than or equal to the current sequence length (q_len).
        During decoding, cache_position exceeds q_len.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
        kwargs : dict
            Keyword arguments from the attention layer's forward pass, expected to contain:
            - cache_position: A list of position indices indicating the current position in the sequence.
        """
        q_len = hidden_states.shape[1]
        cache_position = kwargs.get("cache_position", None)
        if cache_position is None:
            return True # default to prefilling if cache_position is not provided
        return cache_position is not None and cache_position[-1] <= q_len
    
    def should_compress(self, hidden_states: torch.Tensor, kwargs: dict) -> bool:
        """
        Determines whether compression should be applied based on the current phase (prefilling or decoding).

        Parameters
        ----------
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
        kwargs : dict
            Keyword arguments from the attention layer's forward pass.

        Returns
        -------
        bool
            True if compression should be applied, False otherwise.
        """
        if self.is_prefilling(hidden_states, kwargs):
            return True
        return self.decoding
    

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where compression is applied.
        hidden_states : torch.Tensor
            Buffererd hidden states of the current layer from recent decodig steps with shape  [batch, buffer_len, hidden_dim].
            These represent the input to the attention layer.
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are keys ready for compression.
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are values ready for compression.
        attentions : torch.Tensor
            Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if attention weights are not computed or needed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the compressed keys and values tensors. The returned tensors
            should have reduced sequence length dimension compared to the input tensors.
        """

        raise NotImplementedError("compress method must be implemented in subclass")

    @staticmethod
    def _write_to_cache(cache: Cache, cache_layer: Any, keys: torch.Tensor, values: torch.Tensor) -> None:
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys   = cache_layer._quantize(keys,   axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            # clear the dense buffers
            cache_layer.keys   = torch.zeros(0, dtype=keys.dtype,   device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys   = keys
            cache_layer.values = values


    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression by:
        1. Checking if we're still in pre-filling phase or decoding phase 
        (based on cache position and sequence length).
        2. If decoding, only compress if self.decoding is True.
        3. Extracting keys and values from the cache (handling quantization)
        4. Calling the compress method to reduce the cache size
        5. Updating the cache with compressed keys and values

        The hook ensures compression is during both pre-filling and decoding phases and
        handles both quantized and unquantized caches.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        input : list[torch.Tensor]
            Input tensors to the forward pass of the attention layer. This parameter
            is provided by PyTorch's hook mechanism but not used in the default implementation.
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method, including:
            - hidden_states: Input embeddings to the attention layer
            - past_key_values: The KV cache object being modified
            - cache_position: Position indices indicating where we are in the sequence
            - position_embeddings: RoPE embeddings if applicable
        output : list
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        list
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        layer_idx = module.layer_idx

        if cache is None:
            logger.warning(f"Cache is None in forward hook of layer {layer_idx}. Skipping compression.")
            return output

        # skip comprssion if we're past the prefilling phase and self.decoding is False
        if not self.should_compress(hidden_states, kwargs):
            return output
        
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        # output[1] is attn weights when output_attentions=True; else None
        attentions = output[1] if len(output) > 1 and output[1] is not None else None
                    
        # reconstructed keys, values for quantization comrpessions,
        # reduced keys and values for eviction compressions.
        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

        self._write_to_cache(cache, cache_layer, keys, values)

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.

        This method registers forward hooks on all attention layers of the model to enable
        automatic KV cache compression during the pre-filling phase. The hooks are automatically
        removed when exiting the context manager.

        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply compression to.
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(
                            f"Model type {type(model).__name__} has not been tested. "
                            f"Supported models: {[m.__name__ for m in SUPPORTED_MODELS]}"
                        )
        self.post_init_from_model(model)
        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:         
                # Propagate the shared rotary embedding to the attention layer
                # only when the layer doesn't already have its own instance.
                if not hasattr(layer.self_attn, "rotary_emb") or layer.self_attn.rotary_emb is None:
                    layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
                
                # Opt-in pre-RoPE key hook for KVQuantPress only
                if getattr(self, "_needs_pre_rope_keys", False):
                    hooks.append(layer.self_attn.k_proj.register_forward_hook(self._pre_rope_key_hook))

            yield
        finally:
            for h in hooks:
                h.remove()

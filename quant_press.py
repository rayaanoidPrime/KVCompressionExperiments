from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import Any

import torch
from torch import nn
from torch import Tensor
from methods.base_press import BasePress
from utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class QuantizedPress(BasePress):
    """
    Base class for all quantisation-based KV-cache compression methods.

    Extends BasePress with
    ----------------------
    - A per-layer hidden-states buffer for the decoding phase.
    - A configurable compression interval (compress every N decoding steps).
    - An abstract encode / decode interface that concrete subclasses implement.

    Concrete subclasses only need to implement
    ------------------------------------------
    _encode(x)  -> (compressed, meta)
    _decode(compressed, meta, original_shape) -> torch.Tensor

    Everything else — buffering, timing, cache write-back, stats — is handled
    here.

    Compression stats
    -----------------
    Each call to compress() appends a CompressionStat to self.stats.
    Call self.reset_stats() to clear between runs.
    """

    # Compress every N steps (default: every step)
    compression_interval: int = 1

    # Whether to accumulate hidden states across decoding steps
    _needs_hidden_states_buffer: bool = False

    # Max number of hidden-state slices to retain (0 = unlimited)
    _hidden_states_buffer_size: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate compression_interval
        if self.compression_interval < 1:
            raise ValueError(
                f"compression_interval must be >= 1, got {self.compression_interval}"
            )

        if self._needs_hidden_states_buffer:
            # Key: layer_idx  Value: list of [B, 1, H] tensors
            self._hidden_states_buffer: dict[int, list[torch.Tensor]] = defaultdict(list)

        # Number of decoding steps taken since the last compression per layer
        self._layer_step_counts: dict[int, int] = defaultdict(int)

        # Compression statistics collected across all forward passes
        self.stats: list[dict] = []

    def reset_stats(self) -> None:
        """Clear all collected compression statistics."""
        self.stats.clear()

    def _reset_layer_state(self, layer_idx: int) -> None:
        """Reset per-layer buffer and step counter after compression."""
        self._layer_step_counts[layer_idx] = 0
        if self._needs_hidden_states_buffer:
            self._hidden_states_buffer[layer_idx] = []

    @abstractmethod
    def _encode(self, x: Tensor) -> tuple[Any, dict]:
        """
        Encode (quantise) a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The tensor to encode (keys or values).

        Returns
        -------
        compressed : Any
            The compressed representation (e.g. an integer tensor).
        meta : dict
            Any metadata required to reconstruct the original tensor
            (e.g. scale, zero-point, codebook).
        """
        ...

    @abstractmethod
    def _decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        """
        Decode (dequantise) a previously encoded tensor.

        Parameters
        ----------
        compressed : Any
            Output of _encode().
        meta : dict
            Metadata returned by _encode().
        original_shape : tuple
            Shape of the original tensor before encoding.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor in the original dtype.
        """
        ...

    def _roundtrip(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, int, int]:
        """
        Encode then immediately decode keys and values.

        Returns
        -------
        reconstructed_keys : torch.Tensor
        reconstructed_values : torch.Tensor
        original_bytes : int
            Memory footprint of the original tensors.
        compressed_bytes : int
            Memory footprint of the compressed representations.
        """
        original_bytes = (
            keys.numel() * keys.element_size()
            + values.numel() * values.element_size()
        )

        compressed_k, meta_k = self._encode(keys)
        compressed_v, meta_v = self._encode(values)

        compressed_bytes = (
            self._measure_bytes(compressed_k, meta_k)
            + self._measure_bytes(compressed_v, meta_v)
        )

        reconstructed_k = self._decode(compressed_k, meta_k, keys.shape)
        reconstructed_v = self._decode(compressed_v, meta_v, values.shape)

        return reconstructed_k, reconstructed_v, original_bytes, compressed_bytes

    @staticmethod
    def _measure_bytes(compressed: Any, meta: dict) -> int:
        """
        Return the byte size of a compressed tensor plus its metadata.
        Works for torch.Tensor compressed representations; subclasses may
        override for other formats.
        """
        total = 0
        if isinstance(compressed, torch.Tensor):
            total += compressed.numel() * compressed.element_size()
        for v in meta.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
        return total

    def compress(
        self,
        module: nn.Module,
        hidden_states: Tensor,
        keys: Tensor,
        values: Tensor,
        attentions: Tensor | None,
        kwargs: dict,
    ) -> tuple[Tensor, Tensor]:
        """
        Quantise keys and values via _encode → _decode roundtrip and
        record compression statistics.

        hidden_states is accepted for API compatibility (and future
        calibration-aware schemes) but is not used in the base roundtrip.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where quantization is applied.
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor or None
            Attention weights, may be None if not computed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Quantised keys and values tensors.
        """
        reconstructed_k, reconstructed_v, orig_bytes, comp_bytes = self._roundtrip(keys, values)

        ratio = orig_bytes / comp_bytes if comp_bytes > 0 else float("inf")
        stat = {
            "layer_idx":         module.layer_idx,
            "original_bytes":    orig_bytes,
            "compressed_bytes":  comp_bytes,
            "compression_ratio": ratio,
            "dtype_before":      str(keys.dtype),
        }
        self.stats.append(stat)
        logger.debug(
            "[QuantizedPress] layer %d: %d B -> %d B  (ratio %.2fx)",
            module.layer_idx, orig_bytes, comp_bytes, ratio,
        )
        return reconstructed_k, reconstructed_v

    def forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict,
        output: list,
    ) -> list:
        """
        Extended hook that handles both prefilling and buffered decoding.

        Prefilling path
        ---------------
        Delegates directly to BasePress.forward_hook — no buffering needed
        because hidden_states already spans the full prompt.

        Decoding path
        -------------
        1. Append the current hidden-state slice to the per-layer buffer.
        2. Increment the step counter.
        3. When step counter reaches compression_interval, concatenate the
           buffer, compress, write back to cache, and reset state.
        """
        hidden_states = kwargs["hidden_states"]
        layer_idx     = module.layer_idx

        # ── Prefilling ────────────────────────────────────────────────────────
        if self.is_prefilling(hidden_states, kwargs):
            return super().forward_hook(module, input, kwargs, output)

        # ── Decoding — skip entirely if not enabled ───────────────────────────
        if not self.decoding:
            return output

        # ── Hidden-states buffering ───────────────────────────────────────────
        if self._needs_hidden_states_buffer:
            self._hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
            # Cap buffer to the most recent N slices when a limit is set
            if self._hidden_states_buffer_size > 0:
                self._hidden_states_buffer[layer_idx] = (
                    self._hidden_states_buffer[layer_idx][-self._hidden_states_buffer_size:]
                )

        # ── Step counter & compression trigger ───────────────────────────────
        self._layer_step_counts[layer_idx] += 1
        if self._layer_step_counts[layer_idx] < self.compression_interval:
            return output  # not yet — keep accumulating

        logger.debug(
            "[QuantizedPress] Triggering decoding compression on layer %d after %d steps.",
            layer_idx, self._layer_step_counts[layer_idx],
        )

        # ── Fetch keys / values from cache ────────────────────────────────────
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[layer_idx]
        keys, values = extract_keys_and_values(cache, layer_idx)

        attentions = (
            output[1]
            if len(output) > 1 and output[1] is not None
            else None
        )

        # ── Resolve hidden states to pass to compress() ───────────────────────
        if self._needs_hidden_states_buffer:
            buffered_hidden_states = torch.cat(
                self._hidden_states_buffer[layer_idx], dim=1
            )
        else:
            buffered_hidden_states = hidden_states

        # ── Compress & write back ─────────────────────────────────────────────
        keys, values = self.compress(
            module, buffered_hidden_states, keys, values, attentions, kwargs
        )
        self._write_to_cache(cache, cache_layer, keys, values)

        # ── Reset per-layer state for the next compression window ─────────────
        self._reset_layer_state(layer_idx)

        return output
import csv
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Any, List

from attrs import asdict
import torch
from torch import Tensor

from KVCompressionExperiments.base_press import BasePress


@dataclass
class CompressionStat:
    """One row of compression information for a specific layer / call."""
    layer_name: str                # e.g. "model.encoder.layers.3.self_attn"           
    original_bytes: int            # bytes before quantisation
    compressed_bytes: int          # bytes after “compression”
    compression_ratio: float       # original / compressed
    dtype_before: str
    dtype_after: str

class PrecisionReductionPress(BasePress):
    """
    A simple press that simulates compression by reducing the precision of keys and values.

    This is a baseline method that can be used to understand the impact of precision reduction
    on the attention mechanism. It does not perform any actual compression, but rather simulates
    it by quantizing the keys and values to a lower precision (e.g., from float32 to float16).
    """

    def __init__(self, target_dtype: torch.dtype = torch.float16):
        self.target_dtype = target_dtype
        # a list that will be filled by the forward‑hook
        self.stats: List[CompressionStat] = []

    def _encode(self, x: Tensor) -> tuple[Any, dict]:
        raise NotImplementedError

    def _decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        raise NotImplementedError

    def _roundtrip(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, int, int]:
        original_bytes = keys.numel() * keys.element_size() + values.numel() * values.element_size()
        compressed_k, meta_k = self._encode(keys)
        compressed_v, meta_v = self._encode(values)
        compressed_bytes = self._compressed_bytes(compressed_k, meta_k) + self._compressed_bytes(
            compressed_v, meta_v
        )
        reconstructed_k = self._decode(compressed_k, meta_k, keys.shape)
        reconstructed_v = self._decode(compressed_v, meta_v, values.shape)

        return reconstructed_k, reconstructed_v, original_bytes, compressed_bytes

    def compress(
        self,
        module: torch.nn.Module,
        hidden_states: Tensor,
        keys: Tensor,
        values: Tensor,
        attentions: Tensor,
        kwargs: dict,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where quantization is applied.
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
            These represent the input to the attention layer.
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are keys ready for quantization.
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are values ready for quantization.
        attentions : torch.Tensor
            Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if attention weights are not computed or needed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the quantized keys and values tensors. The returned tensors
            should have the target dtype compared to the input tensors.
        """
        reconstructed_k, reconstructed_v, orig_bytes, comp_bytes = self._roundtrip(keys, values)

        # record stats for this call
        if layer_name is None:
            # Fallback – use the module's Python repr if the caller didn’t give a name
            layer_name = f"{module.__class__.__name__}_{id(module)}"

        ratio = orig_bytes / comp_bytes if comp_bytes > 0 else float("inf")
        stat = CompressionStat(
            layer_name=layer_name,
            original_bytes=orig_bytes,
            compressed_bytes=comp_bytes,
            compression_ratio=ratio,
            dtype_before=str(keys.dtype),
            dtype_after=str(self.target_dtype),
        )
        self.stats.append(stat)
        return reconstructed_k, reconstructed_v
    
    def dump_stats_to_csv(self, path: Path) -> None:
        """Write the collected rows to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf‑8") as f:
            writer = csv.DictWriter(
                f, fieldnames=[f.name for f in CompressionStat.__dataclass_fields__.values()]
            )
            writer.writeheader()
            for row in self.stats:
                writer.writerow(asdict(row))

    

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class InstrumentedPress(BasePress):
    compression_ratio: float = 0.0

    captured_keys: dict = field(default_factory=dict, init=False)
    captured_values: dict = field(default_factory=dict, init=False)
    layer_stats: dict = field(default_factory=dict, init=False)

    def compress(
        self,
        module: nn.Module,
        hidden_states: Tensor,
        keys: Tensor,
        values: Tensor,
        attentions: Tensor,
        kwargs: dict,
    ) -> tuple[Tensor, Tensor]:
        layer_idx = getattr(module, "layer_idx", len(self.captured_keys))
        self.captured_keys[layer_idx] = keys.detach().cpu()
        self.captured_values[layer_idx] = values.detach().cpu()
        self.layer_stats[layer_idx] = self._compute_layer_stats(
            keys.detach().cpu().float(), values.detach().cpu().float()
        )
        return keys, values

    def _compute_layer_stats(self, keys: Tensor, values: Tensor) -> dict:
        B, H, S, D = keys.shape

        k_deltas = keys[:, :, 1:, :] - keys[:, :, :-1, :]
        v_deltas = values[:, :, 1:, :] - values[:, :, :-1, :]
        k_delta_norm = float(k_deltas.norm(dim=-1).mean())
        k_abs_norm = float(keys.norm(dim=-1).mean())
        v_delta_norm = float(v_deltas.norm(dim=-1).mean())
        v_abs_norm = float(values.norm(dim=-1).mean())
        k_delta_compressibility = 1.0 - k_delta_norm / max(k_abs_norm, 1e-8)
        v_delta_compressibility = 1.0 - v_delta_norm / max(v_abs_norm, 1e-8)

        k_0 = keys[0]
        k_var_channel = float(k_0.var(dim=-1).mean())
        k_var_token = float(k_0.var(dim=-2).mean())
        channel_structure_ratio = k_var_channel / max(k_var_token, 1e-8)

        k_channel_var = keys.reshape(-1, D).var(dim=0)
        k_channel_var_list = k_channel_var.tolist()

        k_token_var = keys[0].var(dim=-1)
        k_token_var_list = k_token_var.tolist()

        k_flat = keys.flatten()
        k_mean = k_flat.mean()
        k_std = k_flat.std()
        k_outlier_fraction = float(((k_flat - k_mean).abs() > 3 * k_std).float().mean())

        try:
            k_mat = keys[0, 0].float()
            if k_mat.shape[0] >= k_mat.shape[1]:
                S_vals = torch.linalg.svd(k_mat, full_matrices=False).S
                sv_top50_energy = float(
                    S_vals[: len(S_vals) // 2].pow(2).sum() / S_vals.pow(2).sum().clamp(1e-8)
                )
                sv_cumvar = (S_vals.pow(2).cumsum(0) / S_vals.pow(2).sum().clamp(1e-8)).tolist()
                effective_rank_90 = int((torch.tensor(sv_cumvar) < 0.90).sum().item()) + 1
            else:
                sv_top50_energy = float("nan")
                sv_cumvar = []
                effective_rank_90 = -1
        except Exception:
            sv_top50_energy = float("nan")
            sv_cumvar = []
            effective_rank_90 = -1

        head0 = keys[0, 0]
        autocorr = []
        for lag in range(1, min(21, head0.shape[0])):
            a = head0[:-lag]
            b = head0[lag:]
            sim = F.cosine_similarity(a, b, dim=-1).mean().item()
            autocorr.append(sim)
        while len(autocorr) < 20:
            autocorr.append(float("nan"))

        return {
            "k_delta_compressibility": k_delta_compressibility,
            "v_delta_compressibility": v_delta_compressibility,
            "k_delta_norm": k_delta_norm,
            "k_abs_norm": k_abs_norm,
            "v_delta_norm": v_delta_norm,
            "v_abs_norm": v_abs_norm,
            "channel_structure_ratio": channel_structure_ratio,
            "k_var_channel": k_var_channel,
            "k_var_token": k_var_token,
            "k_channel_var": k_channel_var_list,
            "k_token_var": k_token_var_list,
            "k_outlier_fraction": k_outlier_fraction,
            "sv_top50_energy": sv_top50_energy,
            "sv_cumvar": sv_cumvar,
            "effective_rank_90": effective_rank_90,
            "autocorr_lags_1_to_20": autocorr,
            "seq_len": S,
            "head_dim": D,
            "n_heads": H,
        }

    def save(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "instrumented_stats.json", "w") as f:
            json.dump(self.layer_stats, f, indent=2)

        torch.save(self.captured_keys, output_dir / "captured_keys.pt")
        torch.save(self.captured_values, output_dir / "captured_values.pt")

        csv_fields = [
            "layer_idx",
            "k_delta_compressibility",
            "v_delta_compressibility",
            "k_delta_norm",
            "k_abs_norm",
            "v_delta_norm",
            "v_abs_norm",
            "channel_structure_ratio",
            "k_var_channel",
            "k_var_token",
            "k_outlier_fraction",
            "sv_top50_energy",
            "effective_rank_90",
            "seq_len",
            "head_dim",
            "n_heads",
        ]
        with open(output_dir / "layer_stats_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for lidx in sorted(self.layer_stats.keys()):
                s = self.layer_stats[lidx]
                row = {"layer_idx": lidx}
                for col in csv_fields:
                    if col != "layer_idx":
                        row[col] = s.get(col, None)
                writer.writerow(row)

    def load(self, path: Union[str, Path]):
        path = Path(path)

        with open(path / "instrumented_stats.json", "r") as f:
            self.layer_stats = json.load(f)

        self.captured_keys = torch.load(path / "captured_keys.pt")
        self.captured_values = torch.load(path / "captured_values.pt")

    def reset(self):
        self.captured_keys.clear()
        self.captured_values.clear()
        self.layer_stats.clear()

    

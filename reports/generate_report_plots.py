"""Generate additional report plots from existing instrumented data."""
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INST_DIR = ROOT / "experiment-outputs" / "instrumented"
OUT_DIR = ROOT / "report-plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Styling --
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

MODEL_STYLE = {
    "TinyLlama": {"color": "#2196F3", "marker": "o", "ls": "-"},
    "Qwen":      {"color": "#FF9800", "marker": "s", "ls": "--"},
}
CTX_STYLE = {
    "prose": {"alpha": 1.0, "lw": 2.0},
    "code":  {"alpha": 0.45, "lw": 1.2},
}
MODEL_LABEL = {"TinyLlama": "TinyLlama-1.1B", "Qwen": "Qwen2.5-0.5B"}

# ---------------------------------------------------------------------------
# Load all CSVs
# ---------------------------------------------------------------------------
def load_all() -> pd.DataFrame:
    frames = []
    for csv_path in INST_DIR.glob("*/*/layer_stats_summary.csv"):
        model = csv_path.parent.parent.name
        ctx   = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df["model"] = model
        df["context"] = ctx
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

df = load_all()

# ===========================================================================
# Plot 1 -- Effective rank (rank_90) by layer
# ===========================================================================
def plot_effective_rank(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in ["TinyLlama", "Qwen"]:
        for ctx in ["prose", "code"]:
            sub = df[(df["model"] == model) & (df["context"] == ctx)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer_idx")
            label = f"{MODEL_LABEL[model]}  ({ctx})"
            style = MODEL_STYLE[model]
            ctx_s = CTX_STYLE[ctx]
            ax.plot(sub["layer_idx"], sub["effective_rank_90"],
                    marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], alpha=ctx_s["alpha"],
                    linewidth=ctx_s["lw"], label=label, markersize=5)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank (90% variance)")
    ax.set_title("Effective Rank of Key Matrices by Layer")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "effective_rank.png")
    plt.close(fig)
    print("Saved: effective_rank.png")

# ===========================================================================
# Plot 2 -- SVD top-50 energy by layer
# ===========================================================================
def plot_svd_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in ["TinyLlama", "Qwen"]:
        for ctx in ["prose", "code"]:
            sub = df[(df["model"] == model) & (df["context"] == ctx)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer_idx")
            label = f"{MODEL_LABEL[model]}  ({ctx})"
            style = MODEL_STYLE[model]
            ctx_s = CTX_STYLE[ctx]
            ax.plot(sub["layer_idx"], sub["sv_top50_energy"],
                    marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], alpha=ctx_s["alpha"],
                    linewidth=ctx_s["lw"], label=label, markersize=5)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top-50% SVD energy fraction")
    ax.set_title("SVD Energy Concentration by Layer\n(fraction of total energy in top 50% of singular values)")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0.85, 1.01)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "svd_energy.png")
    plt.close(fig)
    print("Saved: svd_energy.png")

# ===========================================================================
# Plot 3 -- Outlier fraction by layer
# ===========================================================================
def plot_outlier_fraction(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in ["TinyLlama", "Qwen"]:
        for ctx in ["prose", "code"]:
            sub = df[(df["model"] == model) & (df["context"] == ctx)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer_idx")
            label = f"{MODEL_LABEL[model]}  ({ctx})"
            style = MODEL_STYLE[model]
            ctx_s = CTX_STYLE[ctx]
            ax.plot(sub["layer_idx"], sub["k_outlier_fraction"] * 100,
                    marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], alpha=ctx_s["alpha"],
                    linewidth=ctx_s["lw"], label=label, markersize=5)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Key outlier fraction (%)")
    ax.set_title("Key Outlier Fraction by Layer\n(values > 3\u03c3 from mean)")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "outlier_fraction.png")
    plt.close(fig)
    print("Saved: outlier_fraction.png")

# ===========================================================================
# Plot 4 -- Delta compressibility by layer
# ===========================================================================
def plot_delta_compressibility(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in [
        (axes[0], "k_delta_compressibility", "Key Delta Compressibility"),
        (axes[1], "v_delta_compressibility", "Value Delta Compressibility"),
    ]:
        for model in ["TinyLlama", "Qwen"]:
            for ctx in ["prose", "code"]:
                sub = df[(df["model"] == model) & (df["context"] == ctx)]
                if sub.empty:
                    continue
                sub = sub.sort_values("layer_idx")
                label = f"{MODEL_LABEL[model]}  ({ctx})"
                style = MODEL_STYLE[model]
                ctx_s = CTX_STYLE[ctx]
                ax.plot(sub["layer_idx"], sub[col],
                        marker=style["marker"], color=style["color"],
                        linestyle=style["ls"], alpha=ctx_s["alpha"],
                        linewidth=ctx_s["lw"], label=label, markersize=5)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Delta compressibility")
        ax.set_title(title)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")

    fig.suptitle("Delta Compressibility by Layer", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_compressibility.png")
    plt.close(fig)
    print("Saved: delta_compressibility.png")

# ===========================================================================
# Plot 5 -- Channel structure ratio by layer
# ===========================================================================
def plot_channel_structure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in ["TinyLlama", "Qwen"]:
        for ctx in ["prose", "code"]:
            sub = df[(df["model"] == model) & (df["context"] == ctx)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer_idx")
            label = f"{MODEL_LABEL[model]}  ({ctx})"
            style = MODEL_STYLE[model]
            ctx_s = CTX_STYLE[ctx]
            ax.plot(sub["layer_idx"], sub["channel_structure_ratio"],
                    marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], alpha=ctx_s["alpha"],
                    linewidth=ctx_s["lw"], label=label, markersize=5)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Channel structure ratio  (k_var_channel / k_var_token)")
    ax.set_title("Channel vs Token Variance Ratio by Layer\n(higher = more channel-structured)")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.axhline(y=1, color="gray", linewidth=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "channel_structure_ratio.png")
    plt.close(fig)
    print("Saved: channel_structure_ratio.png")

# ===========================================================================
# Plot 6 -- Autocorrelation (from JSON, Qwen prose layer 0 only)
# ===========================================================================
def plot_autocorrelation():
    json_path = INST_DIR / "Qwen" / "prose" / "instrumented_stats.json"
    if not json_path.exists():
        print("Skipping autocorr: JSON not found")
        return
    with open(json_path) as f:
        stats = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 4))
    layer0 = stats.get("0", {})
    autocorr = layer0.get("autocorr_lags_1_to_20", [])
    if not autocorr:
        print("Skipping autocorr: no data")
        plt.close(fig)
        return

    lags = list(range(1, len(autocorr) + 1))
    ax.bar(lags, autocorr, color="#2196F3", alpha=0.8, width=0.7)
    ax.set_xlabel("Token lag")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Key Autocorrelation (Qwen2.5-0.5B, prose, layer 0)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "autocorrelation.png")
    plt.close(fig)
    print("Saved: autocorrelation.png")

# ===========================================================================
# Plot 7 -- SVD scree (cumulative variance, Qwen prose layer 0 only)
# ===========================================================================
def plot_scree():
    json_path = INST_DIR / "Qwen" / "prose" / "instrumented_stats.json"
    if not json_path.exists():
        print("Skipping scree: JSON not found")
        return
    with open(json_path) as f:
        stats = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 4))
    layer0 = stats.get("0", {})
    cumvar = layer0.get("sv_cumvar", [])
    if not cumvar:
        print("Skipping scree: no data")
        plt.close(fig)
        return

    ranks = list(range(1, len(cumvar) + 1))
    ax.plot(ranks, cumvar, color="#2196F3", linewidth=2)
    ax.axhline(y=0.90, color="#FF9800", linestyle="--", linewidth=1, label="90% variance")
    ax.set_xlabel("Singular value rank")
    ax.set_ylabel("Cumulative variance fraction")
    ax.set_title("SVD Scree Plot (Qwen2.5-0.5B, prose, layer 0)")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, len(cumvar))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "svd_scree.png")
    plt.close(fig)
    print("Saved: svd_scree.png")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    plot_effective_rank(df)
    plot_svd_energy(df)
    plot_outlier_fraction(df)
    plot_delta_compressibility(df)
    plot_channel_structure(df)
    plot_autocorrelation()
    plot_scree()
    print("\nAll plots generated in", OUT_DIR)

#!/usr/bin/env python3
"""
plot_baseline.py  –  Plot decode latency and perplexity sweeps from run_baseline.sh outputs.

Usage:
    python plot_baseline.py [latency_csv] [ppl_csv] [output_dir]

Defaults:
    latency_csv : ~/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results/latency_vs_ctx.csv
    ppl_csv     : ~/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results/ppl_vs_ctx.csv
    output_dir  : same directory as the CSVs
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Defaults ──────────────────────────────────────────────────────────────────
_BASE = os.path.expanduser(
    "~/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results"
)
LATENCY_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_BASE, "latency_vs_ctx.csv")
PPL_CSV     = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_BASE, "ppl_vs_ctx.csv")
OUT_DIR     = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(LATENCY_CSV)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
KV_ORDER  = ["f16", "q8_0", "q4_0"]
KV_COLORS = {"f16": "#2196F3", "q8_0": "#FF9800", "q4_0": "#4CAF50"}
KV_LABELS = {"f16": "FP16 KV", "q8_0": "Q8_0 KV", "q4_0": "Q4_0 KV"}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Decode latency vs n_depth (prefill context length)
# ══════════════════════════════════════════════════════════════════════════════
def plot_latency(csv_path: str, out_dir: str) -> None:
    print(f"[INFO] Reading latency CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Strip surrounding quotes that llama-bench emits
    df.columns = df.columns.str.strip().str.strip('"')
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.strip('"')

    # Columns we need: n_depth (prefill ctx), type_k (KV type), avg_ns / stddev_ns
    required = {"n_depth", "type_k", "avg_ns", "stddev_ns"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Latency CSV missing columns: {missing}\nFound: {list(df.columns)}")

    df["n_depth"]   = pd.to_numeric(df["n_depth"],   errors="coerce")
    df["avg_ns"]    = pd.to_numeric(df["avg_ns"],     errors="coerce")
    df["stddev_ns"] = pd.to_numeric(df["stddev_ns"],  errors="coerce")

    # Convert ns → ms
    df["avg_ms"]    = df["avg_ns"]    / 1e6
    df["stddev_ms"] = df["stddev_ns"] / 1e6

    # Keep only decode rows: n_gen == 1 and n_prompt == 0
    if {"n_gen", "n_prompt"}.issubset(df.columns):
        df["n_gen"]    = pd.to_numeric(df["n_gen"],    errors="coerce")
        df["n_prompt"] = pd.to_numeric(df["n_prompt"], errors="coerce")
        df = df[(df["n_gen"] == 1) & (df["n_prompt"] == 0)]

    df = df.dropna(subset=["n_depth", "avg_ms"])
    df = df.sort_values("n_depth")

    fig, ax = plt.subplots(figsize=(8, 5))

    for kv in KV_ORDER:
        sub = df[df["type_k"] == kv]
        if sub.empty:
            print(f"[WARN] No latency data for kv={kv}")
            continue
        ax.errorbar(
            sub["n_depth"], sub["avg_ms"],
            yerr=sub["stddev_ms"],
            label=KV_LABELS.get(kv, kv),
            color=KV_COLORS.get(kv, None),
            marker="o", markersize=6,
            linewidth=2, capsize=4, capthick=1.5,
        )

    ax.set_xlabel("Prefill context length  (n_depth, tokens)")
    ax.set_ylabel("Single-token decode latency  (ms)")
    ax.set_title("Decode Latency vs Prefill Context Length\n(1-token generation, KV cache pre-filled)")
    ax.legend(title="KV cache type", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    out_path = os.path.join(out_dir, "latency_vs_ctx.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 – Perplexity vs context length
# ══════════════════════════════════════════════════════════════════════════════
def plot_ppl(csv_path: str, out_dir: str) -> None:
    print(f"[INFO] Reading PPL CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Expected columns: ctx, kv_type, ppl
    required = {"ctx", "kv_type", "ppl"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"PPL CSV missing columns: {missing}\nFound: {list(df.columns)}")

    df["ctx"] = pd.to_numeric(df["ctx"], errors="coerce")
    df["ppl"] = pd.to_numeric(df["ppl"], errors="coerce")   # ERROR rows → NaN, silently dropped
    df = df.dropna(subset=["ctx", "ppl"])
    df = df.sort_values("ctx")

    fig, ax = plt.subplots(figsize=(8, 5))

    for kv in KV_ORDER:
        sub = df[df["kv_type"] == kv]
        if sub.empty:
            print(f"[WARN] No PPL data for kv={kv}")
            continue
        ax.plot(
            sub["ctx"], sub["ppl"],
            label=KV_LABELS.get(kv, kv),
            color=KV_COLORS.get(kv, None),
            marker="o", markersize=6, linewidth=2,
        )

    ax.set_xlabel("Context length  (tokens)")
    ax.set_ylabel("Perplexity (PPL)  ↓ better")
    ax.set_title("Perplexity vs Context Length\n(wikitext-2, BF16 model weights)")
    ax.legend(title="KV cache type", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    out_path = os.path.join(out_dir, "ppl_vs_ctx.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    errors = []

    if os.path.isfile(LATENCY_CSV):
        try:
            plot_latency(LATENCY_CSV, OUT_DIR)
        except Exception as e:
            errors.append(f"Latency plot failed: {e}")
            print(f"[ERROR] {e}")
    else:
        print(f"[WARN] Latency CSV not found, skipping: {LATENCY_CSV}")

    if os.path.isfile(PPL_CSV):
        try:
            plot_ppl(PPL_CSV, OUT_DIR)
        except Exception as e:
            errors.append(f"PPL plot failed: {e}")
            print(f"[ERROR] {e}")
    else:
        print(f"[WARN] PPL CSV not found, skipping: {PPL_CSV}")

    if errors:
        sys.exit(1)
    print("\n[INFO] Done.")
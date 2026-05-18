#!/usr/bin/env python3
"""
plot_baseline.py  -  Plot decode latency, prefill-generate latency, and
                     perplexity sweeps from run_baseline.sh outputs.

Usage:
    python plot_baseline.py [d_latency_csv] [pg_latency_csv] [ppl_csv] [output_dir]

Defaults:
    d_latency_csv  : ~/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results/d_latency_vs_ctx.csv
    pg_latency_csv : ~/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results/pg_latency_vs_ctx.csv
    ppl_csv        : ~/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results/ppl_vs_ctx.csv
    output_dir     : same directory as the CSVs
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
D_LATENCY_CSV  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_BASE, "d_latency_vs_ctx.csv")
PG_LATENCY_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_BASE, "pg_latency_vs_ctx.csv")
PPL_CSV        = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_BASE, "ppl_vs_ctx.csv")
# Fix 4: safe OUT_DIR default — os.path.dirname returns "" for bare filenames
OUT_DIR        = sys.argv[4] if len(sys.argv) > 4 else (os.path.dirname(D_LATENCY_CSV) or ".")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
KV_ORDER  = ["f16", "q8_0", "q4_0"]
KV_COLORS = {"f16": "#2196F3", "q8_0": "#FF9800", "q4_0": "#4CAF50"}
KV_LABELS = {"f16": "FP16 KV", "q8_0": "Q8_0 KV", "q4_0": "Q4_0 KV"}

LLAMA_BENCH_COLUMNS = [
    "build_commit", "build_number", "cpu_info", "gpu_info", "backends",
    "model_filename", "model_type", "model_size", "model_n_params",
    "n_batch", "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll",
    "type_k", "type_v", "n_gpu_layers", "n_cpu_moe", "split_mode",
    "main_gpu", "no_kv_offload", "flash_attn", "devices", "tensor_split",
    "tensor_buft_overrides", "use_mmap", "use_direct_io", "embeddings",
    "no_op_offload", "no_host", "fit_target", "fit_min_ctx",
    "n_prompt", "n_gen", "n_depth", "test_time",
    "avg_ns", "stddev_ns", "avg_ts", "stddev_ts",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Helper: load a llama-bench CSV (with or without header) ──────────────────
def _load_bench_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a llama-bench CSV.  If the first line starts with 'build_commit'
    it has a header; otherwise we inject the known column names.
    """
    with open(csv_path) as fh:
        first_line = fh.readline()

    has_header = first_line.startswith("build_commit")

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=LLAMA_BENCH_COLUMNS)

    # Strip surrounding quotes that llama-bench sometimes emits
    df.columns = df.columns.str.strip().str.strip('"')
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.strip('"')

    return df


def _prep_bench_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns and derive avg_ms / stddev_ms."""
    for col in ("n_depth", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts",
                "n_gen", "n_prompt"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Warn on unexpected NaNs introduced by the cast
    for col in ("n_prompt", "n_gen", "n_depth"):
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                print(f"[WARN] {n_null} NaN value(s) in '{col}' after numeric cast — "
                      "check for malformed rows")

    df["avg_ms"]    = df["avg_ns"]    / 1e6
    df["stddev_ms"] = df["stddev_ns"] / 1e6
    return df


def _ms_per_tok(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add avg_ms_per_tok / stddev_ms_per_tok columns derived from avg_ts
    (tokens/sec) using first-order error propagation:
        L = 1000 / T  →  σ_L = (1000 / T²) × σ_T
    """
    df = df.copy()
    df["avg_ms_per_tok"]    = 1000.0 / df["avg_ts"]
    df["stddev_ms_per_tok"] = 1000.0 * df["stddev_ts"] / (df["avg_ts"] ** 2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Decode latency vs n_depth  (n_prompt=0, n_gen>0)
# ══════════════════════════════════════════════════════════════════════════════
def plot_d_latency(csv_path: str, out_dir: str) -> None:
    print(f"[INFO] Reading decode-latency CSV: {csv_path}")
    df = _load_bench_csv(csv_path)
    df = _prep_bench_df(df)

    required = {"n_depth", "type_k", "avg_ts", "stddev_ts"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Decode-latency CSV missing columns: {missing}\nFound: {list(df.columns)}")

    # Fix 1: use n_gen > 0 (not == 1) so it works regardless of n_gen value
    df = df[(df["n_gen"] > 0) & (df["n_prompt"] == 0)]
    if df.empty:
        raise ValueError(
            "Decode-latency filter (n_gen>0, n_prompt==0) matched no rows — "
            "check n_gen / n_prompt values in the CSV"
        )

    # Fix 2: plot ms/token derived from avg_ts, not total avg_ms
    df = _ms_per_tok(df)
    df = df.dropna(subset=["n_depth", "avg_ms_per_tok"]).sort_values("n_depth")

    fig, ax = plt.subplots(figsize=(8, 5))

    for kv in KV_ORDER:
        sub = df[df["type_k"] == kv]
        if sub.empty:
            print(f"[WARN] No decode-latency data for kv={kv}")
            continue
        ax.errorbar(
            sub["n_depth"], sub["avg_ms_per_tok"],
            yerr=sub["stddev_ms_per_tok"],
            label=KV_LABELS.get(kv, kv),
            color=KV_COLORS.get(kv, None),
            marker="o", markersize=6,
            linewidth=2, capsize=4, capthick=1.5,
        )

    ax.set_xlabel("Prefill context length  (n_depth, tokens)")
    ax.set_ylabel("Decode latency per token  (ms/tok)")
    ax.set_title("Decode Latency vs Prefill Context Length\n"
                 "(n_prompt=0, KV cache pre-filled via n_depth)")
    ax.legend(title="KV cache type", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    out_path = os.path.join(out_dir, "d_latency_vs_ctx.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 – Prefill-generate latency vs n_depth  (three sub-plots)
#
#  The PG bench emits three rows per (kv_type, n_depth) combination:
#    A)  n_prompt=512, n_gen=0    → prefill-only   (ms/tok via avg_ts)
#    B)  n_prompt=0,   n_gen=128  → decode-only    (ms/tok via avg_ts)
#    C)  n_prompt=512, n_gen=128  → prefill+decode  (ms/tok via avg_ts)
#
#  All three sub-plots use ms/token for a consistent Y-axis scale.
# ══════════════════════════════════════════════════════════════════════════════
def plot_pg_latency(csv_path: str, out_dir: str) -> None:
    print(f"[INFO] Reading prefill-generate latency CSV: {csv_path}")
    df = _load_bench_csv(csv_path)
    df = _prep_bench_df(df)

    required = {"n_depth", "type_k", "avg_ts", "stddev_ts", "n_prompt", "n_gen"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"PG-latency CSV missing columns: {missing}\nFound: {list(df.columns)}")

    df = df.dropna(subset=["n_depth", "avg_ts"]).sort_values("n_depth")

    # Partition into the three test types
    df_prefill = df[(df["n_prompt"] == 512) & (df["n_gen"] == 0)].copy()    # A
    df_decode  = df[(df["n_prompt"] == 0)   & (df["n_gen"] == 128)].copy()  # B
    df_pg      = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)].copy()  # C

    for label, sub_df in (("prefill-only", df_prefill),
                           ("decode-only",  df_decode),
                           ("prefill+gen",  df_pg)):
        if sub_df.empty:
            print(f"[WARN] No rows matched for partition '{label}' — "
                  "check n_prompt / n_gen values in the CSV")

    # Fix 3: convert all three partitions to ms/token for consistent units
    df_prefill = _ms_per_tok(df_prefill)
    df_decode  = _ms_per_tok(df_decode)
    df_pg      = _ms_per_tok(df_pg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # ── Sub-plot A: prefill-only (ms/tok) ────────────────────────────────────
    ax = axes[0]
    for kv in KV_ORDER:
        sub = df_prefill[df_prefill["type_k"] == kv]
        if sub.empty:
            print(f"[WARN] No prefill-only data for kv={kv}")
            continue
        ax.errorbar(
            sub["n_depth"], sub["avg_ms_per_tok"],
            yerr=sub["stddev_ms_per_tok"],
            label=KV_LABELS.get(kv, kv),
            color=KV_COLORS.get(kv, None),
            marker="s", markersize=6, linewidth=2, capsize=4, capthick=1.5,
        )
    ax.set_xlabel("Prefill context length  (n_depth, tokens)")
    ax.set_ylabel("Prefill latency per token  (ms/tok)")
    ax.set_title("Prefill-Only Throughput\n(n_prompt=512, n_gen=0)")
    ax.legend(title="KV cache type", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Sub-plot B: decode-only (ms/tok) ─────────────────────────────────────
    ax = axes[1]
    for kv in KV_ORDER:
        sub = df_decode[df_decode["type_k"] == kv]
        if sub.empty:
            print(f"[WARN] No decode-only data for kv={kv}")
            continue
        ax.errorbar(
            sub["n_depth"], sub["avg_ms_per_tok"],
            yerr=sub["stddev_ms_per_tok"],
            label=KV_LABELS.get(kv, kv),
            color=KV_COLORS.get(kv, None),
            marker="^", markersize=6, linewidth=2, capsize=4, capthick=1.5,
        )
    ax.set_xlabel("Prefill context length  (n_depth, tokens)")
    ax.set_ylabel("Decode latency per token  (ms/tok)")
    ax.set_title("Decode-Only Throughput\n(n_prompt=0, n_gen=128)")
    ax.legend(title="KV cache type", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Sub-plot C: prefill+generate (ms/tok) ────────────────────────────────
    ax = axes[2]
    for kv in KV_ORDER:
        sub = df_pg[df_pg["type_k"] == kv]
        if sub.empty:
            print(f"[WARN] No prefill+generate data for kv={kv}")
            continue
        ax.errorbar(
            sub["n_depth"], sub["avg_ms_per_tok"],
            yerr=sub["stddev_ms_per_tok"],
            label=KV_LABELS.get(kv, kv),
            color=KV_COLORS.get(kv, None),
            marker="D", markersize=6, linewidth=2, capsize=4, capthick=1.5,
        )
    ax.set_xlabel("Prefill context length  (n_depth, tokens)")
    ax.set_ylabel("Latency per token  (ms/tok)")
    ax.set_title("Prefill + Generate Latency\n(n_prompt=512, n_gen=128)")
    ax.legend(title="KV cache type", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Fix 9: y=1.02 is safe here because bbox_inches="tight" is used on savefig
    fig.suptitle("Prefill-Generate Latency vs Prefill Context Length", fontsize=13, y=1.02)
    out_path = os.path.join(out_dir, "pg_latency_vs_ctx.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Perplexity vs context length
# ══════════════════════════════════════════════════════════════════════════════
def plot_ppl(csv_path: str, out_dir: str) -> None:
    print(f"[INFO] Reading PPL CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = {"ctx", "kv_type", "ppl"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"PPL CSV missing columns: {missing}\nFound: {list(df.columns)}")

    df["ctx"] = pd.to_numeric(df["ctx"], errors="coerce")
    df["ppl"] = pd.to_numeric(df["ppl"], errors="coerce")  # ERROR rows → NaN, dropped below
    df = df.dropna(subset=["ctx", "ppl"]).sort_values("ctx")

    if df.empty:
        raise ValueError("PPL CSV contains no valid (ctx, ppl) rows after parsing")

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
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    errors = []

    if os.path.isfile(D_LATENCY_CSV):
        try:
            plot_d_latency(D_LATENCY_CSV, OUT_DIR)
        except Exception as e:
            errors.append(f"Decode-latency plot failed: {e}")
            print(f"[ERROR] {e}")
    else:
        print(f"[WARN] Decode-latency CSV not found, skipping: {D_LATENCY_CSV}")

    if os.path.isfile(PG_LATENCY_CSV):
        try:
            plot_pg_latency(PG_LATENCY_CSV, OUT_DIR)
        except Exception as e:
            errors.append(f"PG-latency plot failed: {e}")
            print(f"[ERROR] {e}")
    else:
        print(f"[WARN] PG-latency CSV not found, skipping: {PG_LATENCY_CSV}")

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
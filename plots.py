import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1B2A4A"
BLUE = "#2563EB"
TEAL = "#0D9488"
AMBER = "#D97706"
RED = "#DC2626"
PURPLE = "#7C3AED"
GRAY = "#6B7280"


def plot_kv_growth(df, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in [("TinyLlama", BLUE), ("Qwen", TEAL)]:
        sub = df[df["model_name"] == model_name]
        ax.plot(sub["seq_len"], sub["kv_cache_mb"], label=model_name, color=color, linewidth=2)
    ax.set_xlabel("Sequence Length (log scale)")
    ax.set_ylabel("KV Cache (MB)")
    ax.set_xscale("log")
    ax.legend()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_heatmap(captured_keys_dict, tokenizer, save_path, layer_idx, context_type_label):
    # Derive models and contexts from the keys present in captured_keys_dict
    present_keys = list(captured_keys_dict.keys())
    ctxs   = sorted({k[1] for k in present_keys})
    models = sorted({k[0] for k in present_keys})

    fig, axes = plt.subplots(len(ctxs), len(models), figsize=(7 * len(models), 3 * len(ctxs)))
    # Normalise axes to always be 2-D array
    if len(ctxs) == 1 and len(models) == 1:
        axes = [[axes]]
    elif len(ctxs) == 1:
        axes = [axes]
    elif len(models) == 1:
        axes = [[ax] for ax in axes]

    for row_i, ctx in enumerate(ctxs):
        row_data = []
        for model_name in models:
            key = (model_name, ctx)
            if key in captured_keys_dict:
                k = captured_keys_dict[key][layer_idx][0].norm(dim=-1).numpy()
            else:
                k = np.zeros((1, 1))
            row_data.append(k)
        vmax = max(d.max() for d in row_data)
        for col_j, model_name in enumerate(models):
            ax = axes[row_i][col_j]
            im = ax.imshow(row_data[col_j], aspect="auto", cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"{model_name} - {ctx}")
            ax.set_xlabel("Sequence Position")
            ax.set_ylabel("Head Index")
            plt.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kv_distributions(captured_keys, captured_values, save_path, model_names: list[str], ctx_type: str, layer_idx: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    all_data = []
    for mn in model_names:
        key = (mn, ctx_type)
        print('Attempting key:', key, ' → types:', type(mn), type(ctx_type))
        k = captured_keys[(mn, ctx_type)][layer_idx].flatten()
        v = captured_values[(mn, ctx_type)][layer_idx].flatten()
        all_data.extend([k, v])
    global_lo = np.percentile(np.concatenate([d.numpy() for d in all_data]), 1)
    global_hi = np.percentile(np.concatenate([d.numpy() for d in all_data]), 99)
    for ri, mn in enumerate(model_names):
        k = captured_keys[(mn, ctx_type)][layer_idx].flatten().numpy()
        v = captured_values[(mn, ctx_type)][layer_idx].flatten().numpy()
        k_clip = k[(k >= global_lo) & (k <= global_hi)]
        v_clip = v[(v >= global_lo) & (v <= global_hi)]
        axes[0, ri].hist(k_clip, bins=100, color=BLUE, alpha=0.7, density=True)
        axes[1, ri].hist(v_clip, bins=100, color=TEAL, alpha=0.7, density=True)
        axes[0, ri].set_title(f"{mn} Keys Layer {layer_idx}")
        axes[1, ri].set_title(f"{mn} Values Layer {layer_idx}")
        axes[0, ri].set_xlim(global_lo, global_hi)
        axes[1, ri].set_xlim(global_lo, global_hi)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_channel_variance(instrumented_stats_dict, save_path, context_type):
    model_names = list(instrumented_stats_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(3, n_models, figsize=(7 * n_models, 10))
    # Normalise to always be 2-D
    if n_models == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    for ci, mn in enumerate(model_names):
        layer_indices = sorted(instrumented_stats_dict[mn].keys())
        if not layer_indices:
            continue
        early  = layer_indices[0]
        middle = layer_indices[len(layer_indices) // 2]
        late   = layer_indices[-1]

        for ri, lidx in enumerate([early, middle, late]):
            ax = axes[ri][ci]
            cv = instrumented_stats_dict[mn][lidx]["k_channel_var"]
            ax.bar(range(len(cv)), cv, color=BLUE, alpha=0.7, width=1.0)
            ax.set_title(f"{mn} Layer {lidx}")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Variance")

    fig.suptitle(f"Per-Channel Variance ({context_type})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_token_variance(instrumented_stats_dict, save_path, context_type):
    model_names = list(instrumented_stats_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(3, n_models, figsize=(7 * n_models, 10))
    if n_models == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    for ci, mn in enumerate(model_names):
        layer_indices = sorted(instrumented_stats_dict[mn].keys())
        if not layer_indices:
            continue
        early  = layer_indices[0]
        middle = layer_indices[len(layer_indices) // 2]
        late   = layer_indices[-1]

        for ri, lidx in enumerate([early, middle, late]):
            ax = axes[ri][ci]
            tv = np.array(instrumented_stats_dict[mn][lidx]["k_token_var"])
            for h in range(tv.shape[0]):
                ax.plot(tv[h], linewidth=1, alpha=0.6)
            ax.set_title(f"{mn} Layer {lidx}")
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Variance")

    fig.suptitle(f"Per-Token Variance ({context_type})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_layer_depth_profiles(layer_stats_df, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    metrics = ["k_abs_norm", "k_outlier_fraction", "k_delta_norm", "sv_top50_energy"]
    for ai, metric in enumerate(metrics):
        ax = axes[ai]
        for model_name, color in [("TinyLlama", BLUE), ("Qwen", TEAL)]:
            subset = layer_stats_df[layer_stats_df["model_name"] == model_name]
            for ctx, ls in [("prose", "-"), ("code", "--")]:
                sub2 = subset[subset["context_type"] == ctx].sort_values("layer_idx")
                ax.plot(sub2["layer_idx"], sub2[metric], color=color, linestyle=ls,
                        label=f"{model_name} {ctx}")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Layer Index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_kl_heatmap(kl_dict, save_path):
    model_names = list(kl_dict.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ri, mn in enumerate(model_names[:2]):
        methods = list(kl_dict[mn].keys())
        for ci, method in enumerate(methods[:3]):
            ax = axes[ri, ci]
            data = np.array(kl_dict[mn][method])
            im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn_r")
            ax.set_title(f"{mn} {method}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")
            plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eviction_pattern(token_strings, retention_probs, save_path, title):
    n_tokens = len(token_strings)
    tokens_per_row = 80
    n_rows = max(1, int(np.ceil(n_tokens / tokens_per_row)))
    fig, ax = plt.subplots(figsize=(16, max(4, n_rows * 1.5)))
    for i, (tok, prob) in enumerate(zip(token_strings, retention_probs)):
        row = i // tokens_per_row
        col = i % tokens_per_row
        if prob >= 0.8:
            color = BLUE
        elif prob >= 0.4:
            color = AMBER
        else:
            color = RED
        ax.text(col, n_rows - row - 1, tok, fontfamily="monospace", fontsize=8,
                color=color, ha="center", va="center")
    ax.set_xlim(-0.5, tokens_per_row - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_autocorrelation(autocorr_dict, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = [BLUE, TEAL, RED]
    for ri, ctx in enumerate(["prose", "code"]):
        for ci, mn in enumerate(["TinyLlama", "Qwen"]):
            ax = axes[ri, ci]
            layer_labels = sorted(autocorr_dict.get(mn, {}).get(ctx, {}).keys())
            n = len(layer_labels)
            for i, ll in enumerate(layer_labels):
                ci2 = min(int(i / max(n, 1) * (len(colors) - 1)), len(colors) - 1)
                ac = autocorr_dict[mn][ctx][ll]
                ax.plot(range(1, len(ac) + 1), ac, color=colors[ci2],
                        linewidth=1, alpha=0.7, label=f"L{ll}" if i % max(n // 3, 1) == 0 else "")
            ax.set_title(f"{mn} {ctx}")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Cosine Similarity")
            if len(layer_labels) <= 10:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scree(scree_dict, save_path):
    model_names = list(scree_dict.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(12, 5))
    if len(model_names) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ai, mn in enumerate(model_names):
        ax = axes[ai]
        layer_indices = sorted(scree_dict[mn].keys())
        for i, lidx in enumerate(layer_indices):
            cv = scree_dict[mn][lidx]
            color = cmap(i / max(len(layer_indices) - 1, 1))
            ax.plot(range(1, len(cv) + 1), cv, color=color, linewidth=1, alpha=0.8)
        ax.axhline(y=0.9, color=RED, linestyle="--", linewidth=1)
        ax.set_title(mn)
        ax.set_xlabel("Component Index")
        ax.set_ylabel("Cumulative Variance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effective_rank(eff_rank_df, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    layers = sorted(eff_rank_df["layer_idx"].unique())
    models = sorted(eff_rank_df["model_name"].unique())
    bar_width = 0.35
    x = np.arange(len(layers))
    for mi, (mn, color) in enumerate(zip(models, [BLUE, TEAL])):
        offset = bar_width * (mi - 0.5)
        for ctx_i, ctx in enumerate(["prose", "code"]):
            sub2 = eff_rank_df[(eff_rank_df["model_name"] == mn) & (eff_rank_df["context_type"] == ctx)]
            sub2 = sub2.set_index("layer_idx")
            vals = [sub2.loc[l, "effective_rank_90"] if l in sub2.index else 0 for l in layers]
            ax.bar(x + offset, vals, bar_width / 2, color=color,
                   hatch="" if ctx == "prose" else "//",
                   label=f"{mn} {ctx}" if ctx_i == 0 else "", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ll) for ll in layers])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank 90%")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_codec_latency_scatter(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.scatter(row["compression_ratio"], row["decode_ms"], color=BLUE, s=60)
        ax.annotate(row["codec_name"], (row["compression_ratio"], row["decode_ms"]),
                    fontsize=7, ha="center", va="bottom", textcoords="offset points", xytext=(0, 6))
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Decode Time (ms)")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_codec_mse_bar(df, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sorted = df.sort_values("mse")
    x = np.arange(len(df_sorted))
    w = 0.35
    ax.bar(x - w / 2, df_sorted["mse"], w, label="MSE", color=BLUE)
    ax.bar(x + w / 2, df_sorted["attn_logit_rel_err"], w, label="Attn Logit Rel Err", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["codec_name"], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylabel("Error")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attn_logit_vs_mse(df, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        ax.scatter(row["mse"], row["attn_logit_rel_err"], color=BLUE, s=60)
        ax.annotate(row["codec_name"], (row["mse"], row["attn_logit_rel_err"]),
                    fontsize=7, ha="center", va="bottom", textcoords="offset points", xytext=(0, 6))
    ax.set_xlabel("MSE")
    ax.set_ylabel("Attention Logit Relative Error")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ppl_vs_ratio(df, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline = df[df["codec_name"] == "baseline"]
    if len(baseline) > 0:
        bl_ppl = baseline.iloc[0]["perplexity"]
        ax.axhline(y=bl_ppl, color=GRAY, linestyle="--", label=f"Baseline ({bl_ppl:.2f})")
    codec_colors = {
        "delta_fp16": BLUE, "delta_int8": BLUE, "delta_int4": BLUE,
        "quant_int8": TEAL, "quant_int4": TEAL,
        "kivi_int8": AMBER, "kivi_int4": AMBER, "kivi_int2": AMBER,
        "svd_r0.8": PURPLE, "svd_r0.5": PURPLE,
    }
    for cn in sorted(df["codec_name"].unique()):
        sub = df[df["codec_name"] == cn].sort_values("effective_compression_ratio")
        color = codec_colors.get(cn, GRAY)
        ax.plot(sub["effective_compression_ratio"], sub["perplexity"],
                marker="o", label=cn, color=color, linewidth=1.5)
    ax.set_xlabel("Effective Compression Ratio")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ppl_vs_bits(df, save_path):
    families = {"Quantization": ["quant_int8", "quant_int4"],
                "KIVI": ["kivi_int8", "kivi_int4", "kivi_int2"],
                "Delta": ["delta_int8", "delta_int4"]}
    family_colors = {"Quantization": TEAL, "KIVI": AMBER, "Delta": BLUE}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ri, ctx in enumerate(["prose", "code"]):
        for ci, mn in enumerate(["TinyLlama", "Qwen"]):
            ax = axes[ri, ci]
            sub = df[(df["context_type"] == ctx) & (df["model_name"] == mn)]
            for family, codes in families.items():
                fam_data = sub[sub["codec_name"].isin(codes)]
                if len(fam_data) == 0:
                    continue
                pts = []
                for _, r in fam_data.iterrows():
                    cn = r["codec_name"]
                    b = None
                    if "int8" in cn:
                        b = 8
                    elif "int4" in cn:
                        b = 4
                    elif "int2" in cn:
                        b = 2
                    elif "fp16" in cn:
                        b = 16
                    if b is not None:
                        pts.append((b, r["perplexity"]))
                pts.sort()
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=family, color=family_colors[family], linewidth=2)
            ax.set_title(f"{mn} {ctx}")
            ax.set_xlabel("Bits")
            ax.set_ylabel("Perplexity (log scale)")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eviction_ppl_sweep(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"knorm": BLUE, "snapkv": TEAL}
    for pn in sorted(df["press_name"].unique()):
        sub = df[df["press_name"] == pn].sort_values("compression_ratio")
        ax.plot(sub["compression_ratio"], sub["perplexity"],
                marker="o", label=pn, color=colors.get(pn, GRAY), linewidth=2)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_crossover_comparison(df, save_path):
    model_names = sorted(df["model_name"].unique())
    fig, axes = plt.subplots(len(model_names), 1, figsize=(14, 8), squeeze=False)
    axis_colors = {"precision": BLUE, "eviction": TEAL, "dimension": PURPLE}
    for ri, mn in enumerate(model_names):
        ax = axes[ri, 0]
        sub = df[df["model_name"] == mn]
        methods = sorted(sub["method"].unique())
        x = np.arange(len(methods))
        w = 0.35
        for ti, target in enumerate(["2x", "4x"]):
            sub2 = sub[sub["target_reduction"] == target].set_index("method")
            vals = []
            cols = []
            for m in methods:
                if m in sub2.index:
                    vals.append(sub2.loc[m, "delta_ppl"])
                    hatch = "" if target == "2x" else "//"
                    cols.append((axis_colors.get(sub2.loc[m, "compression_axis"], GRAY), hatch))
                else:
                    vals.append(0)
                    hatch = "" if target == "2x" else "//"
                    cols.append((GRAY, hatch))
            offset = w * (ti - 0.5)
            for mi2, m in enumerate(methods):
                ax.bar(x[mi2] + offset, vals[mi2], w, color=cols[mi2][0],
                       hatch=cols[mi2][1], edgecolor="white", linewidth=0.5,
                       alpha=0.85, label=target if mi2 == 0 else "")
        ax.set_title(mn)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Delta Perplexity (log scale)")
        ax.set_yscale("symlog")
        ax.axhline(y=0, color=GRAY, linestyle="-", linewidth=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


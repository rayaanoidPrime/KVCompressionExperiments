import sys
import traceback
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from transformers import DynamicCache
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from harness import measure_perplexity
from utils import tokenize, MODEL_ID, SUPPORTED_CTX_TYPES, load_model
from instrumented_press import InstrumentedPress
from context_samples import PROSE_CONTEXT, CODE_CONTEXT
import plots

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
OUTPUT_DIR      = CURRENT_DIR / "experiment-outputs"
CONTEXT_MAP = {"prose": PROSE_CONTEXT, "code": CODE_CONTEXT}


# ===========================================================================
# Logging
# ===========================================================================

def configure_logging(level: str, log_file: Path | None = None) -> None:
    log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=log_fmt, handlers=handlers)


# ===========================================================================
# EXPERIMENT FAMILY 1 – KV Cache / Attention Exploration
# ===========================================================================

# ---------------------------------------------------------------------------
# 1a. Data collection
# ---------------------------------------------------------------------------

def collect_instrumented_data(
    model_names: list[str],
    contexts: list[str],
    max_length: int = 1024,
) -> dict:
    """
    Run InstrumentedPress for every (model, context) pair.

    Returns
    -------
    results : dict
        {(model_name, ctx_type): press_instance}
        Callers can use these live objects for plotting without re-loading.
    """
    log = logging.getLogger("collect_instrumented")
    log.info("Running InstrumentedPress for %s × %s", model_names, contexts)

    results: dict[tuple[str, str], InstrumentedPress] = {}

    for model_name in model_names:
        if model_name not in MODEL_ID:
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported: {list(MODEL_ID.keys())}"
            )
        for ctx_type in contexts:
            if ctx_type not in SUPPORTED_CTX_TYPES:
                raise ValueError(
                    f"Unsupported context type '{ctx_type}'. Supported: {SUPPORTED_CTX_TYPES}"
                )
            try:
                out_dir = INST_DIR / model_name / ctx_type
                out_dir.mkdir(parents=True, exist_ok=True)

                model, tokenizer = load_model(model_name)
                context = CONTEXT_MAP[ctx_type]
                input_ids = tokenize(tokenizer, context, max_length=max_length)
                seq_len = input_ids.shape[1]

                press = InstrumentedPress()
                with torch.no_grad():
                    with press(model):
                        cache = DynamicCache()
                        model(
                            input_ids,
                            past_key_values=cache,
                            cache_position=torch.arange(seq_len, device="cpu"),
                            use_cache=True,
                        )

                press.save(str(out_dir))
                results[(model_name, ctx_type)] = press
                log.info("  Saved: %s/%s", model_name, ctx_type)
                del model

            except Exception:
                log.error(
                    "  FAILED: %s/%s\n%s", model_name, ctx_type, traceback.format_exc()
                )

    return results


# ---------------------------------------------------------------------------
# 1b. Load previously saved instrumented data (avoids re-running the model)
# ---------------------------------------------------------------------------

def load_instrumented_data(
    model_names: list[str],
    contexts: list[str],
) -> dict[tuple[str, str], InstrumentedPress]:
    """
    Re-hydrate InstrumentedPress instances from disk (via press.load()).
    """
    log = logging.getLogger("load_instrumented")
    results: dict[tuple[str, str], InstrumentedPress] = {}
    for model_name in model_names:
        for ctx_type in contexts:
            path = INST_DIR / model_name / ctx_type
            if not path.exists():
                log.warning("No saved data at %s – skipping.", path)
                continue
            press = InstrumentedPress()
            press.load(str(path))
            results[(model_name, ctx_type)] = press
            log.info("Loaded: %s/%s", model_name, ctx_type)
    return results


# ---------------------------------------------------------------------------
# 1c.  Helper: unpack InstrumentedPress into plot-ready dicts
# ---------------------------------------------------------------------------

def _build_plot_inputs(
    instrumented: dict[tuple[str, str], InstrumentedPress],
    model_names: list[str],
    contexts: list[str],
    layer_idx: int,
) -> dict:
    captured_keys_dict: dict = {}   # {(model, ctx): full list[Tensor]}
    captured_vals_dict: dict = {}   # {(model, ctx): full list[Tensor]}
    captured_keys: dict      = {}   # {model: full list[Tensor]}  (first ctx)
    captured_values: dict    = {}   # {model: full list[Tensor]}  (first ctx)
    instrumented_stats: dict = {}   # {model: {lidx: {k_channel_var, k_token_var}}}
    layer_stats_rows: list   = []

    for model_name in model_names:
        for ctx_type in contexts:
            key = (model_name, ctx_type)
            press = instrumented.get(key)
            if press is None:
                continue

            # Store full lists — plotters do their own indexing
            captured_keys_dict[key] = press.captured_keys
            captured_vals_dict[key] = press.captured_values

            for lidx, stats in press.layer_stats.items():
                layer_stats_rows.append(dict(
                    model_name         = model_name,
                    context_type       = ctx_type,
                    layer_idx          = lidx,
                    k_abs_norm         = stats["k_abs_norm"],
                    k_outlier_fraction = stats["k_outlier_fraction"],
                    k_delta_norm       = stats["k_delta_norm"],
                    sv_top50_energy    = stats["sv_top50_energy"],
                ))

        # Single-context structures use first available context
        first_ctx = next(
            (c for c in contexts if (model_name, c) in instrumented), None
        )
        if first_ctx is None:
            continue
        press0 = instrumented[(model_name, first_ctx)]
        captured_keys[model_name]   = press0.captured_keys
        captured_values[model_name] = press0.captured_values

        # Pull directly from press.layer_stats — no recomputation
        instrumented_stats[model_name] = {
            lidx: {
                "k_channel_var": stats["k_channel_var"],
                "k_token_var":   stats["k_token_var"],
            }
            for lidx, stats in press0.layer_stats.items()
        }

    return dict(
        captured_keys_dict = captured_keys_dict,
        captured_vals_dict = captured_vals_dict,
        captured_keys      = captured_keys,
        captured_values    = captured_values,
        instrumented_stats = instrumented_stats,
        layer_stats_df     = pd.DataFrame(layer_stats_rows),
    )


# ---------------------------------------------------------------------------
# 1d. Plot all KV-cache / attention figures
# ---------------------------------------------------------------------------

def plot_kv_attention_figures(
    instrumented: dict[tuple[str, str], InstrumentedPress],
    model_names: list[str],
    contexts: list[str],
    layer_idx: int = 0,
    tokenizer=None,
) -> None:
    log = logging.getLogger("plot_kv_attention")
    save_dir = PLOT_DIR / "kv_attention"
    save_dir.mkdir(parents=True, exist_ok=True)

    inputs = _build_plot_inputs(instrumented, model_names, contexts, layer_idx)

    ckeys  = inputs["captured_keys_dict"]   # {(model, ctx): list[Tensor]}
    cvals  = inputs["captured_vals_dict"]   # {(model, ctx): list[Tensor]}
    istats = inputs["instrumented_stats"]   # {model: {lidx: {...}}}
    ls_df  = inputs["layer_stats_df"]

    # ── 1. Activation magnitude heatmaps ────────────────────────────────────
    for ctx_type in contexts:
        path = save_dir / f"magnitude_heatmap_{ctx_type}.png"
        try:
            plots.plot_magnitude_heatmap(
                captured_keys_dict = ckeys,
                tokenizer          = tokenizer,
                save_path          = str(path),
                layer_idx          = layer_idx,
                context_type_label = ctx_type,
            )
            log.info("Saved: %s", path)
        except Exception:
            log.error("magnitude_heatmap %s failed\n%s", ctx_type, traceback.format_exc())

    # ── 2. Keys vs Values distributions ─────────────────────────────────────
    for ctx_type in contexts:
        ck_ctx = {mn: ckeys[(mn, ctx_type)] for mn in model_names if (mn, ctx_type) in ckeys}
        cv_ctx = {mn: cvals[(mn, ctx_type)] for mn in model_names if (mn, ctx_type) in cvals}
        if not ck_ctx:
            continue
        path = save_dir / f"kv_distributions_{ctx_type}.png"
        try:
            plots.plot_kv_distributions(
                captured_keys   = ck_ctx,
                captured_values = cv_ctx,
                save_path       = str(path),
                model_names     = list(ck_ctx.keys()),
                layer_idx       = layer_idx,
            )
            log.info("Saved: %s", path)
        except Exception:
            log.error("kv_distributions %s failed\n%s", ctx_type, traceback.format_exc())

    # ── 3. Across-channel variance ───────────────────────────────────────────
    for ctx_type in contexts:
        # Filter instrumented_stats to models present for this context
        istats_ctx = {
            mn: istats[mn]
            for mn in model_names
            if mn in istats and (mn, ctx_type) in instrumented
        }
        if not istats_ctx:
            continue
        path = save_dir / f"channel_variance_{ctx_type}.png"
        try:
            plots.plot_channel_variance(
                instrumented_stats_dict = istats_ctx,
                save_path               = str(path),
                context_type            = ctx_type,
            )
            log.info("Saved: %s", path)
        except Exception:
            log.error("channel_variance %s failed\n%s", ctx_type, traceback.format_exc())

    # ── 4. Across-token variance ─────────────────────────────────────────────
    for ctx_type in contexts:
        istats_ctx = {
            mn: istats[mn]
            for mn in model_names
            if mn in istats and (mn, ctx_type) in instrumented
        }
        if not istats_ctx:
            continue
        path = save_dir / f"token_variance_{ctx_type}.png"
        try:
            plots.plot_token_variance(
                instrumented_stats_dict = istats_ctx,
                save_path               = str(path),
                context_type            = ctx_type,
            )
            log.info("Saved: %s", path)
        except Exception:
            log.error("token_variance %s failed\n%s", ctx_type, traceback.format_exc())

    # ── 5. Layer depth profiles ──────────────────────────────────────────────
    if not ls_df.empty:
        path = save_dir / "layer_depth_profiles.png"
        try:
            plots.plot_layer_depth_profiles(
                layer_stats_df = ls_df,
                save_path      = str(path),
            )
            log.info("Saved: %s", path)
        except Exception:
            log.error("layer_depth_profiles failed\n%s", traceback.format_exc())

    log.info("All KV/attention plots complete → %s", save_dir)


# ---------------------------------------------------------------------------
# 1e. KV cache memory growth curve
# ---------------------------------------------------------------------------

def experiment_kv_growth(
    model_names: list[str],
    seq_lengths: list[int] | None = None,
    n_heads: int = 32,
    head_dim: int = 128,
    dtype_bytes: int = 2,   # fp16
) -> None:
    """
    Synthetic model of KV cache growth:
        kv_cache_mb = 2 * n_layers * n_heads * seq_len * head_dim * dtype_bytes / 1e6
    Plots with plots.plot_kv_growth.
    """
    log = logging.getLogger("kv_growth")
    save_dir = PLOT_DIR / "kv_attention"
    save_dir.mkdir(parents=True, exist_ok=True)

    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Approximate n_layers per model
    n_layers_map = {"TinyLlama": 22, "Qwen": 32}

    rows = []
    for mn in model_names:
        n_layers = n_layers_map.get(mn, 32)
        for sl in seq_lengths:
            mb = 2 * n_layers * n_heads * sl * head_dim * dtype_bytes / 1e6
            rows.append({"model_name": mn, "seq_len": sl, "kv_cache_mb": mb})

    df = pd.DataFrame(rows)
    path = save_dir / "kv_cache_growth.png"
    try:
        plots.plot_kv_growth(df, str(path))
        log.info("Saved: %s", path)
    except Exception:
        log.error("kv_growth failed\n%s", traceback.format_exc())


# ===========================================================================
# EXPERIMENT FAMILY 2 – Perplexity vs Context Length
# ===========================================================================

def run_ppl_baseline(
    model_names: list[str],
    contexts: list[str],
    seq_lengths: list[int] = None,
    output_dir: Path = OUTPUT_DIR / "perplexity",
) -> pd.DataFrame:
    """
    For every (model, context, seq_length) combination, measure perplexity
    with no compression.  Saves one CSV and one PNG per model.
    """
    log = logging.getLogger("ppl_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if seq_lengths is None:
        seq_lengths = [60, 80, 100, 120, 160, 200, 256, 320, 512]

    for model_name in model_names:
        if model_name not in MODEL_ID:
            log.error("Unknown model '%s' – skipping.", model_name)
            continue

        log.info("Loading model: %s", model_name)
        model, tokenizer = load_model(model_name)

        for ctx_type in contexts:
            context_text = CONTEXT_MAP.get(ctx_type, PROSE_CONTEXT)

            for seq_len in seq_lengths:
                log.info("  PPL | %s | ctx=%s | seq_len=%d", model_name, ctx_type, seq_len)
                result = measure_perplexity(
                    model      = model,
                    tokenizer  = tokenizer,
                    context    = context_text,
                    max_length = seq_len,
                )
                rows.append({
                    "model_name":   model_name,
                    "context_type": ctx_type,
                    "seq_len":      seq_len,
                    **result,
                })

        del model   # free memory before loading the next model

    df = pd.DataFrame(rows)

    # ── Save combined CSV ──────────────────────────────────────────────────
    csv_path = output_dir / "ppl_baseline.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV → %s", csv_path)

    # ── One plot per model ─────────────────────────────────────────────────
    for model_name in df["model_name"].unique():
        _plot_model(df[df["model_name"] == model_name], model_name, output_dir)

    return df


def _plot_model(df: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    """
    One figure per model.
    X-axis : sequence length
    Y-axis : perplexity
    One line per context type (prose / code / …)
    """
    log = logging.getLogger("plot_ppl")
    fig, ax = plt.subplots(figsize=(7, 4))

    for ctx_type, grp in df.groupby("context_type"):
        grp_sorted = grp.sort_values("seq_len")
        ax.plot(
            grp_sorted["seq_len"],
            grp_sorted["perplexity"],
            marker="o",
            label=ctx_type,
        )

    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Baseline PPL vs Sequence Length — {model_name}")
    ax.legend(title="Context")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = output_dir / f"ppl_baseline_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved plot → %s", path)

# ===========================================================================
# CLI
# ===========================================================================

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular experiment runner for KV-cache and perplexity studies."
    )
    parser.add_argument(
        "--models", nargs="+", help="Model identifiers present in utils.MODEL_ID.", default=["Qwen", "TinyLlama"]
    )
    parser.add_argument(
        "--contexts", nargs="+", choices=["prose", "code"], default=["prose", "code"],
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["instrumented", "kv_growth", "ppl_vs_ctx", "all"],
        default=["all"],
        help="Which experiment families to run.",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=0,
        help="Layer index used for per-layer plots.",
    )
    parser.add_argument(
        "--seq-lengths", nargs="+", type=int, default=None,
        help="Sequence lengths for ppl_vs_ctx and kv_growth experiments.",
    )
    parser.add_argument("--output-dir",  type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    configure_logging(args.log_level, args.log_file)
    log = logging.getLogger("run_experiments")

    # Allow --output-dir to override the module-level constants
    OUTPUT_DIR = args.output_dir
    INST_DIR   = OUTPUT_DIR / "instrumented"
    PLOT_DIR   = OUTPUT_DIR / "plots"
    PPL_DIR    = OUTPUT_DIR / "perplexity"

    run_all          = "all" in args.experiments
    run_instrumented = run_all or "instrumented" in args.experiments
    run_kv_growth    = run_all or "kv_growth"    in args.experiments
    run_ppl          = run_all or "ppl_vs_ctx"   in args.experiments

    log.info("Models: %s | Contexts: %s", args.models, args.contexts)

    # ── Experiment Family 1: KV cache / attention exploration ────────────────
    if run_instrumented or run_kv_growth:
        if run_instrumented:
            instrumented = collect_instrumented_data(
                model_names = args.models,
                contexts    = args.contexts,
                max_length  = args.max_length,
            )
        else:
            # Load from disk if we only want plots
            instrumented = load_instrumented_data(args.models, args.contexts)

        plot_kv_attention_figures(
            instrumented = instrumented,
            model_names  = args.models,
            contexts     = args.contexts,
            layer_idx    = args.layer_idx,
        )

    if run_kv_growth:
        experiment_kv_growth(
            model_names = args.models,
            seq_lengths = args.seq_lengths,
        )

    # ── Experiment Family 2: Perplexity vs context length ───────────────────
    if run_ppl:
        run_ppl_baseline(
            model_names = args.models,
            contexts    = args.contexts,
            seq_lengths = args.seq_lengths,
        )

    log.info("All experiments complete. Outputs in: %s", OUTPUT_DIR)
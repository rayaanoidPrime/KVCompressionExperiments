#!/usr/bin/env bash
# CPU Baseline: PPL vs context length, decode latency vs context length
#
# Usage:
#   ./run_baseline.sh [model_name] [bin_dir] [corpus_path] [output_dir]
#
#   model_name   – name of the model (default: Qwen3-1.7B)
#   bin_dir      – path to the built llama.cpp binaries (default: $HOME/a1264472/work/llama.cpp/build-cpu/bin)
#   corpus_path  – optional path to a test corpus; if omitted the script will download wikitext-2
#   output_dir   – directory where results and logs will be written (default: $HOME/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results)

set -euo pipefail

# ── User-provided arguments ────────────────────────────────────────
MODEL="${1:-Qwen3-1.7B}"
BIN_DIR="${2:-$HOME/a1264472/work/llama.cpp/build-cpu/bin}"
CORPUS="${3:-}"
OUT_DIR="${4:-$HOME/a1264472/work/KVCompressionExperiments/llamacpp_baseline_results}"

# ── Paths to the different GGUF formats ───────────────────────────
BF16_PATH="$HOME/a1264472/work/${MODEL}-BF16.gguf"


# ── Experiment parameters ─────────────────────────────────────────
KV_TYPES=("f16" "q8_0" "q4_0")
THREADS=$(nproc)

mkdir -p "$OUT_DIR/logs"

LATENCY_CSV="$OUT_DIR/latency_vs_ctx.csv"
PPL_CSV="$OUT_DIR/ppl_vs_ctx.csv"

# ── FIX 2: Verify required binaries exist ─────────────────────────
for bin in llama-quantize llama-bench llama-perplexity; do
    if [[ ! -x "$BIN_DIR/$bin" ]]; then
        echo "[ERROR] Required binary not found or not executable: $BIN_DIR/$bin"
        exit 1
    fi
done

# ── Helper: resolve the corpus (download wikitext-2 if needed) ────
get_corpus() {
    if [[ -n "$CORPUS" && -f "$CORPUS" ]]; then
        echo "$CORPUS"
        return
    fi

    local corpus_file="$OUT_DIR/wikitext-2-test.txt"
    if [[ -s "$corpus_file" ]]; then
        echo "$corpus_file"
        return
    fi

    local zip_file="$OUT_DIR/wikitext-2-raw-v1.zip"
    local url="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

    echo "[INFO] Downloading wikitext-2 test set…" >&2
    curl -L "$url" -o "$zip_file" || { echo "[ERROR] Download failed"; exit 1; }

    echo "[INFO] Extracting test split…" >&2
    unzip -p "$zip_file" "wikitext-2-raw/wiki.test.raw" > "$corpus_file" \
        || { echo "[ERROR] Unzip failed"; exit 1; }

    rm -f -- "$zip_file"
    echo "$corpus_file"
}



# ── Latency sweep (decode latency vs. context length) ─────────────
echo "=== Latency sweep ==="

# Write header once from the first run
HEADER_WRITTEN=0

for kv in "${KV_TYPES[@]}"; do
    MODEL_PATH=$(get_model_path "$kv")
    if [[ -z "$MODEL_PATH" ]]; then
        echo "[WARN] Unknown KV type '$kv' — skipping."
        continue
    fi
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "[WARN] Model file not found for KV=$kv ($MODEL_PATH) — skipping."
        continue
    fi

    log="$OUT_DIR/logs/bench_kv${kv}.log"

    "$BIN_DIR/llama-bench" \
        -m "$MODEL_PATH" \
        -ngl 0 \
        -t "$THREADS" \
        -p 0 \
        -n 1 \
        -d 60,80,100,120,160,200,256 \
        -ctk "$kv" \
        -ctv "$kv" \
        -fa 1 \
        -o csv > "$log" 2>&1 \
        || { echo "[ERROR] Benchmark failed for KV=$kv — check: $log"; continue; }

    if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
        # Write header + data rows
        cat "$log" >> "$LATENCY_CSV"
        HEADER_WRITTEN=1
    else
        # Skip the header line (line 1), append only data rows
        tail -n +2 "$log" >> "$LATENCY_CSV"
    fi

    echo "[INFO] Latency benchmark for KV=$kv completed — log: $log"
done


# ── Perplexity sweep (PPL vs. context length) ─────────────────────
RESOLVED_CORPUS=$(get_corpus)
echo ""
echo "=== Perplexity sweep (corpus: $RESOLVED_CORPUS) ==="
echo "ctx,kv_type,ppl" > "$PPL_CSV"

for kv in "${KV_TYPES[@]}"; do
    for ctx in 60 80 100 120 160 200 256 320 512; do
        echo -n " ppl ctx=$ctx  kv=$kv ... "
        log="$OUT_DIR/logs/ppl_ctx${ctx}_kv${kv}.log"

        # Disable set -e locally so a single PPL failure doesn't abort the whole sweep
        "$BIN_DIR/llama-perplexity" \
            -m "$BF16_PATH" \
            -t "$THREADS" \
            -c $ctx \
            -ctk "$kv" \
            -ctv "$kv" \
            -fa 1 \
            -f "$RESOLVED_CORPUS" > "$log" 2>&1 \
            || { echo "[ERROR] Perplexity failed for ctx=$ctx KV=$kv — check: $log"
                    echo "$ctx,$kv,ERROR" >> "$PPL_CSV"
                    continue; }

        # FIX 7: llama-perplexity prints "Final estimate: PPL = X.XXXX +/- Y.YYYY"
        ppl=$(grep -oP 'Final estimate: PPL\s*=\s*\K[0-9.]+' "$log" | tail -1 || true)
        if [[ -z "$ppl" ]]; then
            # Fallback: some builds print "Perplexity(context) = X.XX" per chunk; take last
            ppl=$(grep -oP 'Perplexity\(.*?\)\s*=\s*\K[0-9.]+' "$log" | tail -1 || echo "N/A")
        fi

        echo "$ctx,$kv,$ppl" >> "$PPL_CSV"
        echo "PPL = $ppl"
    
    done
done

# ── Final summary ─────────────────────────────────────────────────
echo ""
echo "=== All done! Results saved in $OUT_DIR ==="
echo "  Latency CSV  : $LATENCY_CSV"
echo "  Perplexity CSV : $PPL_CSV"
echo "  Logs folder  : $OUT_DIR/logs"
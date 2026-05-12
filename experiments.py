import sys

from transformers import DynamicCache
import torch 
import traceback
from pathlib import Path
import logging 
import argparse

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from harness import tokenize, measure_perplexity
from InstrumentedPress import InstrumentedPress
from utils import MODEL_ID, SUPPORTED_CTX_TYPES, load_model
from context_samples import PROSE_CONTEXT, CODE_CONTEXT

OUTPUT_DIR = CURRENT_DIR / "ExperimentOutputs"
INST_DIR = OUTPUT_DIR / "instrumented"

# TODO: add the plots to each experiment function

def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Configure root logger."""
    log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf‑8"))
    logging.basicConfig(level=level.upper(), format=log_fmt, handlers=handlers)


def collect_instrumented_data(model_names: list, contexts: list, max_length: int = 1024) -> None:
    print("--- Running InstrumentedPress for all models/contexts ---")
    for model_name in model_names:
        if model_name not in MODEL_ID:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_ID.keys())}")
        for ctx_type in contexts:
            if ctx_type not in SUPPORTED_CTX_TYPES:
                raise ValueError(f"Unsupported context type: {ctx_type}. Supported types: ['prose', 'code']")
            try:
                out_dir = INST_DIR / model_name / ctx_type
                out_dir.mkdir(parents=True, exist_ok=True)
                model, tokenizer = load_model(model_name)
                context = PROSE_CONTEXT if ctx_type == "prose" else CODE_CONTEXT
                input_ids = tokenize(tokenizer, context, max_length=max_length)
                seq_len = input_ids.shape[1]
                press = InstrumentedPress()
                with torch.no_grad():
                    with press(model):
                        cache = DynamicCache()
                        model(input_ids, past_key_values=cache,
                            cache_position=torch.arange(seq_len, device="cpu"), use_cache=True)
                press.save(str(out_dir))
                del model
                print(f"  INSTRUMENTED: {model_name}/{ctx_type} saved")
            except Exception as e:
                print(f"  INSTRUMENTED FAILED: {model_name}/{ctx_type}: {e}")
                traceback.print_exc()

def parse_cli() -> argparse.Namespace:
    """Define and parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments across models and contexts."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model identifiers (must be present in SUPPORTED_MODELS).",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=["prose", "code"],
        default=["prose", "code"],
        help="Which context(s) to evaluate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum token length for the input sequence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base directory for all experiment artefacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of console logging.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to a file where logs will be persisted.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    configure_logging(args.log_level, args.log_file)

    logger = logging.getLogger("run_experiments")
    logger.info("Starting experiment run")
    logger.debug("CLI args: %s", args)

    collect_instrumented_data(model_names=args.models, contexts=args.contexts, max_length=args.max_length)
    print("Experiment completed. Instrumented data saved to:", INST_DIR)
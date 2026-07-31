"""`hypnose-eeg-preprocess train` — train a somnotate model from labelled MAT files.

The model name is the primary argument: it names the output directory under
`derivatives/somnotate_training/<model_name>/`, so it is also the handle
`score --model <model_name>` later resolves.

Existing models are never overwritten. If the target directory already holds a
model, the command stops and asks for a different name (or `--force`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import DEFAULT_SAMPLING_RATE_HZ, DEFAULT_SLEEP_STAGE_RESOLUTION_S
from ._args import add_repo_root_argument
from ._common import (
    MODEL_FILENAME,
    get_model_dir,
    get_training_input_root,
    list_models,
    model_exists,
)


def resolve_input_dir(input_dir: str | Path | None, repo_root: Path | None = None) -> Path:
    """Directory holding the labelled .mat files to train on.

    Resolved against `<eeg root>/somnotate_training/` — the *input* tree — not
    `derivatives/somnotate_training/`, which is where trained models are written.
    The two share a name but are different directories.

    Without `--input-dir` the input root itself is used, which is the layout that
    existed before this flag. With it, training reads from
    `<eeg root>/somnotate_training/<input_dir>/`, so several labelled sets can live
    side by side and be trained separately. An absolute path is honoured as given.
    """
    input_root = get_training_input_root(repo_root)
    if input_dir is None:
        return input_root
    candidate = Path(input_dir)
    if candidate.is_absolute():
        return candidate
    return input_root / candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypnose-eeg-preprocess train",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  hypnose-eeg-preprocess train my-model\n"
            "  hypnose-eeg-preprocess train my-model --input-dir cohort-a\n"
            "  hypnose-eeg-preprocess train my-model --show-plots\n"
        ),
    )
    parser.add_argument(
        "model_name",
        help="Name of the model; also the output dir under derivatives/somnotate_training/.",
    )
    parser.add_argument(
        "--input-dir", metavar="DIR",
        help="Labelled .mat files to train on, relative to "
             "<eeg root>/somnotate_training/ (or an absolute path). "
             "Default: that directory itself.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if a model of this name already exists (overwrites it).",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE_HZ, metavar="HZ",
        help=f"Sampling rate of the recordings (default: {DEFAULT_SAMPLING_RATE_HZ}).",
    )
    parser.add_argument(
        "--sleep-stage-resolution", type=int, default=DEFAULT_SLEEP_STAGE_RESOLUTION_S,
        metavar="SECONDS",
        help=f"Scoring epoch of the manual labels (default: {DEFAULT_SLEEP_STAGE_RESOLUTION_S}).",
    )
    parser.add_argument(
        "--show-plots", action="store_true",
        help="Display hold-one-out evaluation plots instead of only writing them.",
    )
    add_repo_root_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root

    if not args.model_name.strip():
        print("error: model_name must not be empty", file=sys.stderr)
        return 2

    # Never silently clobber a trained model: it is the artefact every later
    # scoring run resolves against.
    if model_exists(args.model_name, repo_root) and not args.force:
        model_dir = get_model_dir(args.model_name, repo_root)
        print(
            f"error: a model named {args.model_name!r} already exists at\n"
            f"  {model_dir / MODEL_FILENAME}\n\n"
            "Choose a different name, or pass --force to overwrite it.",
            file=sys.stderr,
        )
        existing = list_models(repo_root)
        if existing:
            print(f"\nexisting models: {', '.join(existing)}", file=sys.stderr)
        return 1

    training_mat_dir = resolve_input_dir(args.input_dir, repo_root)
    if not training_mat_dir.exists():
        print(f"error: training input directory not found: {training_mat_dir}", file=sys.stderr)
        return 1

    mat_files = sorted(p for p in training_mat_dir.glob("*.mat") if not p.name.startswith("._"))
    if not mat_files:
        print(f"error: no .mat files in {training_mat_dir}", file=sys.stderr)
        return 1

    print(f"Training model {args.model_name!r}")
    print(f"  input : {training_mat_dir} ({len(mat_files)} .mat file(s))")
    print(f"  output: {get_model_dir(args.model_name, repo_root)}")

    # Imported here so `--help` and the guards above stay fast: run_training pulls
    # in pomegranate and the somnotate stack.
    from ..training import run_training

    model_path = run_training(
        model_name=args.model_name,
        training_mat_dir=training_mat_dir,
        repo_root=repo_root,
        sampling_rate_hz=args.sampling_rate,
        sleep_stage_resolution_s=args.sleep_stage_resolution,
        show_plots=args.show_plots,
    )
    print(f"\nSaved model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

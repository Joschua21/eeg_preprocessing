"""Model resolution and session expansion shared by the CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..io.paths import get_derivatives_root, get_eeg_root

TRAINING_SUBDIR = "somnotate_training"
MODEL_FILENAME = "model.pickle"


# `somnotate_training` names two different directories, which is easy to conflate:
#
#   <eeg root>/somnotate_training/               labelled .mat files to train FROM
#   <eeg root>/derivatives/somnotate_training/   trained models written TO
#
# They are kept as separate functions with explicit names for that reason.

def get_training_input_root(repo_root: Path | None = None) -> Path:
    """Directory holding the labelled `.mat` files that training reads."""
    return get_eeg_root(repo_root) / TRAINING_SUBDIR


def get_model_root(repo_root: Path | None = None) -> Path:
    """Directory under derivatives where trained models are written."""
    return get_derivatives_root(repo_root) / TRAINING_SUBDIR


def get_model_dir(model_name: str, repo_root: Path | None = None) -> Path:
    """Output directory for `model_name`."""
    return get_model_root(repo_root) / model_name


def model_exists(model_name: str, repo_root: Path | None = None) -> bool:
    return (get_model_dir(model_name, repo_root) / MODEL_FILENAME).exists()


def list_models(repo_root: Path | None = None) -> list[str]:
    """Names of every trained model under the training root."""
    training_root = get_model_root(repo_root)
    if not training_root.exists():
        return []
    return sorted(
        p.name for p in training_root.iterdir()
        if p.is_dir() and (p / MODEL_FILENAME).exists()
    )


def resolve_model_path(model: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve `--model` to a model.pickle.

    Accepts, in order: a direct path to a .pickle, a directory containing
    model.pickle, or a model *name* under derivatives/somnotate_training/. The
    name form is the usual one, so `train my-model` and `score --model my-model`
    line up.
    """
    candidate = Path(model)

    if candidate.suffix == ".pickle" and candidate.is_file():
        return candidate
    if candidate.is_dir() and (candidate / MODEL_FILENAME).is_file():
        return candidate / MODEL_FILENAME

    by_name = get_model_dir(str(model), repo_root) / MODEL_FILENAME
    if by_name.is_file():
        return by_name

    known = list_models(repo_root)
    hint = f"\nAvailable models: {', '.join(known)}" if known else (
        f"\nNo trained models found under {get_model_root(repo_root)}."
    )
    raise argparse.ArgumentTypeError(f"Could not resolve model {model!r}.{hint}")


def resolve_sessions(
    subjects: list[int],
    dates: list[str] | None,
    date_range: tuple[str, str] | None,
    repo_root: Path,
) -> list[tuple[int, str]]:
    """Expand a selector into the concrete (subject, date) sessions on disk.

    Goes through `find_recordings` so a `--date-range` (which the plotting
    functions do not understand) becomes the real dates it covers, and so
    subjects/dates with no data are dropped rather than failing later. Returns
    unique pairs in discovery order.
    """
    from ..io.paths import find_recordings

    recordings = find_recordings(
        repo_root, subjects, dates=dates, date_range=date_range
    )
    sessions: list[tuple[int, str]] = []
    for rec in recordings:
        digits = "".join(ch for ch in rec.subject if ch.isdigit())
        pair = (int(digits) if digits else rec.subject, rec.date)
        if pair not in sessions:
            sessions.append(pair)
    return sessions

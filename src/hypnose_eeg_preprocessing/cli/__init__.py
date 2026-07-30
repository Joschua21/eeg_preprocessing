"""Command-line interface for hypnose-eeg-preprocessing.

Every command is also importable as a function, so hypnose-eeg-analysis can drive
the pipeline in-process without shelling out::

    from hypnose_eeg_preprocessing.cli import score
    score(["--model", "my-model", "--sub", "66"])

Submodules are imported lazily so `--help` on one command does not pay for the
somnotate/pomegranate import cost of the others.
"""

from __future__ import annotations

__all__ = ["main", "train", "score", "view", "distribution"]


def train(argv: list[str] | None = None) -> int:
    from .training import main as _main

    return _main(argv)


def score(argv: list[str] | None = None) -> int:
    from .scoring import main as _main

    return _main(argv)


def view(argv: list[str] | None = None) -> int:
    from .view import main as _main

    return _main(argv)


def distribution(argv: list[str] | None = None) -> int:
    from .distribution import main as _main

    return _main(argv)


COMMANDS = {
    "train": (train, "Train a somnotate model from labelled MAT files"),
    "score": (score, "Score recordings with a trained model"),
    "view": (view, "Open the detailed viewer for one scored session"),
    "distribution": (distribution, "Save state-distribution figures"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: hypnose-eeg-preprocess <command> [options]",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += ["", "Run 'hypnose-eeg-preprocess <command> --help' for command options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a subcommand."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from .. import __version__

        print(f"hypnose-eeg-preprocessing {__version__}")
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"error: unknown command {command!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    return COMMANDS[command][0](argv[1:])

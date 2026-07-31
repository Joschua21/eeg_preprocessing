"""`hypnose-somnotate distribution` — save state-distribution figures.

Reads predictions already written by `score` and writes the per-state metric
distributions into each session's `figures/` directory. Any number of subjects and
dates may be selected; one figure set is produced per session.
"""

from __future__ import annotations

import argparse
import sys

from ..config import DEFAULT_EEG_CHANNEL
from ._args import add_repo_root_argument, add_selector_arguments, resolve_selector
from ._common import resolve_sessions
from .scoring import add_distribution_arguments, run_distributions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypnose-somnotate distribution",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  hypnose-somnotate distribution --sub 66\n"
            "  hypnose-somnotate distribution --sub 66,67 --date-range 20260707-20260718\n"
            "  hypnose-somnotate distribution --sub 66 --date 20260707 --epoch-length 10\n"
        ),
    )
    add_selector_arguments(parser)
    add_distribution_arguments(parser, as_flags=False)
    parser.add_argument(
        "--eeg-channel", type=int, default=DEFAULT_EEG_CHANNEL, choices=(0, 1),
        help=f"EEG channel to analyse, 0=EEG1 1=EEG2 (default: {DEFAULT_EEG_CHANNEL}).",
    )
    add_repo_root_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        subjects, dates, date_range = resolve_selector(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return 2

    if not subjects:
        parser.error("at least one subject is required (--sub)")

    sessions = resolve_sessions(subjects, dates, date_range, args.repo_root)
    if not sessions:
        print("error: no recordings matched the selection.", file=sys.stderr)
        return 1

    print(f"Computing distributions for {len(sessions)} session(s)…")
    saved = run_distributions(sessions, args, args.repo_root)
    if not saved:
        print(
            "\nNo figures were saved — the selected sessions have no somnotate "
            "predictions yet. Run `score` for them first.",
            file=sys.stderr,
        )
        return 1
    print(f"\nSaved {saved} figure(s) across {len(sessions)} session(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

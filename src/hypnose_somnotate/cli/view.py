"""`hypnose-somnotate view` — open the detailed viewer for one scored session.

Reads predictions already written by `score`; it does not score anything. Exactly
one subject and one date must resolve, since the viewer shows a single recording.
"""

from __future__ import annotations

import argparse
import sys

from ._args import add_repo_root_argument, add_selector_arguments, resolve_selector
from ._common import resolve_sessions
from .scoring import add_viewer_arguments, run_viewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypnose-somnotate view",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  hypnose-somnotate view --sub 66 --date 20260707\n"
            "  hypnose-somnotate view --sub 066 --date 20260707 "
            "--eeg-channel 1 --view-length 60\n"
        ),
    )
    add_selector_arguments(parser)
    add_viewer_arguments(parser, as_flags=False)
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
    if len(sessions) > 1:
        listing = ", ".join(f"sub-{s:03d}/{d}" for s, d in sessions[:6])
        more = "" if len(sessions) <= 6 else f" (+{len(sessions) - 6} more)"
        print(
            f"error: the viewer shows one session, but {len(sessions)} matched: "
            f"{listing}{more}.\nNarrow the selection with --sub and --date.",
            file=sys.stderr,
        )
        return 1

    run_viewer(sessions[0], args, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

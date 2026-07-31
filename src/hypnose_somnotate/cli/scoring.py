"""`hypnose-somnotate score` — score recordings with a trained somnotate model.

Also hosts the two post-scoring views, which are reachable either as flags here
(`--show-viewer`, `--save-distribution`) or as standalone commands (`view`,
`distribution`) that read predictions already on disk.

The interactive viewer only makes sense for one session at a time, so it is
skipped automatically — with a warning, not an error — whenever the selector
resolves to more than one subject/date. Distributions have no such limit and are
written once per session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import (
    DEFAULT_DISTRIBUTION_BIN_WIDTH,
    DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S,
    DEFAULT_EEG_CHANNEL,
    DEFAULT_RECORDING_INDEX,
    DEFAULT_SAMPLING_RATE_HZ,
    DEFAULT_VIEW_LENGTH_S,
)
from ._args import add_repo_root_argument, add_selector_arguments, resolve_selector
from ._common import resolve_model_path, resolve_sessions


# --------------------------------------------------------------------------------
# Shared option groups
# --------------------------------------------------------------------------------

def add_viewer_arguments(parser: argparse.ArgumentParser, *, as_flags: bool = True) -> None:
    """Options controlling the interactive detailed viewer.

    With `as_flags`, `--show-viewer` is added too; the standalone `view` command
    always shows one, so it omits the switch.
    """
    group = parser.add_argument_group("viewer")
    if as_flags:
        group.add_argument(
            "--show-viewer", action="store_true",
            help="Open the interactive detailed view after scoring. "
                 "Single subject and date only; skipped otherwise.",
        )
    group.add_argument(
        "--recording-index", type=int, default=DEFAULT_RECORDING_INDEX, metavar="N",
        help=f"Which EDF to view when a session has several (default: {DEFAULT_RECORDING_INDEX}).",
    )
    group.add_argument(
        "--eeg-channel", type=int, default=DEFAULT_EEG_CHANNEL, choices=(0, 1),
        help=f"EEG channel to plot, 0=EEG1 1=EEG2 (default: {DEFAULT_EEG_CHANNEL}).",
    )
    group.add_argument(
        "--view-length", type=float, default=DEFAULT_VIEW_LENGTH_S, metavar="SECONDS",
        help=f"Width of the viewer window (default: {DEFAULT_VIEW_LENGTH_S}).",
    )


def add_distribution_arguments(parser: argparse.ArgumentParser, *, as_flags: bool = True) -> None:
    """Options controlling the per-state metric distributions."""
    group = parser.add_argument_group("distribution")
    if as_flags:
        group.add_argument(
            "--save-distribution", action="store_true",
            help="Save state-distribution figures for every scored session.",
        )
    group.add_argument(
        "--epoch-length", type=float, default=DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S,
        metavar="SECONDS",
        help="Analysis epoch for the distributions — independent of somnotate's "
             f"scoring epoch (default: {DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S}).",
    )
    group.add_argument(
        "--bin-width", type=float, default=DEFAULT_DISTRIBUTION_BIN_WIDTH, metavar="W",
        help=f"Histogram bin width, fraction of the normalised range "
             f"(default: {DEFAULT_DISTRIBUTION_BIN_WIDTH}).",
    )
    group.add_argument(
        "--no-raw-distributions", action="store_true",
        help="Only produce the normalised overlay, not the per-state raw grid.",
    )


# --------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------

def run_viewer(session: tuple[int, str], args, repo_root: Path):
    """Open the detailed interactive view for one (subject, date)."""
    from ..visualization import plot_scoring_detailed

    subject, date = session
    print(f"\nOpening viewer for sub-{subject:03d} date-{date} …")
    result = plot_scoring_detailed(
        subject, date, repo_root,
        recording_index=args.recording_index,
        eeg_channel=args.eeg_channel,
        view_length_s=args.view_length,
    )
    # A Qt/interactive backend needs an explicit show() to become responsive when
    # driven from a script rather than a notebook.
    import matplotlib.pyplot as plt

    plt.show()
    return result


def run_distributions(sessions: list[tuple[int, str]], args, repo_root: Path) -> int:
    """Save state-distribution figures for each session. Returns the count saved."""
    from ..visualization import plot_state_distributions

    saved = 0
    for subject, date in sessions:
        outputs = plot_state_distributions(
            subject, date, repo_root,
            epoch_length_s=args.epoch_length,
            bin_width=args.bin_width,
            show_raw_distributions=not args.no_raw_distributions,
            eeg_channel=args.eeg_channel,
            inline=False,
            save=True,
        )
        saved += sum(len(o.get("saved_paths", [])) for o in outputs)
    return saved


def _select_viewer_session(sessions: list[tuple[int, str]]) -> tuple[int, str] | None:
    """The single session to view, or None when the selection is ambiguous."""
    if not sessions:
        return None
    if len(sessions) > 1:
        subjects = sorted({s for s, _ in sessions})
        dates = sorted({d for _, d in sessions})
        print(
            f"warning: --show-viewer needs a single subject and date, but the selection "
            f"resolved to {len(sessions)} sessions "
            f"({len(subjects)} subject(s), {len(dates)} date(s)). Skipping the viewer.",
            file=sys.stderr,
        )
        return None
    return sessions[0]


# --------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypnose-somnotate score",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  hypnose-somnotate score --model my-model --sub 66\n"
            "  hypnose-somnotate score --model my-model --sub 066,067 --date 20260707\n"
            "  hypnose-somnotate score --model my-model --sub 66 "
            "--date-range 20260707-20260718 --save-distribution\n"
            "  hypnose-somnotate score --model my-model --sub 66 --date 20260707 --show-viewer\n"
        ),
    )
    parser.add_argument(
        "--model", required=True, metavar="NAME_OR_PATH",
        help="Trained model: a name under derivatives/somnotate_training/, "
             "a model directory, or a path to model.pickle.",
    )
    add_selector_arguments(parser)
    parser.add_argument(
        "--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE_HZ, metavar="HZ",
        help=f"Sampling rate of the recordings (default: {DEFAULT_SAMPLING_RATE_HZ}).",
    )
    parser.add_argument(
        "--no-visbrain", action="store_true",
        help="Skip writing the visbrain hypnogram .txt alongside the predictions.",
    )
    add_viewer_arguments(parser)
    add_distribution_arguments(parser)
    add_repo_root_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root

    try:
        subjects, dates, date_range = resolve_selector(args)
        model_path = resolve_model_path(args.model, repo_root)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return 2  # unreachable; parser.error exits

    if not subjects:
        parser.error("at least one subject is required (--sub)")

    print(f"Scoring with model {model_path}")
    print(f"  subjects: {', '.join(f'sub-{s:03d}' for s in subjects)}")
    if dates:
        print(f"  dates   : {', '.join(dates)}")
    elif date_range:
        print(f"  dates   : {date_range[0]}–{date_range[1]} (inclusive)")
    else:
        print("  dates   : all")

    from ..scoring import score_recordings

    output_paths = score_recordings(
        subjids=subjects,
        model_path=model_path,
        repo_root=repo_root,
        dates=dates,
        date_range=date_range,
        export_visbrain=not args.no_visbrain,
        sampling_rate_hz=args.sampling_rate,
    )
    if not output_paths:
        print("\nNo recordings were scored.", file=sys.stderr)
        return 1
    print(f"\nScored {len(output_paths)} recording(s).")

    if args.show_viewer or args.save_distribution:
        sessions = resolve_sessions(subjects, dates, date_range, repo_root)

        if args.save_distribution:
            saved = run_distributions(sessions, args, repo_root)
            print(f"Saved {saved} distribution figure(s) across {len(sessions)} session(s).")

        if args.show_viewer:
            session = _select_viewer_session(sessions)
            if session is not None:
                run_viewer(session, args, repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

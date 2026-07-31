"""Validation of automated scoring against manual annotations.

The loaders below now live in `hypnose_somnotate.io.loading`; they are
re-exported here so existing `from …testing import load_…` callers keep working.
"""

from ..io.loading import (
    align_vectors,
    find_somnotate_predictions,
    get_csv_dir,
    get_predictions_dir,
    get_test_root,
    list_csv_files,
    load_aligned_vectors,
    load_manual_vectors_from_csvs,
    load_raw_signals_from_csv,
    load_signal_table,
    load_somnotate_predictions,
    load_testing_csv,
)
from .testing import (
    agreement_matrix,
    ensure_csvs,
    ensure_somnotate_predictions,
    plot_agreement_matrix,
    plot_testing_comparison,
    plot_testing_comparison_detailed,
    prepare_testing_csvs,
    print_agreement_diagnostic,
    save_somnotate_predictions,
    score_somnotate,
)

__all__ = [
    "agreement_matrix",
    "align_vectors",
    "ensure_csvs",
    "ensure_somnotate_predictions",
    "find_somnotate_predictions",
    "get_csv_dir",
    "get_predictions_dir",
    "get_test_root",
    "list_csv_files",
    "load_aligned_vectors",
    "load_manual_vectors_from_csvs",
    "load_raw_signals_from_csv",
    "load_signal_table",
    "load_somnotate_predictions",
    "load_testing_csv",
    "plot_agreement_matrix",
    "plot_testing_comparison",
    "plot_testing_comparison_detailed",
    "prepare_testing_csvs",
    "print_agreement_diagnostic",
    "save_somnotate_predictions",
    "score_somnotate",
]

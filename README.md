# hypnose-eeg-preprocessing

Somnotate-based EEG preprocessing and scoring pipeline. Importable as
`hypnose_eeg_preprocessing`; intended to be consumed as the preprocessing submodule of
`hypnose-eeg-analysis`.

## Layout

```
src/hypnose_eeg_preprocessing/
├── config.py               # sampling rates, channel labels, state-label maps
├── io/
│   ├── paths.py            # dataset discovery + data-location resolution
│   └── loading.py          # all readers for pipeline artifacts on disk
├── preprocessing/          # preprocessing.py, gap_correction.py
├── training/training.py    # run_training
├── scoring/scoring.py      # score_recordings
├── testing/testing.py      # manual-vs-automated validation
├── visualization/          # plotting
├── utils/labels.py         # manual-label loading
├── somnotate/              # vendored somnotate library (upstream, unmodified layout)
└── somnotate_pipeline/     # vendored somnotate example pipeline
    ├── io/data_io.py
    ├── utils/configuration.py
    ├── preprocessing/      # mat_to_csv.py, preprocess_signals.py
    ├── processing/         # adjust_delimiters.py, edf_vis_gen.py
    └── state_annotation/   # train/run/test_state_annotation.py
scripts/                    # CLI entry points (training.py, scoring.py)
notebooks/
```

Each subpackage re-exports its public API, so the import path stays flat:

```python
from hypnose_eeg_preprocessing.scoring import score_recordings
from hypnose_eeg_preprocessing.training import run_training
from hypnose_eeg_preprocessing.preprocessing import prepare_recording
from hypnose_eeg_preprocessing.visualization import plot_scoring_detailed
```

All disk reads live in `io/loading.py` — path resolution, directory iteration, and
parsing of the files the pipeline writes. It depends only on `io/paths.py` and
`config.py`, so it stays free of cycles with the modules that consume it. The
selector-driven entry point is:

```python
from hypnose_eeg_preprocessing.io import load_scored_recording
recording, raw_signals, somnotate_vec = load_scored_recording(sub, date, repo_root)
```

## Setup

### 1. Environment

This pipeline uses **hypnose-analysis** for data-location resolution and for the
shared figure styles. Install both into one conda env:

1. Clone hypnose-analysis (anywhere; a sibling folder is convenient):
   ```bash
   git clone https://github.com/SainsburyWellcomeCentre/hypnose-analysis.git
   ```

2. From **this** repo's root, create and activate the env. This installs
   hypnose-eeg-preprocessing itself (`-e .` is the last line of `environment.yml`),
   so no separate install is needed:
   ```bash
   conda env create -f environment.yml
   conda activate hypnose-eeg-preprocessing
   ```

3. Install hypnose-analysis into the env, from its main folder:
   ```bash
   cd /path/to/hypnose-analysis
   pip install -e .
   cd -
   ```
   Use the **base** install — no `[behavioral]` extra. This repo needs only
   `hypnose.io.paths` and `hypnose.io.save`; the behavioural stack pins `swc-aeon`,
   which requires Python ≥3.11 and cannot install here (this repo is pinned to 3.9
   by pomegranate).

4. Register the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name hypnose-eeg-preprocessing \
       --display-name "Python (hypnose-eeg-preprocessing)"
   ```

5. Check it worked:
   ```bash
   hypnose-eeg-preprocess --help
   ```

> **Reusing an older env** (e.g. `eeg_preprocessing` from before the rename)?
> Run `pip install -e .` in it by hand — the old editable install still points at
> the pre-restructure `utils/` layout and will not import.

### 2. Set the data location

The pipeline finds the EEG dataset's `rawdata/` and `derivatives/`
(`…/hypnose_eeg/{rawdata,derivatives}`) through hypnose-analysis's data-location system —
no symlink needed. Resolution priority:

`HYPNOSE_EEG_*` env vars → active hypnose-analysis profile (`server_root/hypnose_eeg`) →
legacy `data/hypnose_eeg` symlink (fallback only).

Pick the option that matches your machine.

#### Option 1 — full Hypnose layout (behavioral data + `hypnose_eeg` sibling)

If your disk looks like:
```
…/hypnose/
├── rawdata/          # behavioral
├── derivatives/      # behavioral
└── hypnose_eeg/
    ├── rawdata/      # EEG
    └── derivatives/  # EEG
```
add a profile to hypnose-analysis's `configs/data_locations.yml`:
```yaml
  my-machine:
    rawdata: /path/to/hypnose/rawdata #behavioral 
    derivatives: /path/to/hypnose/derivatives #behavioral 
```
then activate it from the hypnose-analysis repo:
```bash
python scripts/set_data_location.py my-machine
python scripts/set_data_location.py --show      # verify resolved roots
```
The EEG code derives `…/hypnose/hypnose_eeg/{rawdata,derivatives}` automatically from the
profile's server root.

#### Option 2 — standalone EEG data (no behavioral folders)

Point directly at the EEG roots with env vars — no profile, no server-root convention.

**2a. Set them in the conda env (recommended — works however the kernel is launched,
including VS Code from the Dock):**
```bash
conda activate hypnose-eeg-preprocessing
conda env config vars set \
  HYPNOSE_EEG_RAWDATA_ROOT=/path/to/eeg/rawdata \
  HYPNOSE_EEG_DERIVATIVES_ROOT=/path/to/eeg/derivatives
conda deactivate && conda activate hypnose-eeg-preprocessing   # reactivate to apply
conda env config vars list                                      # verify
```
Test (from anywhere, once the package is installed):
```bash
python -c "from hypnose_eeg_preprocessing.io import get_derivatives_root; print(get_derivatives_root())"
```
It should print your EEG derivatives path.

**2b. Or set them per notebook — first cell, before importing `hypnose_eeg_preprocessing`:**
```python
import os
os.environ["HYPNOSE_EEG_RAWDATA_ROOT"] = "/path/to/eeg/rawdata"
os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"] = "/path/to/eeg/derivatives"
```
The resolver reads the env at call time, so this takes effect immediately (no restart).
Downside: it lives in the notebook and must be re-run each session — don't commit
machine-specific paths into a shared notebook.

## Notebooks
- notebooks/training.ipynb
- notebooks/scoring.ipynb
- notebooks/testing.ipynb
- notebooks/utils.ipynb

## CLI

Installed as `hypnose-eeg-preprocess`; also runnable as
`python -m hypnose_eeg_preprocessing.cli` from a checkout.

```
hypnose-eeg-preprocess train         Train a somnotate model from labelled MAT files
hypnose-eeg-preprocess score         Score recordings with a trained model
hypnose-eeg-preprocess view          Open the detailed viewer for one scored session
hypnose-eeg-preprocess distribution  Save state-distribution figures
```

### Selectors

Subjects and dates are forgiving — these all mean the same thing:

```bash
--sub 66            --sub 066            --sub sub-066
--sub 66,67,68      --sub 66 67 68       --subjid / --subjids also work
--date 20260707     --dates 20260707,20260708
--date-range 20260707,20260718           --date-range 20260707-20260718
```

Omit `--date`/`--date-range` to select every session for those subjects.

### Training

```bash
hypnose-eeg-preprocess train my-model
hypnose-eeg-preprocess train my-model --input-dir cohort-a
```

The model name is also the output directory
(`derivatives/somnotate_training/my-model/model.pickle`), which is what
`score --model my-model` resolves later. An existing model is never overwritten:
the command stops, lists the models that exist, and asks for a different name —
pass `--force` to overwrite deliberately. `--input-dir` reads labelled `.mat`
files from `somnotate_training/<dir>/` so several labelled sets can be trained
separately.

### Scoring

```bash
hypnose-eeg-preprocess score --model my-model --sub 66
hypnose-eeg-preprocess score --model my-model --sub 066,067 --date 20260707
hypnose-eeg-preprocess score --model my-model --sub 66 \
    --date-range 20260707-20260718 --save-distribution
hypnose-eeg-preprocess score --model my-model --sub 66 --date 20260707 --show-viewer
```

`--model` accepts a model name, a model directory, or a path to `model.pickle`.

`--show-viewer` opens the interactive detailed view after scoring. It needs a
single session, so it is skipped with a warning (not an error) whenever the
selection resolves to more than one subject/date. Tune it with
`--recording-index`, `--eeg-channel`, `--view-length`.

`--save-distribution` writes state-distribution figures for every scored session
and works across any number of subjects/dates. Tune it with `--epoch-length`,
`--bin-width`, `--no-raw-distributions`.

### Standalone views

Both views also run on their own against predictions already on disk:

```bash
hypnose-eeg-preprocess view --sub 66 --date 20260707
hypnose-eeg-preprocess distribution --sub 66,67 --date-range 20260707-20260718
```

### From Python

Each command is importable, so hypnose-eeg-analysis can drive the pipeline
in-process rather than shelling out:

```python
from hypnose_eeg_preprocessing.cli import score
score(["--model", "my-model", "--sub", "66", "--save-distribution"])
```

`scripts/training.py` and `scripts/scoring.py` are thin shims over the same code,
for running straight from a checkout.

## Figure styles

Figures are styled by hypnose-analysis so every Hypnose repo produces the same
look. `config.DEFAULT_FIGURE_STYLE` selects it (`"nature"`, `"poster"`,
`"presentation"`, or `None` to leave matplotlib alone); edit it and restart the
kernel, or switch for one session:

```python
from hypnose_eeg_preprocessing.io.style import ensure_style
ensure_style("presentation", force=True)
```

The style is applied when a figure is created, never at import, so importing this
package does not touch your rcParams and the result does not depend on import
order. If hypnose-analysis is not installed, plots render unstyled and only
figure *saving* raises.

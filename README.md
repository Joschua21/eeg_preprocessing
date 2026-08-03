# hypnose-somnotate

Somnotate-based EEG sleep scoring pipeline. Importable as `hypnose_somnotate`.

## Layout

```
src/hypnose_somnotate/
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
from hypnose_somnotate.scoring import score_recordings
from hypnose_somnotate.training import run_training
from hypnose_somnotate.preprocessing import prepare_recording
from hypnose_somnotate.visualization import plot_scoring_detailed
```

All disk reads live in `io/loading.py` — path resolution, directory iteration, and
parsing of the files the pipeline writes. It depends only on `io/paths.py` and
`config.py`, so it stays free of cycles with the modules that consume it. The
selector-driven entry point is:

```python
from hypnose_somnotate.io import load_scored_recording
recording, raw_signals, somnotate_vec = load_scored_recording(sub, date, repo_root)
```

## Setup

### 1. Environment

This pipeline uses **hypnose-behavior-analysis** for data-location resolution and for the
shared figure styles. Install both into one conda env:

1. Clone hypnose-behavior-analysis (anywhere; a sibling folder is convenient):
   ```bash
   git clone https://github.com/SainsburyWellcomeCentre/hypnose-behavior-analysis.git
   ```

2. From **this** repo's root, create and activate the env. This installs
   hypnose-somnotate itself (`-e .` is the last line of `environment.yml`),
   so no separate install is needed:
   ```bash
   conda env create -f environment.yml
   conda activate hypnose-somnotate
   ```

3. Install hypnose-behavior-analysis into the env, from its main folder:
   ```bash
   cd /path/to/hypnose-behavior-analysis
   pip install -e .
   cd -
   ```
   Use the **base** install — no `[behavioral]` extra. This repo needs only
   `hypnose_behavior.io.paths` and `hypnose_behavior.io.save`; the behavioural stack (`swc-aeon`,
   `harp-python`, `moviepy`, `opencv-python`) is for behavioural data and video and
   is never imported here.

4. Register the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name hypnose-somnotate \
       --display-name "Python (hypnose-somnotate)"
   ```

5. Check it worked:
   ```bash
   hypnose-somnotate --help
   ```

> **Reusing an older env** (this repo was previously `eeg_preprocessing`, then
> `hypnose-eeg-preprocessing`). Run `pip install -e .` in it by hand — an editable
> install from before a rename still points at the old package name and will not
> import.

### 2. Set the data location

The pipeline finds the EEG dataset's `rawdata/` and `derivatives/`
(`…/hypnose_eeg/{rawdata,derivatives}`) through hypnose-behavior-analysis's data-location system —
no symlink needed. Resolution priority:

`HYPNOSE_EEG_*` env vars → active hypnose-behavior-analysis profile (`server_root/hypnose_eeg`) →
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
add a profile to hypnose-behavior-analysis's `configs/data_locations.yml`:
```yaml
  my-machine:
    rawdata: /path/to/hypnose/rawdata #behavioral 
    derivatives: /path/to/hypnose/derivatives #behavioral 
```
then activate it from the hypnose-behavior-analysis repo:
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
conda activate hypnose-somnotate
conda env config vars set \
  HYPNOSE_EEG_RAWDATA_ROOT=/path/to/eeg/rawdata \
  HYPNOSE_EEG_DERIVATIVES_ROOT=/path/to/eeg/derivatives
conda deactivate && conda activate hypnose-somnotate   # reactivate to apply
conda env config vars list                                      # verify
```
Test (from anywhere, once the package is installed):
```bash
python -c "from hypnose_somnotate.io import get_derivatives_root; print(get_derivatives_root())"
```
It should print your EEG derivatives path.

**2b. Or set them per notebook — first cell, before importing `hypnose_somnotate`:**
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

Installed as `hypnose-somnotate`; also runnable as
`python -m hypnose_somnotate.cli` from a checkout.

```
hypnose-somnotate train         Train a somnotate model from labelled MAT files
hypnose-somnotate score         Score recordings with a trained model
hypnose-somnotate view          Open the detailed viewer for one scored session
hypnose-somnotate distribution  Save state-distribution figures
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
hypnose-somnotate train my-model
hypnose-somnotate train my-model --input-dir cohort-a
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
hypnose-somnotate score --model my-model --sub 66
hypnose-somnotate score --model my-model --sub 066,067 --date 20260707
hypnose-somnotate score --model my-model --sub 66 \
    --date-range 20260707-20260718 --save-distribution
hypnose-somnotate score --model my-model --sub 66 --date 20260707 --show-viewer
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
hypnose-somnotate view --sub 66 --date 20260707
hypnose-somnotate distribution --sub 66,67 --date-range 20260707-20260718
```

### From Python

Each command is importable, so hypnose-eeg-analysis can drive the pipeline
in-process rather than shelling out:

```python
from hypnose_somnotate.cli import score
score(["--model", "my-model", "--sub", "66", "--save-distribution"])
```

`scripts/training.py` and `scripts/scoring.py` are thin shims over the same code,
for running straight from a checkout.

## Loading scored data

The everyday loop is: **score once**, then load the results back to compute things.
`load_scores` is the read side — it takes the same selectors as the CLI and returns one
tidy DataFrame, reading only the predictions parquet (never the EDF), so it is fast
enough to call across a whole cohort.

```python
from hypnose_somnotate.io import load_scores

df = load_scores(66, date=20260707)          # one session
df = load_scores([66, 67, 68])               # several animals, all their dates
df = load_scores(66, date_range="20260707-20260718")
```

`66`, `"066"` and `"sub-066"` are interchangeable, as are lists and `"66,67"`.

Every row is one epoch. Four identifier columns are prepended to what was stored:

| | |
|---|---|
| `subject` `date` `session` `recording` | which recording the row came from |
| `epoch_s` | duration of one epoch, in seconds — **use this for durations** |
| `time_s` | epoch start, seconds into the recording |
| `label_output` | 0=Wake, 1=NREM, 2=REM, 3=Undefined |
| `label_model` | somnotate's own coding (1=Wake, 2=NREM, 3=REM, 0=Undefined) |
| `segment_id` `kind` | which continuous chunk the epoch belongs to (recordings are split around dropouts) |
| `prob_wake` `prob_nrem` `prob_rem` `prob_undef` | per-state model confidence |

Because it is one frame, per-animal summaries are a groupby rather than a loop:

```python
df = load_scores([66, 67], date_range="20260707-20260718")
nrem = df[df["label_output"] == 1]
nrem.groupby(["subject", "date"])["epoch_s"].sum() / 3600      # NREM hours
```

> Sum `epoch_s` rather than counting rows and multiplying by a constant. Scored epochs
> are 1 s (somnotate's `time_resolution`) — *not* the 10 s
> `DEFAULT_SLEEP_STAGE_RESOLUTION_S`, which is the epoch of the **manual** labels used
> for training. Those two are independent, and mixing them silently scales every
> duration. `epoch_s` is read from each file, so it stays right regardless.

Pass `columns=[...]` to read only what you need — it is pushed down to the parquet
reader, so it saves IO as well as memory:

```python
load_scores(66, columns=["time_s", "label_output"])
```

Recordings that have not been scored yet are skipped with a warning; use
`missing="raise"` to fail instead, or `missing="ignore"` to stay quiet. If *nothing*
in the selection is scored you get a `FileNotFoundError`, so an empty result is never
mistaken for "no sleep found".

Two companions:

```python
from hypnose_somnotate.io import find_scored, load_segments

find_scored(66)                        # what exists on disk, without reading it
find_scored(66, scored_only=False)     # ...including what still needs scoring
load_segments(ref)                     # chunk/gap metadata; segment_id indexes into it
```

`find_scored` returns `ScoredRef` objects (`subject`, `date`, `session`, `recording`,
`edf_path`, `predictions_path`, `segments_path`, `scored`) for when you want per-file
control rather than one concatenated frame.

> This package deliberately stops at loading. Downstream computation — bout detection,
> sleep-period statistics, cross-animal modelling — belongs in `hypnose-eeg-analysis`.

## Figure styles

Figures are styled by hypnose-behavior-analysis so every Hypnose repo produces the same
look. `config.DEFAULT_FIGURE_STYLE` selects it (`"nature"`, `"poster"`,
`"presentation"`, or `None` to leave matplotlib alone); edit it and restart the
kernel, or switch for one session:

```python
from hypnose_somnotate.io.style import ensure_style
ensure_style("presentation", force=True)
```

The style is applied when a figure is created, never at import, so importing this
package does not touch your rcParams and the result does not depend on import
order. If hypnose-behavior-analysis is not installed, plots render unstyled and only
figure *saving* raises.

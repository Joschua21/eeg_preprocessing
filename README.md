# eeg_preprocessing

Somnotate-based EEG preprocessing and scoring pipeline.

## Setup

### 1. Environment

This pipeline depends on **hypnose-analysis** for data-location resolution (the shared
`set_data_location` config that replaces the old per-clone symlink). Install both into
one conda env:

1. Clone hypnose-analysis (anywhere; a sibling folder is convenient):
   ```bash
   git clone https://github.com/SainsburyWellcomeCentre/hypnose-analysis.git
   ```
2. Create and activate the env (installs this repo and its dependencies):
   ```bash
   conda env create -f environment.yml
   conda activate eeg_preprocessing
   ```
3. Install hypnose-analysis into the env (editable), from its main folder:
   ```bash
   cd /path/to/hypnose-analysis
   pip install -e .
   cd -
   ```
4. Register the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name eeg_preprocessing --display-name "Python (eeg_preprocessing)"
   ```

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
conda activate eeg_preprocessing
conda env config vars set \
  HYPNOSE_EEG_RAWDATA_ROOT=/path/to/eeg/rawdata \
  HYPNOSE_EEG_DERIVATIVES_ROOT=/path/to/eeg/derivatives
conda deactivate && conda activate eeg_preprocessing   # reactivate to apply
conda env config vars list                              # verify
```
Test (from the eeg_preprocessing repo root):
```bash
python -c "from utils.paths import get_derivatives_root; print(get_derivatives_root())"
```
It should print your EEG derivatives path.

**2b. Or set them per notebook — first cell, before importing `utils`:**
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

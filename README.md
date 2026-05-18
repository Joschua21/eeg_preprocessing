# eeg_preprocessing

Somnotate-based EEG preprocessing and scoring pipeline.

Setup notes:
- Create a symlink at `data/hypnose_eeg` that points to the hypnose_eeg root (contains rawdata/ and derivatives/).

- Conda env: 
    1. conda env create -f environment.yml
    2. python -m ipykernel install --user --name eeg_preprocessing --display-name "Python (eeg_preprocessing)"

Notebooks:
- notebooks/training.ipynb
- notebooks/scoring.ipynb

import pandas as pd
import numpy as np
import pyedflib
import os
import warnings


def generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, test_train_or_to_score, base_directory=None, sampling_rate=None):
    '''
    Generate EDF and Visbrain stage duration format files from CSV files, respectively for EEG and EMG data and sleep stage annotations.
    
    Inputs:
    mouse_ids: list of str, mouse IDs
    sessions: list of str, session IDs
    recordings: list of str, recording IDs
    test_train_or_to_score: str, 'test', 'train' or 'to_score' to specify which dataset to process
    base_directory: str, path to the base directory where the CSV files are stored and where the output EDF and annotations files should be saved

    Outputs:
    EDF files and annotations in Visbrain stage duration format are saved in the 'edfs' and '{test_train_or_to_score}_manual_annotation' directories, respectively. 
    
    '''
    
    if base_directory is None or sampling_rate is None:
        raise ValueError("base_directory and sampling_rate must be provided")

    csv_input_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set/{test_train_or_to_score}_csv_files") # this directory should already exist and contain the CSV files

    edf_output_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set", 'edfs') # this directory will be created to store the EDF files
    annotations_output_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set", f"{test_train_or_to_score}_manual_annotation") # this directory will be created to store the annotations files
    
    if not os.path.exists(edf_output_dir):
        os.makedirs(edf_output_dir)
    if not os.path.exists(annotations_output_dir):
        os.makedirs(annotations_output_dir)


    # Suppress pyedflib EDF header precision warnings
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pyedflib\.edfwriter",
        message=r"Physical (minimum|maximum) for channel",
    )

    # Read the CSV file
    for mouse_id in mouse_ids:
            for session in sessions:
                for recording in recordings:
                    stem = f"{mouse_id}_{session}_{recording}"
                    parquet_file = os.path.join(csv_input_dir, f"{stem}.parquet")
                    csv_file = os.path.join(csv_input_dir, f"{stem}.csv")
                    edf_file = os.path.join(edf_output_dir, f"output_{stem}.edf")
                    visbrain_file = os.path.join(annotations_output_dir, f"annotations_visbrain_{stem}.txt")

                    if os.path.isfile(parquet_file):
                        signal_file = parquet_file
                    elif os.path.isfile(csv_file):
                        signal_file = csv_file
                    else:
                        print(f"File not found: {parquet_file} (or .csv)")
                        continue
                    if os.path.exists(edf_file):
                        print(f"EDF file already exists: output_{stem}.edf")
                        continue

                    print(f"Processing file: {signal_file}")
                    if signal_file.endswith(".parquet"):
                        df = pd.read_parquet(signal_file)
                    else:
                        df = pd.read_csv(signal_file, encoding="latin1")
                    base_filename = stem
                    print(base_filename)
                    sampling_rate = sampling_rate  # Hz (samples per second)

                    # Extract EEG and EMG data
                    eeg1_data = df["EEG1"].to_numpy()
                    eeg2_data = df["EEG2"].to_numpy()
                    emg_data = df["EMG"].to_numpy()

                    # Combine all data
                    all_data = np.array([eeg1_data, eeg2_data, emg_data])

                    # Create an EDF file
                    f = pyedflib.EdfWriter(edf_file, len(all_data), file_type=pyedflib.FILETYPE_EDFPLUS)

                    # Define EDF header information
                    # Define signal info
                    labels = ["EEG1", "EEG2", "EMG"]
                    for i, label in enumerate(labels):
                        signal_info = {
                            'label': label,
                            'dimension': 'uV',
                            'sample_frequency': sampling_rate,
                            'physical_min': np.min(all_data[i]),
                            'physical_max': np.max(all_data[i]),
                            'digital_min': -32768,
                            'digital_max': 32767,
                            'transducer': '',
                            'prefilter': ''
                        }
                        f.setSignalHeader(i, signal_info)

                    # Write EEG and EMG data to the EDF file
                    f.writeSamples(all_data)

                    # Close the EDF file
                    f.close()

                    # Prepare annotations in Visbrain stage duration format
                    annotations = [(0, 10, "Undefined")]  # Initial undefined stage
                    current_stage = None
                    start_time = 10 / sampling_rate  # Convert start time to seconds

                    for i, label in enumerate(df["sleepStage"]):
                        current_time = i / sampling_rate  # Convert sample index to time in seconds
                        if label != current_stage:
                            if current_stage is not None:
                                annotations.append((start_time, current_time, current_stage))
                            current_stage = label
                            start_time = current_time
                    annotations.append((start_time, len(df) / sampling_rate, current_stage))  # Last stage duration

                    # Write annotations in Visbrain stage duration format to a text file
                    last_time_value = annotations[-1][1]
                    with open(visbrain_file, "w") as f:
                        f.write(f"*Duration_sec    {last_time_value}\n")
                        f.write("*Datafile\tUnspecified\n")
                        for start, end, stage in annotations:
                            stage_label = {1: "awake", 2: "non-REM", 3: "REM", 4: 'ambiguous', 5: 'doubt'}.get(stage, "Undefined")
                            f.write(f"{stage_label}    {end}\n")

                    print(f"EDF file and annotations in Visbrain stage duration format have been created successfully for {mouse_id}, {session}, {recording}.")
                    


if __name__ == "__main__":
    mouse_ids = input("Enter mouse IDs comma-separated without spaces (e.g., sub-001,sub-002): ").split(',')
    sessions = input("Enter session IDs comma-separated without spaces (e.g., ses-01,ses-02): ").split(',')
    recordings = input("Enter recording IDs comma-separated without spaces (e.g., recording-01): ").split(',')
    test_train_or_to_score = input("Enter dataset type ('test', 'train', or 'to_score'): ").strip()
    base_directory = input("Enter the base somnotate directory path without quotes (e.g., Z:/somnotate): ").strip()
    sampling_rate = float(input("Enter the sampling rate in Hz (e.g., 512.0): "))

    generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, test_train_or_to_score, base_directory, sampling_rate)
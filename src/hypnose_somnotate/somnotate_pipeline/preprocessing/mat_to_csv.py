# Convert Spike2-exported .mat recordings into per-recording signal tables.
# Writes parquet by default; pass output_format="csv" to keep the legacy format.

import glob
import os

import h5py
import numpy as np
import pandas as pd


def mat_to_signal_tables(
    input_directory_path,
    output_directory_path,
    sampling_rate,
    sleep_stage_resolution,
    output_format="parquet",
):
    '''
    Convert .mat files extracted from Spike2 into per-recording signal tables.

    Inputs:
    input_directory_path: str, directory containing .mat files
    output_directory_path: str, directory to save signal tables in
    sampling_rate: int, Hz
    sleep_stage_resolution: int, seconds
    output_format: "parquet" (default) or "csv"
    '''
    if output_format not in ("parquet", "csv"):
        raise ValueError(f"output_format must be 'parquet' or 'csv', got {output_format!r}")

    mat_files = glob.glob(os.path.join(input_directory_path, '*.mat'))
    print(f"Found .mat files: {mat_files}")

    large_mismatch_files = []

    for file_path in mat_files:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        with h5py.File(file_path, 'r') as raw_data:
            print(f'Processing file: {file_path}')

            eeg1_data, eeg2_data, emg_data, sleep_stages = None, None, None, None

            for key in raw_data.keys():
                if key.endswith('_EEG_EEG1A_B') or key.endswith('_EEGorig'):
                    eeg1_data = np.array(raw_data[key]['values'])
                elif key.endswith('_EEG_EEG2A_B') or key.endswith('_EEGorig'):
                    eeg2_data = np.array(raw_data[key]['values'])
                elif key.endswith('_EMG_EMG'):
                    emg_data = np.array(raw_data[key]['values'])
                elif key.endswith('_Stage_1_'):
                    sleep_stages = np.array(raw_data[key]['codes'])
                    sleep_stages = sleep_stages[0, :]

            if eeg1_data is not None:
                print("EEG1 data extracted successfully.")
            if eeg2_data is not None:
                print("EEG2 data extracted successfully.")
            if emg_data is not None:
                print("EMG data extracted successfully.")
            if sleep_stages is not None:
                print("Sleep stage data extracted successfully.")

            eeg1_flattened = eeg1_data.flatten()
            eeg2_flattened = eeg2_data.flatten()
            emg_flattened = emg_data.flatten()
            assert eeg1_flattened.shape == eeg2_flattened.shape == emg_flattened.shape, \
                "The flattened shapes of the EEG and EMG data do not match"

            upsampled_sleep_stages = np.repeat(sleep_stages, sampling_rate * sleep_stage_resolution)
            if len(upsampled_sleep_stages) != len(eeg1_flattened):
                mismatch = len(eeg1_flattened) - len(upsampled_sleep_stages)
                print(
                    f"Length of upsampled sleep stages ({len(upsampled_sleep_stages)}) "
                    f"does not match length of EEG data ({len(eeg1_flattened)}) "
                    f"by {mismatch} samples"
                )
                if abs(mismatch) > 1:
                    large_mismatch_files.append((file_path, mismatch))
                if len(upsampled_sleep_stages) > len(eeg1_flattened):
                    upsampled_sleep_stages = upsampled_sleep_stages[:len(eeg1_flattened)]
                    print("Upsampled sleep stages truncated to match length of EEG data")
                else:
                    padding_length = len(eeg1_flattened) - len(upsampled_sleep_stages)
                    upsampled_sleep_stages = np.pad(upsampled_sleep_stages, (0, padding_length), mode='constant')
                    print("Upsampled sleep stages padded with zeros to match length of EEG data")
                assert len(upsampled_sleep_stages) == len(eeg1_flattened), \
                    "Length of upsampled sleep stages does not match length of EEG data after truncation"
                print("Length of upsampled sleep stages matches length of EEG data after truncation")

            df = pd.DataFrame({
                'sleepStage': upsampled_sleep_stages,
                'EEG1': eeg1_flattened,
                'EEG2': eeg2_flattened,
                'EMG': emg_flattened,
            })

            if not os.path.exists(output_directory_path):
                os.makedirs(output_directory_path)

            if output_format == "parquet":
                output_file_path = os.path.join(output_directory_path, base_filename + '.parquet')
                df.to_parquet(output_file_path, index=False, compression='snappy')
                print(f'Saved parquet to: {output_file_path}')
            else:
                output_file_path = os.path.join(output_directory_path, base_filename + '.csv')
                df.to_csv(output_file_path, index=False)
                print(f'Saved CSV to: {output_file_path}')

    if large_mismatch_files:
        message_lines = [
            "Detected labelling mismatches larger than 1 sample.",
            "Please fix the input files before continuing:",
        ]
        for path, mismatch in large_mismatch_files:
            message_lines.append(f"- {path} (mismatch {mismatch} samples)")
        raise ValueError("\n".join(message_lines))


# Back-compat alias — callers that still import mat_to_csv keep working but now produce parquet.
def mat_to_csv(input_directory_path, output_directory_path, sampling_rate, sleep_stage_resolution):
    """Deprecated alias for mat_to_signal_tables (still writes parquet now)."""
    return mat_to_signal_tables(
        input_directory_path,
        output_directory_path,
        sampling_rate,
        sleep_stage_resolution,
        output_format="parquet",
    )


if __name__ == "__main__":
    print("Starting main block...")
    train_test_or_to_score = input("Enter dataset type ('train', 'test', or 'to_score'): ")
    input_directory_path = input(
        f"Enter the input directory path for {train_test_or_to_score} files, without quotes: "
    )
    output_directory_path = input(
        f"Enter the output directory path for {train_test_or_to_score} signal tables, without quotes: "
    )
    sampling_rate = int(input("Enter the sampling rate in Hz (e.g., 512): "))
    sleep_stage_resolution = int(input("Enter the sleep stage resolution in seconds (e.g., 10): "))
    fmt = (input("Output format ('parquet' [default] or 'csv'): ").strip() or "parquet")
    mat_to_signal_tables(
        input_directory_path,
        output_directory_path,
        sampling_rate,
        sleep_stage_resolution,
        output_format=fmt,
    )

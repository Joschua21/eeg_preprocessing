# import required libraries
import numpy as np
import pandas as pd
import h5py
import os
import glob


def mat_to_csv(input_directory_path, output_directory_path, sampling_rate, sleep_stage_resolution):
    '''
    Converts .mat files extraced from Spike2 into .csv files to be used in the somnotate pipeline.
    Checks length of upsampled sleep stages and EEG data to ensure they match. If they do not match, the function will truncate or pad the sleep stages to match the length of the EEG data.
    Inputs:
    directory_path: str, path to the directory containing the .mat files
    output_directory_path: str, path to the directory where the .csv files should be saved
    sampling_rate: int, Hz (samples per second)
    sleep_stage_resolution: int, seconds 

    '''
    mat_files = glob.glob(os.path.join(input_directory_path, '*.mat'))
    print(f"Found .mat files: {mat_files}")  # Debugging print for found files

    large_mismatch_files = []

    for file_path in mat_files:
        # Extract the base filename without the extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        with h5py.File(file_path, 'r') as raw_data:
            print(f'Processing file: {file_path}')

            # Determine the sampling rate based on the filename
            sampling_rate = sampling_rate # Hz (samples per second)

            # Define the sleep stage resolution
            sleep_stage_resolution = sleep_stage_resolution # seconds

            # Initialize variables to store data
            eeg1_data, eeg2_data, emg_data, sleep_stages = None, None, None, None

            # Iterate over all keys in the HDF5 file to extract data
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
            
        # Check if the data was found
            if eeg1_data is not None:
                print("EEG1 data extracted successfully.")
            if eeg2_data is not None:
                print("EEG2 data extracted successfully.")
            if emg_data is not None:
                print("EMG data extracted successfully.")
            if sleep_stages is not None:
                print("Sleep stage data extracted successfully.")

            # format data for saving to a CSV file
            eeg1_flattened = eeg1_data.flatten()
            eeg2_flattened = eeg2_data.flatten()
            emg_flattened = emg_data.flatten()
            assert eeg1_flattened.shape == eeg2_flattened.shape == emg_flattened.shape, "The flattened shapes of the EEG and EMG data do not match"

            # upsample the sleep stages to match the resolution of the EEG and EMG data
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
                assert len(upsampled_sleep_stages) == len(eeg1_flattened), "Length of upsampled sleep stages does not match length of EEG data after truncation" 
                print("Length of upsampled sleep stages matches length of EEG data after truncation")


            extracted_data = {
                'sleepStage': upsampled_sleep_stages,
                'EEG1': eeg1_flattened,
                'EEG2': eeg2_flattened,
                'EMG': emg_flattened

            }

            df = pd.DataFrame(extracted_data)

            # Save DataFrame to a CSV file
            if not os.path.exists(output_directory_path):
                os.makedirs(output_directory_path)
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


if __name__ == "__main__":
    print("Starting main block...")
    train_test_or_to_score = input("Enter dataset type ('train', 'test', or 'to_score'): ")
    input_directory_path = input(
        f"Enter the input directory path for {train_test_or_to_score} files, without quotes: "
    )
    output_directory_path = input(
        f"Enter the output directory path for {train_test_or_to_score} CSV files, without quotes (e.g., Z:/somnotate/to_score_set/to_score_csv_files): "
    )
    sampling_rate = int(input("Enter the sampling rate in Hz (e.g., 512): "))
    sleep_stage_resolution = int(input("Enter the sleep stage resolution in seconds (e.g., 10): "))
    mat_to_csv(
        input_directory_path,
        output_directory_path,
        sampling_rate,
        sleep_stage_resolution=sleep_stage_resolution,
    )


        


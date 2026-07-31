import re
import glob
import os 
import shutil 

def adjust_delimiters_in_txt_files(directory_path):
    '''
    Adjust delimiters to four spaces in .txt files in the specified directory.
    Input:
    - directory_path: str, the directory path containing the .txt files.
    Output:
    - .txt files with delimiters adjusted to four spaces.
    - the original files backed up with the '_original' suffix.
    '''

    txt_files = glob.glob(os.path.join(directory_path, '*.txt'))

    for txt_file in txt_files:
        basename, extension = os.path.splitext(txt_file)
        if '_original' not in basename:
            backup_file = f"{basename}_original{extension}"
            shutil.copyfile(txt_file, backup_file) 
            print(f"Original file {txt_file} copied to {basename}_original{extension}")

        with open(txt_file, 'r') as file:
            lines = file.readlines()

        cleaned_lines = []
        for line in lines:
            parts = re.split(r'\s+', line.strip()) # Split the line by whitespace into labels and values
            cleaned_line = '    '.join(parts) # Join the parts back together with four spaces
            cleaned_lines.append(cleaned_line)

        # Write the cleaned lines to the same file
        with open(txt_file, 'w') as file:
            file.write('\n'.join(cleaned_lines))

    print("Delimiters adjusted successfully.")

if __name__ == "__main__":
    # Prompt the user to input the directory path
    directory_path = input("Enter the directory path containing the Visbrain .txt files without quotes: ")
    adjust_delimiters_in_txt_files(directory_path)


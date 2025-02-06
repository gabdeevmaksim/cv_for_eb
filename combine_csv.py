import pandas as pd
import glob
import os

def append_files(input_dir, output_dir, combinations):
    for combination in combinations:
        gaia_dfs = []
        i_dfs = []
        
        # Find all files that match the combination
        file_paths = glob.glob(f"{input_dir}/*{combination}*.csv")
        print(f"Found {len(file_paths)} files for combination '{combination}'")
        
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            # Read each file into a DataFrame
            try:
                df = pd.read_csv(file_path)
                if 'gaia' in file_path:
                    gaia_dfs.append(df)
                elif '_I' in file_path:
                    i_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if gaia_dfs:
            # Concatenate all Gaia DataFrames
            combined_gaia_df = pd.concat(gaia_dfs, ignore_index=True)
            
            # Check if output_dir exists and create if not
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save the combined Gaia DataFrame to a new CSV file
            output_file = os.path.join(output_dir, f"combined_{combination}_gaia.csv")
            combined_gaia_df.to_csv(output_file, index=False)
            print(f"Combined Gaia dataset for {combination} created successfully at {output_file}")
        
        if i_dfs:
            # Concatenate all I DataFrames
            combined_i_df = pd.concat(i_dfs, ignore_index=True)
            
            # Save the combined I DataFrame to a new CSV file
            output_file = os.path.join(output_dir, f"combined_{combination}_I.csv")
            combined_i_df.to_csv(output_file, index=False)
            print(f"Combined I dataset for {combination} created successfully at {output_file}")

# Directory to search for files
input_dir = '/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries'

# Output directory
output_dir = '/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries/combined_datasets'

# Combinations to search for
combinations = ['detached_spots', 'detached_nospots', 'overcontact_spots', 'overcontact_nospots']

# Append files based on the combinations
append_files(input_dir, output_dir, combinations)
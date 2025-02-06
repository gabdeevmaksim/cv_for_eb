import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import glob

def create_sample(input_csv, output_dir, sample_size=5000, random_state=69):
    """
    Create a sample from a CSV file based on specific conditions and save it to a new CSV file.
    Parameters:
    input_csv (str): Path to the input CSV file.
    output_dir (str): Directory where the output CSV file will be saved.
    output_prefix (str): Prefix for the output CSV file name.
    sample_size (int, optional): Number of samples to generate. Default is 5000.
    random_state (int, optional): Seed for random number generator. Default is 42.
    Returns:
    None
    The function performs the following steps:
    1. Loads the CSV file using Dask.
    2. Filters the DataFrame based on specific conditions.
    3. Samples random rows from the filtered DataFrame.
    4. Processes each row to add noise and potential outliers.
    5. Merges the sampled DataFrame with the original DataFrame to retain all columns.
    6. Saves the sampled DataFrame to a new CSV file in the specified output directory.
    7. Prints a success message with the path to the output file.

    # Example usage
    create_sample(
    input_csv='/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries/Syntetic LC/detached/detached_spots_period(0-1)_I.csv',
    output_dir='/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries',
    output_prefix='sample'
    )
    """
    
    # Load the CSV file using Dask
    df = dd.read_csv(input_csv)

    # Filter the DataFrame based on the given conditions
    df = df[(df['period'] < 1.5) & (df['t1/t2'] > 1.3) & (df['inc'] > 50.) & (df['pot1'] < 3.) & (df['pot2'] < 4.5)]

    # Sample random rows
    sampled_df = df.sample(frac=(sample_size * 1.1) / len(df), random_state=random_state)

    # Compute the sampled DataFrame to get a Pandas DataFrame
    sampled_df = sampled_df.compute()

    def process_row(row):
        flux_o = list(row[:100])
        noise1 = np.random.normal(0, 0.01, 100)  # 0.0, 0.01, 0.005
        flux = flux_o + noise1
        is_outlier = np.random.randint(0, 5, 1)
        if is_outlier > 2:
            outlier_val = np.random.normal(0, 0.3, 1)[0]
            outlier_pos = np.random.randint(0, 100, 1)[0]
            flux[outlier_pos] += outlier_val
        flux = flux / np.max(flux) # Normalize the flux values on max
        if any(flux < 0):
            return None  # Return None if there are negative values
        row[:100] = flux
        return row

    # Apply the process_row function and filter out rows with negative flux values
    sampled_df = sampled_df.apply(process_row, axis=1).dropna()

    # Ensure the final sample size is as specified
    sampled_df = sampled_df.sample(n=sample_size, random_state=random_state)

    # Create the output file name
    input_filename = os.path.splitext(os.path.basename(input_csv))[0]
    output_file = f"{output_dir}/{input_filename}_{sample_size}.csv"

    # Save the sampled DataFrame to a new CSV file
    sampled_df.to_csv(output_file, index=False)

    print(f"Sampled dataset created successfully at {output_file}")


# Directory to search for files
base_dir = '/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries'

# Find all files that have 'gaia' or 'I' in the name
file_paths = glob.glob(f"{base_dir}/**/*gaia*.csv", recursive=True) + glob.glob(f"{base_dir}/**/*I*.csv", recursive=True)

# Process each file based on the conditions
for file_path in file_paths:
    if 'detached' in file_path:
        sample_size = 1375
    elif 'overcontact' in file_path:
        sample_size = 2750
    else:
        continue  # Skip files that do not match the conditions

    create_sample(file_path, base_dir, sample_size)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import re

def create_polar_hexbin(phase, flux, r_min=0.2, r_max=1., image_size=224):
    """
    Generates a polar hexbin plot from a vector and phase.

    Args:
        vector: The input vector (NumPy array) containing radius values (0 to 1).
        phase: The phase value (in turns, where 1 turn is a full circle).
        image_size: The size of the output image in pixels (square).

    Returns:
        A NumPy array representing the hexbin plot image.
    """
    # remove some points randomly
    num_points_to_remove = np.random.randint(60, 80)
    random_indices = np.random.permutation(len(phase))[:num_points_to_remove]
    mask = np.ones(len(phase), dtype=bool)  # Create a boolean mask
    mask[random_indices] = False  # Set mask to False for selected rows
    phase = phase[mask]
    flux = flux[mask]

    # Calculate angle based on phase (in degrees)
    angle = phase * 360 - 90  # Subtract 90 to start from vertical line

    flux_min, flux_max = flux.min(), flux.max()
    r = r_max - (r_max - r_min) * (flux_max - flux) / (flux_max - flux_min)

    # Calculate x and y coordinates in radial coordinates
    x = r * np.cos(np.deg2rad(angle))
    y = r * np.sin(np.deg2rad(angle))

    # Create a figure and axes with the desired image size
    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)

    # Create the hexbin plot
    hb = ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1)  # Extent for radial coordinates

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    plt.xticks([])
    plt.yticks([])
    ax.grid(False)
    fig.tight_layout(pad=0)

    # Save the figure as a NumPy array with specified size
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    image = Image.open(buf).resize((image_size, image_size), Image.LANCZOS)
    image_array = np.array(image)
    plt.close(fig)
    del fig

    return image_array

def process_files(file_path):
    # Dictionary to store DataFrames for each combination
    dfs = {
        'detached_gaia': [],
        'detached_I': [],
        'overcontact_gaia': [],
        'overcontact_I': []
    }
    file_names = {
        'detached_gaia': 'detached.+gaia',
        'detached_I': 'detached.+I',
        'overcontact_gaia': 'overcontact.+gaia',
        'overcontact_I': 'overcontact.+I'
    }

    for name, pattern in file_names.items():
        for filename in os.listdir(file_path):
            if re.search(pattern, filename):
                try:
                    df = pd.read_csv(os.path.join(file_path, filename))
                    dfs[name].append(df)  # Append to the corresponding list
                except pd.errors.EmptyDataError:
                    print(f"Skipping empty file: {filename}")
                except pd.errors.ParserError:
                    print(f"Skipping file with parsing error: {filename}")

    # Concatenate DataFrames for each combination
    for name in dfs:
        dfs[name] = pd.concat(dfs[name], ignore_index=True)

    return dfs  # Return the dictionary of DataFrames

def generate_images(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for index, row in df.iterrows():
        vector = row[:100].values
        phase = np.linspace(0, 1, 100)
        image_array = create_polar_hexbin(phase, vector)
        image = Image.fromarray(image_array)
        image.save(os.path.join(output_dir, f"{index}.png"))

file_path = '/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries/test_sample/combined_datasets'
dfs = process_files(file_path)

# Generate images for each category
for category, df in dfs.items():
    if not df.empty:
        upper_dir = category.split('_')[1]
        class_dir = f"class_{0 if 'overcontact' in category else 1}"
        output_dir = os.path.join(file_path, upper_dir, class_dir)
        generate_images(df, output_dir)
    else:
        print(f"No data found for category: {category}")


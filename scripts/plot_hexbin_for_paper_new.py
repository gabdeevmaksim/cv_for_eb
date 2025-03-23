import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import Arc
import matplotlib.gridspec as gridspec

def process_row(row):
    """
    Process a single row by adding noise, introducing potential outliers, 
    and normalizing the flux values.
    """
    flux_o = list(row[:100])
    noise1 = np.random.normal(0, 0.001, 100)  # Add Gaussian noise
    flux = flux_o + noise1
    is_outlier = np.random.randint(0, 5, 1)
    if is_outlier > 2:
        outlier_val = np.random.normal(0, 0.3, 1)[0]
        outlier_pos = np.random.randint(0, 100, 1)[0]
        flux[outlier_pos] += outlier_val
    flux = flux / np.max(flux)  # Normalize the flux values
    if any(flux < 0):
        return None  # Return None if there are negative values
    row[:100] = flux
    return row

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
    ax.set_axis_off()

    # Save the figure as a NumPy array with specified size
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    image = Image.open(buf).resize((image_size, image_size), Image.LANCZOS)
    image_array = np.array(image)
    plt.close(fig)
    del fig

    return image_array

def choose_and_plot_light_curves(dataset_dir, filters={}, num_curves=5, save_plots=False):
    """
    Chooses light curves from the dataset based on filters and plots them.

    Args:
        dataset_dir: Directory containing the CSV dataset files.
        filters: Dictionary containing filters for 'type', 'spots', and 'source'.
        num_curves: Number of light curves to plot. Defaults to 5 random curves if not provided.
        save_plots: Boolean indicating whether to save the plots to files.
    """
    # Add the type folder to the dataset directory
    dataset_dir = os.path.join(dataset_dir, filters['type'])

    # Find the file that matches all filter values in the filename
    matching_file = None
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.csv'):
            match = True
            for key, value in filters.items():
                if f"{value}" not in filename:
                    match = False
                    break
            if match:
                matching_file = os.path.join(dataset_dir, filename)
                break

    if not matching_file:
        raise ValueError("No file matches the given filters.")

    # Load the matching dataset
    df = pd.read_csv(matching_file)

    # Select random light curves if num_curves is not provided
    if num_curves > len(df):
        num_curves = len(df)
    selected_curves = df.sample(n=num_curves)

    # Plot each selected light curve
    for index, row in selected_curves.iterrows():
        processed_row = process_row(row)
        vector = np.array(processed_row[:100], dtype=float)
        phase = np.linspace(0, 1, len(vector))

        # Extract information from the last 8 columns of the row
        additional_info = row[-8:].to_dict()
        info_text = ', '.join([f"{key}: {value}" for key, value in filters.items()])

        image_size = 224

        # Create a figure with a specific gridspec layout
        fig = plt.figure(figsize=(10, 5))

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])

        # Standard scatter plot
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(phase, vector)  # Change the size of the scatter plot points
        ax1.set_aspect('auto')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Normalized Flux')

        # Set left tick labels with two digits precision
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        # Invert the y-axis ticks to look inside
        ax1.tick_params(axis='y', direction='in')
        ax1.tick_params(axis='x', direction='in')

        #ax1.set_title("Standard Scatter Plot")

        # Polar hexbin plot
        ax2 = fig.add_subplot(gs[1])
        image_array = create_polar_hexbin(phase, vector)
        ax2.imshow(image_array)
        ax2.set_aspect('equal')
        #ax2.set_title("Polar Hexbin Plot")

        # Set custom ticks
        ticks = [0.5 * image_size]
        tick_labels_bottom = ['0.5']
        tick_labels_left = ['0.75']

        
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(tick_labels_bottom)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.tick_params(axis='x', direction='in', pad=1)

        ax2.set_yticks(ticks)
        ax2.set_yticklabels(tick_labels_left)
        ax2.yaxis.set_ticks_position('left')
        ax2.tick_params(axis='y', direction='in', pad=1)

        # Add a secondary x-axis on top with custom ticks
        secondary_top_ax = ax2.secondary_xaxis('top')
        secondary_top_ax.set_xticks([0.5 * image_size])
        secondary_top_ax.set_xticklabels(['1'])
        secondary_top_ax.tick_params(axis='x', direction='in', pad=1)

        secondary_right_ax = ax2.secondary_yaxis('right')
        secondary_right_ax.set_yticks([0.5 * image_size])
        secondary_right_ax.set_yticklabels(['0.25'])
        secondary_right_ax.tick_params(axis='y', direction='in', pad=1)

        # Draw vertical arrow from center and label 'r'
        plt.annotate('', xy=(112, 0), xytext=(112, 112),
             arrowprops=dict(facecolor='black', edgecolor='none', shrink=0.01, width=1, headwidth=2, headlength=4))
        plt.text(0.45*image_size, 0.1*image_size, 'r', color='red', fontsize=12)

        plt.annotate('', xy=(152, 5), xytext=(112, 112),
                 arrowprops=dict(facecolor='black', edgecolor='none', shrink=0.01, width=1, headwidth=2, headlength=4))

        # Draw curved arrow between the two arrows and label 'φ'
        arc = Arc((0.5*image_size, 0.5*image_size), 0.7*image_size, 0.7*image_size, theta1=270, theta2=291, color='green')
        plt.gca().add_patch(arc)
        plt.text(0.55*image_size, 0.13*image_size, r'$\phi$', color='green', fontsize=12)
        plt.tight_layout() 

        if save_plots:
            # Save the plots to a file
            plot_filename = f"light_curve_{filters['type']}_{filters['spots']}_{filters['band']}_{index}.png"
            plt.savefig(plot_filename)
            print(f"Saved plot to {plot_filename}")
        else:    
            # Display the plots
            plt.show()

if __name__ == "__main__":
    dataset_directory = "/Users/wera/Max_astro/Slovakia/elisa_on_a_server/ML-EclipsingBinaries/Syntetic LC/"
    types = ['detached'] #['overcontact', 
    spots_options = ['nospots'] #['spots', 
    band = 'gaia'

    for t in types:
        for spots in spots_options:
            filters = {
                'type': t,
                'spots': spots,
                'band': band
            }
            choose_and_plot_light_curves(dataset_directory, filters, num_curves=1, save_plots=True)
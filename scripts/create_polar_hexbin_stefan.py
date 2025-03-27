import numpy as np

import matplotlib.pyplot as plt

def create_polar_hexbin(phase, flux, r_min=0.2, r_max=1., image_size=224, save_name=None):
    """
        Generates a polar hexbin plot based on phase and flux values.

        phase (float): The phase value in turns (where 1 turn is a full circle).
        flux (numpy.ndarray): The input flux values.
        r_min (float, optional): The minimum radius value. Defaults to 0.2.
        r_max (float, optional): The maximum radius value. Defaults to 1.0.
        image_size (int, optional): The size of the output image in pixels (square). Defaults to 224.
        save_name (str, optional): The file name to save the plot. If None, the plot is not saved. Defaults to None.

        bool: True if the plot is successfully created.
    """

    angle = phase * 360 - 90 # Subtract 90 to start from vertical line

    flux_min, flux_max = flux.min(), flux.max()
    r = r_max - (r_max - r_min) * (flux_max - flux) / (flux_max - flux_min)

    # Calculate x and y coordinates in radial coordinates
    x = r * np.cos(np.deg2rad(angle))
    y = r * np.sin(np.deg2rad(angle))

    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)

    # Create the hexbin plot
    hb = ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    plt.xticks([])
    plt.yticks([])
    ax.grid(False)
    fig.tight_layout(pad=0)
    plt.savefig(save_name, dpi=100, bbox_inches='tight', pad_inches = 0.1)
    plt.close(fig)
    del fig

    return True
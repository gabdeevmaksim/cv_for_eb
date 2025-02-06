import numpy as np
import io
from PIL import Image

import matplotlib.pyplot as plt

def create_polar_hexbin(vector, phase, period=1, image_size=224):
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

    # Calculate x and y coordinates in radial coordinates
    x = vector * np.cos(np.deg2rad(angle))
    y = vector * np.sin(np.deg2rad(angle))

    # Create a figure and axes with the desired image size
    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)

    # Create the hexbin plot
    hb = ax.hexbin(x, y, gridsize=25, cmap="gray", extent=[-1, 1, -1, 1])  # Extent for radial coordinates

    # Remove axes, margins, and grid
    ax.set_axis_off()
    ax.grid(False)
    fig.tight_layout(pad=0)

    # Save the figure as a NumPy array with specified size
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).resize((image_size, image_size), Image.LANCZOS)
    image_array = np.array(image)
    plt.close(fig)

    return image_array
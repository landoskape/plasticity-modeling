import numpy as np
from typing import Optional


def create_gabor(
    orientation: float,
    width: float = 0.6,
    envelope: float = 0.5,
    gamma: float = 1.5,
    halfsize: int = 25,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Create a Gabor pattern with specified parameters.

    Parameters
    ----------
    orientation : float
        The orientation angle in radians (3π/4 = vertical)
        This is weird like this to match the edge orientation in the
        source_population Gabor!
    width : float
        The width of the sinusoidal grating (relative to fullsize)
    envelope : float
        The standard deviation of the Gaussian envelope (relative to fullsize)
    gamma : float
        The spatial aspect ratio of the envelope
    halfsize : int
        The number of pixels in half of the grid
    phase : float
        The phase of the sinusoidal grating

    Returns
    -------
    np.ndarray
        A 2D array containing the Gabor pattern
    """
    orientation += np.pi / 4
    orientation = -1 * orientation

    # Convert relative parameters to pixel units
    width = width * halfsize * 2
    envelope = envelope * halfsize * 2

    # Create coordinate grid
    x = np.linspace(-halfsize, halfsize, 2 * halfsize + 1)
    y = np.linspace(-halfsize, halfsize, 2 * halfsize + 1)
    x_grid, y_grid = np.meshgrid(x, y)

    # Rotate coordinates
    x_rot = x_grid * np.cos(orientation) + y_grid * np.sin(orientation)
    y_rot = -x_grid * np.sin(orientation) + y_grid * np.cos(orientation)
    y_rot = y_rot / gamma

    # Create Gabor pattern
    gaussian = np.exp(-(x_rot**2 + y_rot**2) / (envelope**2))
    grating = np.cos(2 * np.pi * x_rot / width + phase)
    return gaussian * grating


def create_gabor_grid(
    orientations: np.ndarray,
    spacing: int = 1,
    gabor_params: Optional[dict] = {},
) -> np.ndarray:
    """
    Create a grid of Gabor patterns from a 3x3 array of orientations.

    Parameters
    ----------
    orientations : np.ndarray
        A 3x3 array of orientation angles in radians
    spacing : int
        Number of pixels to add between Gabors
    gabor_params : dict, optional
        Additional parameters to pass to create_gabor()

    Returns
    -------
    np.ndarray
        A 2D array containing the grid of Gabor patterns
    """
    if orientations.shape != (3, 3):
        raise ValueError("orientations must be a 3x3 array")

    # Create individual Gabors
    gabors = [[None for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            gabors[i][j] = create_gabor(orientations[i, j], **gabor_params)

    return stitch_gabor_grid(gabors, spacing)


def stitch_gabor_grid(gabors, spacing=1):
    # Calculate output size
    gabor_size = gabors[0][0].shape[0]
    output_size = 3 * gabor_size + 2 * spacing
    output = np.zeros((output_size, output_size))

    # Place Gabors in grid
    for i in range(3):
        for j in range(3):
            y_start = i * (gabor_size + spacing)
            x_start = j * (gabor_size + spacing)
            output[y_start : y_start + gabor_size, x_start : x_start + gabor_size] = gabors[i][j]

    return output

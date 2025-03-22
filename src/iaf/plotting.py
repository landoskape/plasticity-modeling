import numpy as np
from typing import Optional, List


def create_gabor(
    orientation: float,
    width: float = 0.6,
    envelope: float = 0.5,
    gamma: float = 1.5,
    halfsize: int = 25,
    phase: float = 0.0,
) -> np.ndarray:
    """Create a Gabor pattern with specified parameters.

    This function generates a 2D Gabor pattern, which is the product of a
    sinusoidal grating and a Gaussian envelope. The pattern is commonly used
    to model the receptive fields of simple cells in the visual cortex.

    Parameters
    ----------
    orientation : float
        The orientation angle in radians (3π/4 = vertical).
        Note: This is offset by π/4 and negated internally to match the
        edge orientation in the source_population Gabor.
    width : float, optional
        The width of the sinusoidal grating (relative to fullsize),
        default is 0.6.
    envelope : float, optional
        The standard deviation of the Gaussian envelope (relative to fullsize),
        default is 0.5.
    gamma : float, optional
        The spatial aspect ratio of the envelope, default is 1.5.
    halfsize : int, optional
        The number of pixels in half of the grid, default is 25.
    phase : float, optional
        The phase of the sinusoidal grating, default is 0.0.

    Returns
    -------
    np.ndarray
        A 2D array of shape (2*halfsize+1, 2*halfsize+1) containing the
        Gabor pattern.
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
    """Create a grid of Gabor patterns from a 3x3 array of orientations.

    This function generates a grid of Gabor patterns with the orientations
    specified in the input array, with optional spacing between the patterns.

    Parameters
    ----------
    orientations : np.ndarray
        A 3x3 array of orientation angles in radians.
    spacing : int, optional
        Number of pixels to add between Gabors, default is 1.
    gabor_params : dict, optional
        Additional parameters to pass to create_gabor(), default is {}.

    Returns
    -------
    np.ndarray
        A 2D array containing the grid of Gabor patterns.

    Raises
    ------
    ValueError
        If orientations is not a 3x3 array.
    """
    if orientations.shape != (3, 3):
        raise ValueError("orientations must be a 3x3 array")

    # Create individual Gabors
    gabors = [[None for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            gabors[i][j] = create_gabor(orientations[i, j], **gabor_params)

    return stitch_gabor_grid(gabors, spacing)


def stitch_gabor_grid(gabors: List[List[np.ndarray]], spacing: int = 1) -> np.ndarray:
    """Stitch a 3x3 grid of Gabor patterns into a single array.

    Parameters
    ----------
    gabors : list of list of np.ndarray
        A 3x3 list of lists, where each element is a 2D Gabor pattern array.
    spacing : int, optional
        Number of pixels to add between Gabors, default is 1.

    Returns
    -------
    np.ndarray
        A 2D array containing the stitched grid of Gabor patterns.
    """
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

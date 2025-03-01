import math
import numpy as np
import torch


def save_figure(fig, path, **kwargs):
    """
    Save a figure with high resolution in png and svg formats
    """
    fig.savefig(path.with_suffix(".png"), dpi=300, **kwargs)
    fig.savefig(path.with_suffix(".svg"), **kwargs)


def make_rf_display(u, disp_buffer=2, flip_sign=False, background_value=-1.0):
    """
    Convert a weight matrix into a receptive field display

    u is a input_dim x num_cells matrix, where input_dim = im_length**2
    """
    num_cells, im_length = u.shape[1], int(math.sqrt(u.shape[0]))
    assert num_cells * im_length**2 == u.numel(), "Input dimension isn't a square"
    u = u.T.view(num_cells, im_length, im_length)

    # Determine rows and cols
    if math.floor(math.sqrt(num_cells)) ** 2 != num_cells:
        rows = int(math.sqrt(num_cells))
        cols = num_cells // rows
    else:
        rows = cols = int(math.sqrt(num_cells))

    # Create the display array
    disp_array = torch.full(
        (disp_buffer + rows * (im_length + disp_buffer), disp_buffer + cols * (im_length + disp_buffer)),
        background_value,
    )

    # Prepare indices for efficient assignment
    row_indices = torch.arange(im_length).unsqueeze(0).expand(rows, im_length)
    col_indices = torch.arange(im_length).unsqueeze(0).expand(cols, im_length)

    row_offsets = torch.arange(rows) * (im_length + disp_buffer)
    col_offsets = torch.arange(cols) * (im_length + disp_buffer)

    # Compute clim values for all cells at once
    u_reshaped = u.view(num_cells, -1)
    clim_values = torch.abs(u_reshaped).max(1).values

    if flip_sign:
        clim_signs = torch.sign(u_reshaped[torch.arange(num_cells), torch.argmax(torch.abs(u_reshaped), dim=1)])
        clim_values *= clim_signs

    # Normalize u_reshaped
    u_normalized = u / clim_values.view(-1, 1, 1)

    # Assign values to disp_array
    for i in range(rows):
        for j in range(cols):
            cell_idx = i * cols + j
            if cell_idx < num_cells:
                x_coord = disp_buffer + row_offsets[i] + row_indices[i]
                y_coord = disp_buffer + col_offsets[j] + col_indices[j]
                disp_array[x_coord[:, None], y_coord] = u_normalized[cell_idx]

    return disp_array


def beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j + 1 :: 2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

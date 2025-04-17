from typing import Any, Literal
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from dataclasses import dataclass


# Centimeters to inches
cm = 1 / 2.54


@dataclass(init=False, frozen=True)
class FigParams:
    """Plotting parameters for the project."""

    single_width: float = 8.5 * cm
    onepointfive_width: float = 11.4 * cm
    double_width: float = 17.4 * cm
    fontsize: float = 7
    smallfontsize: float = 5.5
    tinyfontsize: float = 4
    thicklinewidth: float = 1.4
    linewidth: float = 1.0
    thinlinewidth: float = 0.7
    scattersize: float = 6.0
    scatteralpha: float = 0.5
    markersize: float = 3.0
    tick_length: float = 2.5
    tick_width: float = 1
    tick_fontsize: float = 7
    spine_pos: float = -0.0175

    @classmethod
    def all_fig_params(cls) -> dict[str, Any]:
        return dict(
            layout="constrained",
            dpi=300,
            frameon=False,
        )

    @classmethod
    def kwargs_spines(cls) -> dict[str, Any]:
        return dict(
            x_pos=FigParams.spine_pos,
            y_pos=FigParams.spine_pos,
            spine_linewidth=FigParams.linewidth,
            tick_length=FigParams.tick_length,
            tick_width=FigParams.tick_width,
            tick_fontsize=FigParams.tick_fontsize,
        )


def add_group_legend(
    ax: plt.Axes,
    x: float,
    y_start: float,
    y_offset: float,
    y_extra: float = 0,
    ha: str = "center",
    va: str = "center",
    fontsize: float = FigParams.fontsize,
    label_type: Literal["normal", "nl", "short", "experimental", "tiny"] = "normal",
    extra_label_simple: str = "",
    extra_label_complex: str = "",
):
    y_proximal = y_start
    y_distal_simple = y_start + y_offset
    y_distal_complex = y_start + 2 * y_offset + y_extra

    attribute = dict(
        normal="label",
        nl="labelnl",
        short="shortlabel",
        experimental="experimental",
        tiny="tinylabel",
    )
    attr = attribute.get(label_type, None)
    if attr is None:
        raise ValueError(f"Invalid label_type: {label_type}. Choose from {list(attribute.keys())}.")

    # Get default labels
    label_proximal = getattr(Proximal, attr)
    label_simple = getattr(DistalSimple, attr)
    label_complex = getattr(DistalComplex, attr)

    # Add extra info to distal labels if requested
    label_simple += extra_label_simple
    label_complex += extra_label_complex

    # Plot them
    ax.text(x, y_proximal, label_proximal, color=Proximal.color, ha=ha, va=va, fontsize=fontsize)
    ax.text(x, y_distal_simple, label_simple, color=DistalSimple.color, ha=ha, va=va, fontsize=fontsize)
    ax.text(x, y_distal_complex, label_complex, color=DistalComplex.color, ha=ha, va=va, fontsize=fontsize)


def add_dpratio_legend(
    ax: plt.Axes,
    ratios: list[float],
    x: float,
    y: float,
    fontsize: float = FigParams.fontsize,
    cmap: str = "plasma_r",
    cmap_pinch: float = 0.25,
):
    num_ratios = len(ratios)
    cmap = colormaps[cmap]
    colors = [cmap(ii) for ii in np.linspace(cmap_pinch, 1 - cmap_pinch, num_ratios)]
    labels = [f"{(ratio-1.0)*100:1g}" for ratio in ratios]
    text = ax.text(x, y, "Distal-Complex", ha="right", va="top", color="k", fontsize=fontsize)
    text = ax.annotate("Depression (%)", xycoords=text, xy=(1, 0), fontsize=fontsize, color="k", va="top", ha="right")
    text = ax.annotate(labels[-1], xycoords=text, xy=(1, 0), fontsize=fontsize, color=colors[-1], va="top", ha="right")
    kwargs = dict(fontsize=fontsize, va="bottom", ha="right")
    for ii in range(num_ratios - 2, -1, -1):
        text = ax.annotate(", ", xycoords=text, xy=(0, 0), color="k", **kwargs)
        text = ax.annotate(labels[ii], xycoords=text, xy=(0, 0), color=colors[ii], **kwargs)


@dataclass(init=False, frozen=True)
class Proximal:
    """Plotting properties for proximal dendritic sites.

    These are intended to be constants and used throughout the codebase
    without change - so the optimal usage is to call as class methods.

    Example usage:
    >>> plt.plot(..., color=Proximal.color, label=Proximal.label)
    """

    color: str = "black"
    tinylabel: str = "p"
    shortlabel: str = "prox"
    label: str = "proximal"
    labelnl: str = "proximal"
    experimental: str = "proximal"


@dataclass(init=False, frozen=True)
class DistalSimple:
    """Plotting properties for distal simple dendritic sites.

    These are intended to be constants and used throughout the codebase
    without change - so the optimal usage is to call as class methods.

    Use labelnl for labels that look better with a new line instead
    of a hyphen.

    Example usage:
    >>> plt.plot(..., color=DistalSimple.color, label=DistalSimple.label)
    """

    # Color options:
    # - cornflowerblue
    # - blueviolet
    # - darkmagenta
    # - darkorchid
    # - darkviolet
    # - dodgerblue
    # - gray
    # - lightslategrey
    # - mediumpurple
    # - mediumorchid
    # - mediumslateblue
    # - mediumturquoise
    # - purple
    # - steelblue
    # - teal

    color: str = "teal"
    tinylabel: str = "d-s"
    shortlabel: str = "dist-simp"
    label: str = "distal-simple"
    labelnl: str = "distal\nsimple"
    experimental: str = "high $\Delta Ca_{AP}$"


@dataclass(init=False, frozen=True)
class DistalComplex:
    """Plotting properties for distal complex dendritic sites.

    These are intended to be constants and used throughout the codebase
    without change - so the optimal usage is to call as class methods.

    Use labelnl for labels that look better with a new line instead
    of a hyphen.

    Example usage:
    >>> plt.plot(..., color=DistalComplex.color, label=DistalComplex.label)
    """

    color: str = "blue"
    tinylabel: str = "d-c"
    shortlabel: str = "dist-comp"
    label: str = "distal-complex"
    labelnl: str = "distal\ncomplex"
    experimental: str = "low $\Delta Ca_{AP}$"


def save_figure(fig, path, **kwargs):
    """
    Save a figure with high resolution in png and svg formats
    """
    fig.savefig(path.with_suffix(".png"), dpi=300, **kwargs)
    fig.savefig(path.with_suffix(".svg"), dpi=300, **kwargs)


def make_rf_display(u, disp_buffer=2, flip_sign=False, background_value=-1.0):
    """
    Convert a weight matrix into a receptive field display

    u is a input_dim x num_cells matrix, where input_dim = im_length**2
    """
    try:
        import torch
    except ImportError:
        raise ImportError("This function requires torch to be installed.")

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


def errorPlot(x, data, axis=-1, se=False, ax=None, handle_nans=True, **kwargs):
    """
    convenience method for making a plot with errorbars
    kwargs go into fill_between and plot, so they have to work for both...
    to make that more flexible, we could add a list of kwargs that work for
    one but not the other and pop them out as I did with the 'label'...
    """
    mean = np.nanmean if handle_nans else np.mean
    std = np.nanstd if handle_nans else np.std
    if handle_nans:
        num_valid_points = np.sum(~np.isnan(data), axis=axis)
    else:
        num_valid_points = data.shape[axis]
    if ax is None:
        ax = plt.gca()
    meanData = mean(data, axis=axis)
    correction = np.sqrt(num_valid_points) if se else 1
    errorData = std(data, axis=axis) / correction
    fillBetweenArgs = kwargs.copy()
    fillBetweenArgs.pop("label", None)
    if "edgecolor" not in fillBetweenArgs:
        fillBetweenArgs["edgecolor"] = "none"
    ax.fill_between(x, meanData + errorData, meanData - errorData, **fillBetweenArgs)
    kwargs.pop("alpha", None)
    ax.plot(x, meanData, **kwargs)


def format_spines(
    ax,
    x_pos=0.0,
    y_pos=0.0,
    xbounds=None,
    ybounds=None,
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    xrotation=0,
    yrotation=0,
    xrotationmode=None,
    yrotationmode=None,
    xva="center",
    yva="center",
    xha="center",
    yha="center",
    spine_linewidth=1,
    tick_length=6,
    tick_width=1,
    tick_direction="out",
    tick_fontsize=None,
):
    """
    Format a matplotlib axis to have separated spines with data offset from axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to format
    x_pos : int or float, optional
        The fractional value of the y-axis to offset the x-axis
    y_pos : int or float, optional
        The fractional value of the x-axis to offset the y-axis
    xbounds : tuple, optional
        The x-axis bounds as (min, max)
    ybounds : tuple, optional
        The y-axis bounds as (min, max)
    xticks : list or array, optional
        Custom x-axis tick positions
    yticks : list or array, optional
        Custom y-axis tick positions
    xlabels : list or array, optional
        Custom x-axis tick labels
    ylabels : list or array, optional
        Custom y-axis tick labels
    xrotation : int or float, optional
        Rotation angle for x-axis tick labels
    yrotation : int or float, optional
        Rotation angle for y-axis tick labels
    xrotationmode : str, optional
        Rotation mode for x-axis tick labels
    yrotationmode : str, optional
        Rotation mode for y-axis tick labels
    xva : str, optional
        Vertical alignment for x-axis tick labels
    yva : str, optional
        Vertical alignment for y-axis tick labels
    xha : str, optional
        Horizontal alignment for x-axis tick labels
    yha : str, optional
        Horizontal alignment for y-axis tick labels
    spine_linewidth : int or float, optional
        Width of the axis spines
    tick_length : int or float, optional
        Length of the tick marks
    tick_width : int or float, optional
        Width of the tick marks
    tick_direction : str, optional
        Direction of the tick marks
    tick_fontsize : int or float, optional
        Font size of the tick labels

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The formatted axis
    """
    # Hide the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set spine line width
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)

    # Move bottom spine down and left spine left
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]
    x_pos = x_pos * x_range + x_lims[0]
    y_pos = y_pos * y_range + y_lims[0]
    ax.spines["bottom"].set_position(("data", y_pos))
    ax.spines["left"].set_position(("data", x_pos))

    # Set axis limits if provided
    if xbounds is not None:
        ax.spines["bottom"].set_bounds(xbounds[0], xbounds[1])

    if ybounds is not None:
        ax.spines["left"].set_bounds(ybounds[0], ybounds[1])

    # Set custom ticks if provided
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlabels is not None:
        ax.set_xticklabels(xlabels, rotation=xrotation, rotation_mode=xrotationmode, ha=xha, va=xva)

    if yticks is not None:
        ax.set_yticks(yticks)
    if ylabels is not None:
        ax.set_yticklabels(ylabels, rotation=yrotation, rotation_mode=yrotationmode, ha=yha, va=yva)

    # Adjust tick appearance
    ax.tick_params(
        axis="both",
        which="major",
        direction=tick_direction,
        length=tick_length,
        width=tick_width,
        labelsize=tick_fontsize,
    )

    return ax


def add_dpratio_inset(
    ax: plt.Axes,
    inset_position: list[float],
    dpratio_colors: list[np.ndarray],
    dpratios: list[float] | np.ndarray,
    label: str = "Extra LTD (%)",
    fontsize: float = FigParams.tick_fontsize,
    reverse: bool = True,
    label_padding: float = -1,
):
    num_ratios = len(dpratio_colors)

    color_stack = np.stack(dpratio_colors)
    if reverse:
        color_stack = color_stack[::-1]

    inset = ax.inset_axes(inset_position)
    inset.imshow(
        color_stack[:, None],
        aspect="auto",
        extent=[0, 1, -0.5, num_ratios - 0.5],
    )
    for iratio, ratio in enumerate(dpratios):
        inset.text(
            0.5,
            iratio,
            f"{ratio*10:1g}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="w",
        )

    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_ylabel(label, fontsize=fontsize, labelpad=label_padding)
    for spine in inset.spines.values():
        spine.set_visible(False)

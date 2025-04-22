from typing import Optional, List
import numpy as np
from scipy.signal import filtfilt
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.typing import LineStyleType
from matplotlib.patches import FancyArrowPatch
from .analysis import get_norm_factor, get_groupnames, get_sigmoid_params, sigmoid
from .source_population import SourcePopulationGabor, vonmises
from ..plotting import FigParams, Proximal, DistalSimple, DistalComplex, beeswarm, format_spines
from ..conductance import NMDAR, VGCC
from ..schematics import Neuron, neuron_color_kwargs, create_dpratio_colors


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


def gabor_rgba(gabor: np.ndarray, cmap: str = "bwr", vmax_scale: float = 1, alpha_power: float = 1) -> np.ndarray:
    """Convert a Gabor pattern to RGBA format.

    Parameters
    ----------
    gabor : np.ndarray
        The Gabor pattern to convert.
    cmap : str, optional
        The colormap to use, default is "bwr".
    vmax_scale : float, optional
        How much to scale the maximum value of the colormap, default is 1.
    alpha_power : float, optional
        The power to use for the alpha channel, default is 1.

    Returns
    -------
    rgba : np.ndarray
        The Gabor pattern in RGBA format.
    """
    vmax = np.nanmax(np.abs(gabor)) * vmax_scale
    gabor = gabor / vmax
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = colormaps[cmap]
    rgba = cmap(norm(gabor))
    alpha = np.abs(gabor) ** alpha_power
    rgba[..., -1] = alpha
    rgba[np.isnan(gabor)] = 0
    return rgba


def create_gabor_grid(
    orientations: np.ndarray,
    spacing: int = 1,
    gabor_params: Optional[dict] = {},
    center_only: bool = False,
    highlight_edge: bool = False,
    highlight_magnitude: int = 1,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
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
    center_only : bool, optional
        If True, only the center Gabor will be generated, the other 8 will be `np.nan`.
    highlight_edge : bool, optional
        If True, the edge gabors will be highlighted by being multiplied by `highlight_magnitude`.
    highlight_magnitude : int, optional
        Non-edge gabors will be divided by this value if `highlight_edge` is True.

    Returns
    -------
    grid : np.ndarray
        A 2D array containing the stitched grid of Gabor patterns.
    gabors : list[list[np.ndarray]]
        A 3x3 list of lists, where each element is a 2D Gabor pattern array.

    Raises
    ------
    ValueError
        If orientations is not a 3x3 array.
    """
    if orientations.shape != (3, 3):
        raise ValueError("orientations must be a 3x3 array")

    multiplier = np.ones((3, 3))
    if highlight_edge:
        if np.any(np.isnan(orientations)):
            print("Warning: highlight_edge is True, but orientations array contains NaNs")
        else:
            # NOTE this requires SourcePopulationGabor.orientations to be sorted...
            stimuli = np.searchsorted(SourcePopulationGabor.orientations, orientations)
            edge0, edge1 = SourcePopulationGabor.stimulus_to_edge_positions(stimuli[1, 1])
            multiplier /= highlight_magnitude
            if stimuli[edge0] == stimuli[1, 1] and stimuli[edge1] == stimuli[1, 1]:
                multiplier[1, 1] = 1
                multiplier[edge0] = 1
                multiplier[edge1] = 1

    # Create individual Gabors
    gabors = [[np.nan for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            if not center_only or (i == 1 and j == 1):
                gabors[i][j] = create_gabor(orientations[i, j], **gabor_params) * multiplier[i, j]

    return stitch_gabor_grid(gabors, spacing), gabors


def weights_to_gabor(weights: np.ndarray, orientations: np.ndarray, spacing: int = 2, **params):
    """Convert a 4x9 array of weights to a 3x3 grid of Gabor patterns.

    Parameters
    ----------
    weights : np.ndarray
        A 4x9 array of weights.
    orientations : np.ndarray
        A 9-element array of orientations.
    spacing : int, optional
        The spacing between the Gabor patterns, default is 2.
    params : dict, optional
        Additional parameters to pass to create_gabor(), default is {}.

    Returns
    -------
    np.ndarray
        A 3x3 grid of Gabor patterns.
    """
    weights = weights.T.reshape(9, 4)
    gabors = [[None for _ in range(weights.shape[1])] for _ in range(weights.shape[0])]
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            gabor = weights[i, j] * create_gabor(orientation=orientations[j], **params)
            gabors[i][j] = gabor
        gabors[i] = np.sum(np.stack(gabors[i]), axis=0)
    gwidth = gabors[0].shape[0]
    gabors = np.stack(gabors).reshape(3, 3, gwidth, gwidth)
    return stitch_gabor_grid(gabors, spacing=spacing)


def stitch_gabor_grid(gabors: List[List[np.ndarray]], spacing: int = 1) -> np.ndarray:
    """Stitch a 3x3 grid of Gabor patterns into a single array.

    Parameters
    ----------
    gabors : list of list of np.ndarray
        A 3x3 list of lists, where each element is a 2D Gabor pattern array.
        If any element is `np.nan`, the corresponding area won't be filled in.
    spacing : int, optional
        Number of pixels to add between Gabors, default is 1.

    Returns
    -------
    np.ndarray
        A 2D array containing the stitched grid of Gabor patterns.
    """
    # Calculate output size
    gabor_size = gabors[1][1].shape[0]
    output_size = 3 * gabor_size + 2 * spacing
    output = np.zeros((output_size, output_size))

    # Place Gabors in grid
    for i in range(3):
        for j in range(3):
            if not np.all(np.isnan(gabors[i][j])):
                y_start = i * (gabor_size + spacing)
                x_start = j * (gabor_size + spacing)
                output[y_start : y_start + gabor_size, x_start : x_start + gabor_size] = gabors[i][j]

    return output


def overlay_empty_pixels_with_x(
    ax: plt.Axes,
    gabors: List[List[np.ndarray]],
    spacing: int = 1,
    x_extent_fraction: float = 0.5,
    color: str = "k",
    linewidth: float = FigParams.linewidth,
    linestyle: LineStyleType = (0, (1, 1)),
):
    gabor_size = gabors[1][1].shape[0]
    x_extent = x_extent_fraction * gabor_size
    x_kwargs = {"color": color, "linewidth": linewidth, "linestyle": linestyle}

    for i in range(3):
        for j in range(3):
            if np.all(np.isnan(gabors[i][j])):
                y_start = i * (gabor_size + spacing)
                x_start = j * (gabor_size + spacing)
                x_center = x_start + gabor_size / 2
                y_center = y_start + gabor_size / 2
                x_limits = [x_center - x_extent / 2, x_center + x_extent / 2]
                y_limits = [y_center - x_extent / 2, y_center + x_extent / 2]
                ax.plot(x_limits, y_limits, **x_kwargs)
                ax.plot(x_limits, [y_limits[1], y_limits[0]], **x_kwargs)


def build_ax_latent_correlation_demonstration(
    ax: plt.Axes,
    T: int = 250,
    N: int = 8,
    sigma_latent: int = 15,
    sigma_noise: int = 12,
    max_correlation: float = 0.8,
    y_latent: float = 0.5,
    y_offsets: float = -1.0,
    colormap: str = "Reds",
    latent_color: str = "black",
    text_offset_fraction: float = 0.1,
):
    corr_val = np.linspace(max_correlation, 0, N)
    b_latent = np.ones(sigma_latent) / sigma_latent
    b_noise = np.ones(sigma_noise) / sigma_noise

    latent = filtfilt(b_latent, 1, np.random.normal(0, 1, T), padtype=None).reshape(-1, 1)
    noise = filtfilt(b_noise, 1, np.random.normal(0, 1, (T, N)), padtype=None, axis=0)

    signal = latent * (corr_val**2) + noise * (1 - corr_val**2)
    signal = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0))
    latent = (latent - np.min(latent)) / (np.max(latent) - np.min(latent))

    text_position = -T * text_offset_fraction
    offsets = y_offsets * np.arange(N)
    cmap = colormaps[colormap]
    color_axis = np.linspace(1, 0, N + 4)[1 : 1 + N]
    colors = cmap(color_axis)
    ax.plot(
        range(T),
        y_latent + latent,
        color=latent_color,
        linewidth=FigParams.linewidth,
    )
    for isignal in range(N):
        ax.plot(range(T), latent + offsets[isignal], color=latent_color, alpha=0.3, linewidth=FigParams.thinlinewidth)
        ax.fill_between(
            range(T),
            latent[:, 0] + offsets[isignal],
            signal[:, isignal] + offsets[isignal],
            color=colors[isignal],
            alpha=0.3,
            linewidth=0,
        )
        ax.plot(range(T), signal[:, isignal] + offsets[isignal], color=colors[isignal], linewidth=FigParams.linewidth)
    ax.text(
        text_position,
        y_latent + 1.0,
        "Latent",
        ha="right",
        va="top",
        rotation=90,
        color=latent_color,
        fontsize=FigParams.fontsize,
    )
    ax.text(
        text_position,
        np.mean(offsets) + 0.5,
        "Correlated Inputs",
        ha="right",
        va="center",
        rotation=90,
        color=colors[0],
        fontsize=FigParams.fontsize,
    )
    return ax


def build_ax_corrcoef(
    ax_imshow: plt.Axes,
    ax_left: plt.Axes,
    ax_top: plt.Axes,
    results: dict,
    feature_colors: np.ndarray,
    source_name: str,
    rho_max: float,
    rho_min: float = 0,
):
    source_rates = results["source_rates"][source_name]
    source_intervals = results["source_intervals"][source_name]
    steps_per_second = int(1 / results["sim"].dt)
    seconds = results["weights"][0]["proximal"].shape[0]
    num_steps = seconds * steps_per_second
    stimulus = np.zeros((num_steps, source_rates.shape[1]))
    next_position = 0
    for interval, rate in zip(source_intervals, source_rates):
        stimulus[next_position : next_position + interval, :] = rate
        next_position += interval

    corrcoef = np.corrcoef(stimulus, rowvar=False)
    corrcoef = corrcoef - np.eye(corrcoef.shape[0])
    vmax = np.max(np.abs(corrcoef))

    extent = [0, 1, 0, 1]
    ax_imshow.imshow(
        corrcoef,
        cmap="bwr",
        aspect="auto",
        interpolation="none",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
        extent=extent,
    )
    ax_left.imshow(feature_colors[:, None], aspect="auto", extent=extent)
    ax_top.imshow(feature_colors[None], aspect="auto", extent=extent)
    ax_left.text(0.5, 0.5, "Input Features", ha="center", va="center", rotation=90, fontsize=FigParams.smallfontsize)
    ax_left.text(0.5, 1, rf"$\rho$={rho_max}", ha="center", va="top", rotation=90, fontsize=FigParams.smallfontsize)
    ax_left.text(0.5, 0, rf"$\rho$={rho_min}", ha="center", va="bottom", rotation=90, fontsize=FigParams.smallfontsize)
    ax_top.text(0.5, 0.5, "Input Features", ha="center", va="center", rotation=0, fontsize=FigParams.smallfontsize)
    ax_top.text(0, 0.5, rf"$\rho$={rho_max}", ha="left", va="center", rotation=0, fontsize=FigParams.smallfontsize)
    ax_top.text(1, 0.5, rf"$\rho$={rho_min}", ha="right", va="center", rotation=0, fontsize=FigParams.smallfontsize)
    return ax_imshow, ax_left, ax_top


def build_ax_trajectory(
    ax_psth: plt.Axes | None,
    ax_basal: plt.Axes,
    ax_dsimple: plt.Axes,
    ax_dcomplex: plt.Axes,
    results: dict,
    firing_rates: np.ndarray,
    ineuron: int,
    feature_colors: np.ndarray,
    cmap: str = "gray_r",
    feature_offset_fraction: float = 0.025,
    feature_width_fraction: float = 0.04,
):
    vmin = 0
    vmax = 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    weight_names = get_groupnames()
    norm_factor = get_norm_factor(results["sim"].neurons[ineuron], normalize=True)
    weight_trajs = [results["weights"][ineuron][name] / norm_factor[name] for name in weight_names]

    num_samples = weight_trajs[0].shape[0]
    num_features = weight_trajs[0].shape[1]
    offset_samples = int(num_samples * feature_offset_fraction)
    width_samples = int(num_samples * feature_width_fraction)

    # Apply normalization to get values between 0 and 1
    cmap = colormaps[cmap]
    rgb_trajs = [cmap(norm(wt)) for wt in weight_trajs]

    # Add white offset space
    white_offset = np.ones((offset_samples, num_features, 4))
    white_offset[:, :, -1] = 0
    rgb_trajs = [np.concatenate((white_offset, wt), axis=0) for wt in rgb_trajs]

    # Expand feature colors
    feature_colors = np.repeat(feature_colors[None], width_samples, axis=0)
    rgb_trajs = [np.concatenate((feature_colors, wt), axis=0) for wt in rgb_trajs]

    # Transpose so samples are on the x-axis
    rgb_trajs = [wt.transpose((1, 0, 2)) for wt in rgb_trajs]

    # Plot!
    extent = [-feature_offset_fraction - feature_width_fraction, 1, 0, 1]
    if ax_psth is not None:
        ax_psth.plot(np.linspace(0, 1, len(firing_rates)), firing_rates, color="k", linewidth=FigParams.thinlinewidth)
    for a, wt in zip([ax_basal, ax_dsimple, ax_dcomplex], rgb_trajs):
        a.imshow(wt, aspect="auto", interpolation="none", extent=extent, origin="upper")

    return ax_psth, ax_basal, ax_dsimple, ax_dcomplex


def build_ax_weight_summary(
    ax_basal: plt.Axes,
    ax_dsimple: plt.Axes,
    ax_dcomplex: plt.Axes,
    metadata: dict,
    weights: dict,
    cmap: str = "plasma_r",
    cmap_pinch: float = 0.25,
):
    num_ratios = len(metadata["dp_ratios"])
    colors = create_dpratio_colors(num_ratios, cmap=cmap, cmap_pinch=cmap_pinch)[0]
    num_inputs = metadata["base_config"].sources["excitatory"].num_inputs
    max_corr = metadata["base_config"].sources["excitatory"].max_correlation
    yvals = np.linspace(max_corr, 0, num_inputs)

    for iratio in range(num_ratios):
        proximal_mean = np.mean(weights["proximal"][iratio], axis=1)
        distal_simple_mean = np.mean(weights["distal-simple"][iratio], axis=1)
        distal_complex_mean = np.mean(weights["distal-complex"][iratio], axis=1)
        pmean = np.mean(proximal_mean, axis=0)
        dsmean = np.mean(distal_simple_mean, axis=0)
        dcmean = np.mean(distal_complex_mean, axis=0)
        pse = np.std(proximal_mean, axis=0)
        dse = np.std(distal_simple_mean, axis=0)
        dce = np.std(distal_complex_mean, axis=0)
        ax_basal.fill_betweenx(
            yvals,
            pmean - pse,
            pmean + pse,
            facecolor=colors[iratio],
            edgecolor="none",
            alpha=0.2,
        )
        ax_dsimple.fill_betweenx(
            yvals,
            dsmean - dse,
            dsmean + dse,
            facecolor=colors[iratio],
            edgecolor="none",
            alpha=0.2,
        )
        ax_dcomplex.fill_betweenx(
            yvals,
            dcmean - dce,
            dcmean + dce,
            facecolor=colors[iratio],
            edgecolor="none",
            alpha=0.2,
        )
        ax_basal.plot(pmean, yvals, color=colors[iratio], linewidth=FigParams.thicklinewidth)
        ax_dsimple.plot(dsmean, yvals, color=colors[iratio], linewidth=FigParams.thicklinewidth)
        ax_dcomplex.plot(dcmean, yvals, color=colors[iratio], linewidth=FigParams.thicklinewidth)

    return ax_basal, ax_dsimple, ax_dcomplex


def build_ax_weight_fits(
    ax_basal: plt.Axes,
    ax_dsimple: plt.Axes,
    ax_dcomplex: plt.Axes,
    metadata: dict,
    weights: dict,
    cmap: str = "plasma_r",
    cmap_pinch: float = 0.25,
    beeswarm_width: float = 0.3,
    share_ylim: bool = True,
):
    num_ratios = len(metadata["dp_ratios"])
    colors = create_dpratio_colors(num_ratios, cmap=cmap, cmap_pinch=cmap_pinch)[0]

    max_corr = metadata["base_config"].sources["excitatory"].max_correlation
    xvals = np.linspace(max_corr, 0, weights["proximal"].shape[-1])

    proximal_x0 = get_sigmoid_params(weights["proximal"], xvals)[1]
    distal_simple_x0 = get_sigmoid_params(weights["distal-simple"], xvals)[1]
    distal_complex_x0 = get_sigmoid_params(weights["distal-complex"], xvals)[1]

    if share_ylim:
        ylim_min = min([np.nanmin(d) for d in [proximal_x0, distal_simple_x0, distal_complex_x0]])
        ylim_max = max([np.nanmax(d) for d in [proximal_x0, distal_simple_x0, distal_complex_x0]])
        ylim_range = ylim_max - ylim_min
        ylim_min -= ylim_range * 0.02
        ylim_max += ylim_range * 0.02
    for iratio in range(num_ratios):
        for ax, data in zip([ax_basal, ax_dsimple, ax_dcomplex], [proximal_x0, distal_simple_x0, distal_complex_x0]):
            cdata = data[iratio].flatten()
            cxvals = beeswarm(cdata, 10)
            ax.scatter(
                iratio + cxvals * beeswarm_width,
                cdata,
                color=colors[iratio],
                alpha=0.5,
                s=FigParams.scattersize,
            )
    if share_ylim:
        for ax in [ax_basal, ax_dsimple, ax_dcomplex]:
            ax.set_ylim(ylim_min, ylim_max)
    return ax_basal, ax_dsimple, ax_dcomplex


def build_ax_sigmoid_example(
    ax: plt.Axes,
    results: dict,
    ineuron: int,
    color: str,
):
    # Get weights
    norm_factor = get_norm_factor(results["sim"].neurons[ineuron], normalize=True)
    proximal_weights = results["weights"][ineuron]["proximal"]
    num_samples = proximal_weights.shape[0]
    weights = np.mean(proximal_weights[-int(num_samples * 0.1) :], axis=0) / norm_factor["proximal"]

    max_correlation = results["sim"].source_populations["excitatory"].max_correlation
    xvals = np.linspace(max_correlation, 0, len(weights))
    prms = get_sigmoid_params(weights, xvals)

    ax.scatter(xvals, weights, color=color, s=FigParams.markersize / 2, zorder=5)
    ax.plot(xvals, sigmoid(xvals, *prms), color=color, linewidth=FigParams.linewidth, zorder=0)
    ax.scatter(prms[1], 0.5, color="black", s=2 * FigParams.scattersize, zorder=10)
    ax.annotate(
        r"$\sigma$ h-p ",
        xy=(prms[1] * 0.95, 0.525),
        ha="right",
        va="bottom",
        color="black",
        fontsize=FigParams.fontsize,
    )


def build_plasticity_rule_axes(
    ax_stdp: plt.Axes,
    ax_homeostasis: plt.Axes,
    time_constant: float = 20,
    max_delay: float = 100,
    depression_ratio: float = 1.1,
    homeostasis_max_ratio: float = 5,
    ltp_color: str = NMDAR.color,
    ltd_color: str = VGCC.color,
    homeostasis_color: str = "k",
    markersize: float = 10,
    fontsize: float = FigParams.fontsize,
):
    xvals = np.linspace(0, max_delay, 501)
    ltp_curve = np.exp(-xvals / time_constant)
    ltd_curve = -depression_ratio * np.exp(-xvals / time_constant)

    ylim = depression_ratio * 1.1
    ax_stdp.plot(-xvals, ltp_curve, color=ltp_color, linewidth=FigParams.linewidth)
    ax_stdp.plot(xvals, ltd_curve, color=ltd_color, linewidth=FigParams.linewidth)
    ax_stdp.plot(0, 1, color=ltp_color, marker=".", markersize=markersize, zorder=10)
    ax_stdp.plot(0, -depression_ratio, color=ltd_color, marker=".", markersize=markersize, zorder=10)
    ax_stdp.text(
        -xvals[-1] * 0.1,
        ylim * 0.8,
        r"$Pre \rightarrow Post$",
        ha="right",
        va="top",
        color=ltp_color,
        fontsize=fontsize,
    )
    ax_stdp.text(
        xvals[-1] * 0.1,
        -ylim * 0.8,
        r"$Post \rightarrow Pre$",
        ha="left",
        va="bottom",
        color=ltd_color,
        fontsize=fontsize,
    )
    text = ax_stdp.text(xvals[-1] * 0.05, 0.8, "LTP", ha="left", va="top", color=ltp_color, fontsize=fontsize)
    ax_stdp.annotate("LTD", xycoords=text, xy=(0, 0), ha="left", va="top", color=ltd_color, fontsize=fontsize)
    ax_stdp.text(
        xvals[-1] * 0.98,
        ylim * 0.025,
        r"$\Delta T$ (ms)",
        ha="right",
        va="bottom",
        color="black",
        fontsize=fontsize,
    )
    ax_stdp.text(
        -xvals[-1] * 0.05,
        -ylim * 0.5,
        r"$+\Delta W$" + "\n" + r"$\propto max$",
        ha="right",
        va="center",
        rotation=90,
        color="black",
        fontsize=fontsize,
    )
    ax_stdp.set_xlim(-xvals[-1], xvals[-1])
    ax_stdp.set_ylim(-ylim, ylim)

    xvals = np.linspace(-homeostasis_max_ratio, homeostasis_max_ratio, 1001)
    homeostasis_curve = -xvals / homeostasis_max_ratio / 0.8
    ylim = max(np.abs(homeostasis_curve)) * 1.1
    ax_homeostasis.plot(xvals, homeostasis_curve, color=homeostasis_color, linewidth=FigParams.linewidth)
    ax_homeostasis.text(
        xvals[-1],
        ylim * 0.99,
        "Homeostasic\nScaling",
        ha="right",
        va="top",
        color=homeostasis_color,
        fontsize=fontsize,
    )
    ax_homeostasis.text(
        xvals[-1] * 0.98,
        ylim * 0.05,
        r"$log(\frac{rate}{setpoint})$",
        ha="right",
        va="bottom",
        color="black",
        fontsize=fontsize,
    )
    ax_homeostasis.text(
        -xvals[-1] * 0.025,
        -ylim * 0.5,
        r"x$\Delta W$" + "\n" + r"$\propto W$",
        ha="right",
        va="center",
        rotation=90,
        color="black",
        fontsize=fontsize,
    )
    ax_homeostasis.set_xlim(xvals[0], xvals[-1])
    ax_homeostasis.set_ylim(-ylim, ylim)


def build_environment_compartment_mapping_ax(
    ax: plt.Axes,
    xrange_buffer: float = 0.01,
    yrange_buffer: float = 0.05,
    proximal_inset_xoffset: float = 0.2,
    simple_tuft_inset_xoffset: float = 0.18,
    complex_tuft_inset_xoffset: float = -0.18,
    visual_inset_length: float = 1.35,
    tuft_yoffset: float = 0.0,
    gabor_width: float = 0.6,
    gabor_envelope: float = 0.4,
    gabor_gamma: float = 1.5,
    gabor_halfsize: float = 25,
    gabor_phase: float = 0,
    gabor_spacing: int = 2,
    gabor_x_extent_fraction: float = 0.5,
    gabor_spine_linewidth: float = FigParams.thinlinewidth,
    gabor_label_yoffset: float = 0.05,
    gabor_highlight_magnitude: float = 4,
    fontsize: float = FigParams.smallfontsize,
):
    def set_spine_properties(ax, spine_color: str):
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(gabor_spine_linewidth)

    group_colors = [Proximal.color, DistalSimple.color, DistalComplex.color]
    bounds = []
    neuron = Neuron(linewidth=FigParams.thicklinewidth)
    elements = neuron.plot(
        ax,
        origin=(0, 0),
        **neuron_color_kwargs(*group_colors),
    )
    bounds.append(neuron.get_bounds(elements))

    xmin = np.min([b[0] for b in bounds])
    xmax = np.max([b[1] for b in bounds])
    ymin = np.min([b[2] for b in bounds])
    ymax = np.max([b[3] for b in bounds])
    xrange = xmax - xmin
    yrange = ymax - ymin
    ymin = ymin - yrange * yrange_buffer
    ymax = ymax + yrange * yrange_buffer

    # Calculate positions for insets based on neuron elements
    stim = SourcePopulationGabor.make_stimulus(edge_probability=100, center_orientation=0)
    stimori = SourcePopulationGabor.orientations[stim]
    params = dict(
        width=gabor_width,
        envelope=gabor_envelope,
        gamma=gabor_gamma,
        halfsize=gabor_halfsize,
        phase=gabor_phase,
    )

    trunk = elements["trunk"]
    trunk_xcenter = trunk.get_xdata()[0]
    trunk_ycenter = sum(trunk.get_ydata()) / 2

    simple_tuft = elements["simple_tuft"]
    simple_tuft_outer_x = min(simple_tuft.get_xdata())
    simple_tuft_ycenter = sum(simple_tuft.get_ydata()) / 2

    complex_tuft = elements["complex_tuft"][0]
    complex_tuft_branches = elements["complex_branches"]
    complex_tuft_outer_x = max([b.get_xdata()[1] for b in complex_tuft_branches + [complex_tuft]])
    complex_tuft_min_y = min([b.get_ydata()[0] for b in complex_tuft_branches + [complex_tuft]])
    complex_tuft_max_y = max([b.get_ydata()[1] for b in complex_tuft_branches + [complex_tuft]])
    complex_tuft_ycenter = (complex_tuft_min_y + complex_tuft_max_y) / 2

    # Proximal branch (trunk)
    ax.text(
        trunk_xcenter + proximal_inset_xoffset + visual_inset_length / 2,
        trunk_ycenter + visual_inset_length / 2 + gabor_label_yoffset,
        "Proximal\nInputs",
        ha="center",
        va="bottom",
        color=Proximal.color,
        fontsize=fontsize,
    )
    xmax = max(xmax, trunk_xcenter + proximal_inset_xoffset + visual_inset_length)
    proximal_inset_position = [
        trunk_xcenter + proximal_inset_xoffset,
        trunk_ycenter - visual_inset_length / 2,
        visual_inset_length,
        visual_inset_length,
    ]
    ax_proximal_environment = ax.inset_axes(
        proximal_inset_position,
        transform=ax.transData,
    )
    proximal_grid, proximal_gabors = create_gabor_grid(
        stimori,
        spacing=gabor_spacing,
        gabor_params=params,
        center_only=True,
        highlight_edge=True,
        highlight_magnitude=gabor_highlight_magnitude,
    )
    max_grid = np.nanmax(np.abs(proximal_grid))
    extent = [0, proximal_grid.shape[1], 0, proximal_grid.shape[0]]
    ax_proximal_environment.imshow(
        proximal_grid,
        aspect="auto",
        cmap="bwr",
        vmin=-max_grid,
        vmax=max_grid,
        interpolation="bilinear",
        extent=extent,
    )
    overlay_empty_pixels_with_x(
        ax_proximal_environment,
        gabors=proximal_gabors,
        spacing=gabor_spacing,
        x_extent_fraction=gabor_x_extent_fraction,
        color="black",
        linewidth=FigParams.thinlinewidth,
        linestyle="-",
    )
    ax_proximal_environment.set_xlim(0, proximal_grid.shape[1])
    ax_proximal_environment.set_ylim(0, proximal_grid.shape[0])
    ax_proximal_environment.set_xticks([])
    ax_proximal_environment.set_yticks([])
    set_spine_properties(ax_proximal_environment, Proximal.color)
    ax_proximal_environment.set_aspect("equal")

    # Simple tuft
    ax.text(
        simple_tuft_outer_x + simple_tuft_inset_xoffset - visual_inset_length,
        simple_tuft_ycenter + visual_inset_length / 2 + gabor_label_yoffset + tuft_yoffset,
        "Distal\nSimple\nInputs",
        ha="left",
        va="bottom",
        color=DistalSimple.color,
        fontsize=fontsize,
    )
    xmin = min(xmin, simple_tuft_outer_x + simple_tuft_inset_xoffset - visual_inset_length)
    simple_tuft_inset_position = [
        simple_tuft_outer_x + simple_tuft_inset_xoffset - visual_inset_length,
        simple_tuft_ycenter - visual_inset_length / 2 + tuft_yoffset,
        visual_inset_length,
        visual_inset_length,
    ]
    ax_simple_tuft_environment = ax.inset_axes(
        simple_tuft_inset_position,
        transform=ax.transData,
    )

    simple_tuft_grid = create_gabor_grid(
        stimori,
        spacing=gabor_spacing,
        gabor_params=params,
        center_only=False,
        highlight_edge=True,
        highlight_magnitude=gabor_highlight_magnitude,
    )[0]
    max_grid = np.nanmax(np.abs(simple_tuft_grid))
    extent = [0, simple_tuft_grid.shape[1], 0, simple_tuft_grid.shape[0]]
    ax_simple_tuft_environment.imshow(
        simple_tuft_grid,
        aspect="auto",
        cmap="bwr",
        vmin=-max_grid,
        vmax=max_grid,
        interpolation="bilinear",
        extent=extent,
    )
    ax_simple_tuft_environment.set_xlim(0, simple_tuft_grid.shape[1])
    ax_simple_tuft_environment.set_ylim(0, simple_tuft_grid.shape[0])
    ax_simple_tuft_environment.set_xticks([])
    ax_simple_tuft_environment.set_yticks([])
    set_spine_properties(ax_simple_tuft_environment, DistalSimple.color)
    ax_simple_tuft_environment.set_aspect("equal")

    # Complex tuft
    ax.text(
        complex_tuft_outer_x + complex_tuft_inset_xoffset + visual_inset_length,
        complex_tuft_ycenter + visual_inset_length / 2 + gabor_label_yoffset + tuft_yoffset,
        "Distal\nComplex\nInputs",
        ha="right",
        va="bottom",
        color=DistalComplex.color,
        fontsize=fontsize,
    )
    xmax = max(xmax, complex_tuft_outer_x + complex_tuft_inset_xoffset + visual_inset_length)
    complex_tuft_inset_position = [
        complex_tuft_outer_x + complex_tuft_inset_xoffset,
        complex_tuft_ycenter - visual_inset_length / 2 + tuft_yoffset,
        visual_inset_length,
        visual_inset_length,
    ]
    ax_complex_tuft_environment = ax.inset_axes(
        complex_tuft_inset_position,
        transform=ax.transData,
    )

    complex_tuft_grid = create_gabor_grid(
        stimori,
        spacing=gabor_spacing,
        gabor_params=params,
        center_only=False,
        highlight_edge=True,
        highlight_magnitude=gabor_highlight_magnitude,
    )[0]
    max_grid = np.nanmax(np.abs(complex_tuft_grid))
    extent = [0, complex_tuft_grid.shape[1], 0, complex_tuft_grid.shape[0]]
    ax_complex_tuft_environment.imshow(
        complex_tuft_grid,
        aspect="auto",
        cmap="bwr",
        vmin=-max_grid,
        vmax=max_grid,
        interpolation="bilinear",
        extent=extent,
    )
    ax_complex_tuft_environment.set_xlim(0, complex_tuft_grid.shape[1])
    ax_complex_tuft_environment.set_ylim(0, complex_tuft_grid.shape[0])
    ax_complex_tuft_environment.set_xticks([])
    ax_complex_tuft_environment.set_yticks([])
    set_spine_properties(ax_complex_tuft_environment, DistalComplex.color)
    ax_complex_tuft_environment.set_aspect("equal")

    # Set limits
    xrange = xmax - xmin
    xmin = xmin - xrange * xrange_buffer
    xmax = xmax + xrange * xrange_buffer
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    label_text = "Stimulus\nMapping:\n\nDistal" + r"$\leftarrow$" + "All\nProx." + r"$\leftarrow$" + "Center"
    ax.text(
        xmin,
        trunk_ycenter,
        label_text,
        ha="left",
        va="center",
        fontsize=fontsize,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_receptive_field_ax(
    ax: plt.Axes,
    field_width: float = 0.8,
    field_scale: float = 1.75,
    field_inset_yoffset_fraction: float = 0.15,
    input_inset_yoffset_extra: float = 0.05,
    vonmises_concentration: float = 1.0,
    baseline_rate: float = 5.0,
    driven_rate: float = 45.0,
    gabor_width: float = 0.6,
    gabor_envelope: float = 0.4,
    gabor_gamma: float = 1.5,
    gabor_halfsize: float = 25,
    gabor_phase: float = 0,
    x_offset_input_label: float = -0.7,
    x_offset_rate_label: float = -0.7,
    x_offset_field_label: float = -0.7,
    fontsize: float = FigParams.fontsize,
    include_arrows: bool = True,
):
    params = dict(
        width=gabor_width,
        envelope=gabor_envelope,
        gamma=gabor_gamma,
        halfsize=gabor_halfsize,
        phase=gabor_phase,
    )

    circular_offsets = np.arange(4) * np.pi / 4
    tuning_curve = vonmises(circular_offsets, vonmises_concentration) * driven_rate + baseline_rate
    max_curve = np.max(tuning_curve)
    y_offset = max_curve * field_inset_yoffset_fraction
    y_extra = max_curve * input_inset_yoffset_extra

    ax.bar(np.arange(4), tuning_curve, width=field_width, color="black")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    yxratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    field_height = field_width * yxratio

    inset_width = field_width * field_scale
    inset_height = field_height * field_scale

    max_ylim = max_curve + 2 * y_offset + 2 * inset_height + 2 * y_extra

    gabors = [create_gabor(offset, **params) for offset in circular_offsets]
    vmax = np.max([np.max(np.abs(gabor)) for gabor in gabors])

    tuning_inset_position = [
        1.5 - inset_width / 2,
        max_curve + 2 * y_offset + inset_height + y_extra,
        inset_width,
        inset_height,
    ]
    tuning_inset = ax.inset_axes(tuning_inset_position, transform=ax.transData)
    tuning_inset.imshow(gabors[0], aspect="equal", cmap="bwr", vmin=-vmax, vmax=vmax)
    for spine in tuning_inset.spines.values():
        spine.set_color("black")
        spine.set_linewidth(FigParams.thinlinewidth)
    tuning_inset.set_xticks([])
    tuning_inset.set_yticks([])

    # Calculate the center bottom of the tuning_inset
    tuning_inset_center_bottom = (
        tuning_inset_position[0] + inset_width / 2,
        tuning_inset_position[1],
    )

    for ifield in range(4):
        inset_position = [
            ifield - inset_width / 2,
            max_curve + y_offset,
            inset_width,
            inset_height,
        ]
        inset = ax.inset_axes(inset_position, transform=ax.transData)
        inset.imshow(gabors[ifield], aspect="equal", cmap="bwr", vmin=-vmax, vmax=vmax)
        inset.set_xticks([])
        inset.set_yticks([])

        # Calculate the center top of each inset
        inset_center_top = (inset_position[0] + inset_width / 2, inset_position[1] + inset_height)
        inset_center_bottom = (inset_position[0] + inset_width / 2, inset_position[1])
        firing_rate_center_top = (inset_position[0] + inset_width / 2, tuning_curve[ifield])

        for spine in inset.spines.values():
            spine.set_color("black")
            spine.set_linewidth(FigParams.thinlinewidth)

        # Draw an arrow from the tuning_inset to each inset
        if include_arrows:
            ax.plot(
                [tuning_inset_center_bottom[0], inset_center_top[0]],
                [tuning_inset_center_bottom[1], inset_center_top[1]],
                "k-",
                lw=FigParams.thinlinewidth,
            )
            ax.plot(
                [inset_center_bottom[0], firing_rate_center_top[0]],
                [inset_center_bottom[1], firing_rate_center_top[1]],
                "k-",
                lw=FigParams.thinlinewidth,
            )

    ax.text(
        x_offset_field_label,
        max_curve + y_offset + inset_height / 10,
        "Receptive\nFields",
        ha="left",
        va="bottom",
        rotation=90,
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    ax.text(
        x_offset_rate_label,
        max_curve / 2,
        "Firing Rates",
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
    )
    ax.text(
        1.5 - inset_width / 2 + x_offset_input_label,
        max_curve + 2 * y_offset + y_extra + inset_height * 3 / 2,
        "Input",
        ha="center",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=fontsize,
    )

    xlabels = [r"0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"]
    yboundmax = np.ceil(max_curve)
    ax.set_ylim(0, max_ylim)
    format_spines(
        ax,
        x_pos=-0.0,
        y_pos=-0.01,
        xbounds=(-0.5, 3.5),
        ybounds=(0, yboundmax),
        xticks=range(4),
        yticks=(0, yboundmax),
        spine_linewidth=FigParams.thinlinewidth,
        tick_length=FigParams.tick_length / 2,
        tick_width=FigParams.tick_width / 2,
        tick_fontsize=FigParams.tick_fontsize * 0.7,
    )
    ax.set_xticks(range(4), xlabels)
    ax.set_xlabel("Tuning Orientation", fontsize=fontsize)


def build_tuning_representation_ax(
    ax: plt.Axes,
    gabor_width: float = 0.6,
    gabor_envelope: float = 0.4,
    gabor_gamma: float = 1.5,
    gabor_halfsize: float = 25,
    gabor_phase: float = 0,
    hspacing: float = 5,
    vspacing: float = 15,
    fontsize_label: float = FigParams.smallfontsize,
    fontsize_title: float = FigParams.fontsize,
):
    params = dict(
        width=gabor_width,
        envelope=gabor_envelope,
        gamma=gabor_gamma,
        halfsize=gabor_halfsize,
        phase=gabor_phase,
    )

    gabors = [create_gabor(offset, **params) for offset in SourcePopulationGabor.orientations]
    tuned = [1] + [0] * (len(gabors) - 1)
    untuned = [0.1] * len(gabors)
    net_tuned = np.sum(np.stack([t * g for t, g in zip(tuned, gabors)]), axis=0)
    net_untuned = np.sum(np.stack([u * g for u, g in zip(untuned, gabors)]), axis=0)

    gabor_height = gabors[0].shape[0]
    hspacer = np.full((gabor_height, hspacing), np.nan)
    vspacer = np.full((vspacing, gabor_height), np.nan)
    gabor_row = []
    for gabor in gabors:
        gabor_row.append(gabor)
        gabor_row.append(hspacer)
    gabor_row.append(hspacer)

    gabor_grid = np.hstack(gabor_row)
    tuning_gabor = np.vstack([net_tuned, vspacer, net_untuned])
    height_difference = tuning_gabor.shape[0] - gabor_grid.shape[0]
    extra_vspacer = np.full((height_difference // 2, gabor_grid.shape[1]), np.nan)

    full_grid = np.hstack([np.vstack([extra_vspacer, gabor_grid, extra_vspacer]), tuning_gabor])
    full_grid = np.vstack([np.full((vspacing, full_grid.shape[1]), np.nan), full_grid])

    extent = (0, full_grid.shape[1], 0, full_grid.shape[0])

    x_pos_strength = np.arange(4) * (gabor_height + hspacing) + gabor_height / 2
    x_pos_strength = np.append(x_pos_strength, (full_grid.shape[1] - gabor_height - gabor_height / 20))
    y_pos_untuned = gabor_height / 2
    y_pos_tuned = full_grid.shape[0] - vspacing - gabor_height / 2
    x_pos_description = full_grid.shape[1] - gabor_height / 2
    y_pos_description_untuned = gabor_height
    y_pos_description_tuned = gabor_height * 2 + vspacing

    x_pos_ax_description = gabor_height * 2 + hspacing / 2
    y_pos_ax_description = (full_grid.shape[0] + full_grid.shape[0] - height_difference / 2) / 2

    vmax = np.nanmax(np.abs(full_grid))
    cmap = plt.get_cmap("bwr")
    cmap.set_bad("white", alpha=0.0)
    ax.imshow(full_grid, aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent, interpolation="bilinear")

    for i in range(5):
        if i < 4:
            tuned_text = f"{tuned[i]:2g}"
            untuned_text = f"{untuned[i]:2g}"
            ha = "center"
        else:
            tuned_text = "="
            untuned_text = "="
            ha = "right"
        untuned_text = ax.text(
            x_pos_strength[i],
            y_pos_untuned,
            untuned_text,
            ha=ha,
            va="center",
            fontsize=fontsize_label,
        )
        tuned_text = ax.text(
            x_pos_strength[i],
            y_pos_tuned,
            tuned_text,
            ha=ha,
            va="center",
            fontsize=fontsize_label,
        )

        if i == 0:
            ax.annotate("(", xy=(0, 0), xycoords=untuned_text, ha="right", va="bottom", fontsize=fontsize_label)
            ax.annotate("(", xy=(0, 0), xycoords=tuned_text, ha="right", va="bottom", fontsize=fontsize_label)
        if i == 3:
            ax.annotate(")", xy=(1, 0), xycoords=untuned_text, ha="left", va="bottom", fontsize=fontsize_label)
            ax.annotate(")", xy=(1, 0), xycoords=tuned_text, ha="left", va="bottom", fontsize=fontsize_label)
        if i > 0 and i < 4:
            middle_x = (x_pos_strength[i] + x_pos_strength[i - 1]) / 2
            ax.text(middle_x, y_pos_untuned, "x", ha="center", va="center", fontsize=fontsize_label - 1)
            ax.text(middle_x, y_pos_tuned, "x", ha="center", va="center", fontsize=fontsize_label - 1)

    ax.text(x_pos_description, y_pos_description_untuned, "Untuned", ha="center", va="bottom", fontsize=fontsize_label)
    ax.text(x_pos_description, y_pos_description_tuned, "Tuned", ha="center", va="bottom", fontsize=fontsize_label)
    ax.text(
        x_pos_ax_description,
        y_pos_ax_description,
        "Net Tuning Representation",
        ha="center",
        va="top",
        fontsize=fontsize_title,
    )

    ax.set_xlim(0, full_grid.shape[1])
    ax.set_ylim(0, full_grid.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")


def build_stimulus_trajectory_ax(
    ax: plt.Axes,
    stims_per_row: int = 5,
    num_edges: int = 7,
    highlight_magnitude: int = 4,
    hspacing: float = 0.1,
    vspacing: float = 0.05,
    arrow_width: float = 0.75,
    vmax_scale: float = 1.5,
    arrow_mutation: float = 10,
):
    # Determine stimulus indices
    num_stims = 2 * stims_per_row
    edge_stims = sorted(np.random.permutation(num_stims)[:num_edges])

    # Generate stimuli and grids
    stim_grids = [
        SourcePopulationGabor.orientations[
            SourcePopulationGabor.make_stimulus(edge_probability=1 if i in edge_stims else 0)
        ]
        for i in range(num_stims)
    ]
    gabor_grids = [
        create_gabor_grid(
            stim_grids[i],
            spacing=2,
            highlight_edge=i in edge_stims,
            highlight_magnitude=highlight_magnitude,
        )[0]
        for i in range(num_stims)
    ]
    vmax = np.max(np.abs(np.stack(gabor_grids))) * vmax_scale

    # Calculate row positions
    total_width = stims_per_row + stims_per_row * hspacing + arrow_width
    total_height = 2 + vspacing

    axs_inset = []

    # Create first row of insets
    for i in range(stims_per_row):
        x = i + i * hspacing
        y = 1 + vspacing
        inset = ax.inset_axes([x, y, 1, 1], transform=ax.transData)
        axs_inset.append(inset)

    # Create second row of insets
    for i in range(stims_per_row):
        x = arrow_width + hspacing + i + i * hspacing
        inset = ax.inset_axes([x, 0, 1, 1], transform=ax.transData)
        axs_inset.append(inset)

    # Create arrow insets
    arrow_row_one = ax.inset_axes(
        [stims_per_row + stims_per_row * hspacing, 1 + vspacing, arrow_width, 1], transform=ax.transData
    )
    arrow_row_two = ax.inset_axes([0, 0, arrow_width, 1], transform=ax.transData)

    # Display images
    for inset_ax, grid in zip(axs_inset, gabor_grids):
        inset_ax.imshow(grid, aspect="equal", cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="bilinear")
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("k")
            spine.set_linewidth(FigParams.thinlinewidth)

    for ax_arrow in [arrow_row_one, arrow_row_two]:
        arrow = FancyArrowPatch(
            posA=(0, 0),
            posB=(1.0, 0),
            arrowstyle="-|>",
            linewidth=FigParams.thicklinewidth,
            mutation_scale=arrow_mutation,
            color="black",
        )
        ax_arrow.add_patch(arrow)
        ax_arrow.axis("off")
        ax_arrow.set_xlim(0, 1)
        ax_arrow.set_ylim(-0.1, 0.1)

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    ax.set_title("Stimulus Trajectory", fontsize=FigParams.smallfontsize)


def build_orientation_confusion_axes(
    ax_distal_simple,
    ax_distal_complex,
    orientation_preferences,
    fontsize: float = FigParams.smallfontsize,
    tickfontsize: float = FigParams.tinyfontsize,
):
    ConfusionMatrixDisplay.from_predictions(
        orientation_preferences["proximal"].reshape(-1),
        orientation_preferences["distal-simple"].reshape(-1),
        ax=ax_distal_simple,
        cmap="gray_r",
        colorbar=False,
        normalize="all",
        text_kw={"fontsize": tickfontsize},
    )
    ConfusionMatrixDisplay.from_predictions(
        orientation_preferences["proximal"].reshape(-1),
        orientation_preferences["distal-complex"].reshape(-1),
        ax=ax_distal_complex,
        cmap="gray_r",
        colorbar=False,
        normalize="all",
        text_kw={"fontsize": tickfontsize},
    )
    ax_distal_complex.set_xlabel("Proximal\nPreferred Orientation", fontsize=fontsize)
    ax_distal_simple.set_xlabel("", fontsize=fontsize)
    ax_distal_simple.set_ylabel("Distal-Simple\nPreferred Orientation", fontsize=fontsize)
    ax_distal_complex.set_ylabel("Distal-Complex\nPreferred Orientation", fontsize=fontsize)
    ticks = np.arange(4)
    labels = np.array(ticks * 180 / 4, dtype=int)
    ax_distal_simple.set_xticks(ticks, [])
    ax_distal_complex.set_xticks(ticks, labels, fontsize=tickfontsize)
    ax_distal_simple.set_yticks(ticks, labels, fontsize=tickfontsize)
    ax_distal_complex.set_yticks(ticks, labels, fontsize=tickfontsize)

    for ax in [ax_distal_simple, ax_distal_complex]:
        ax.tick_params(
            axis="both",
            which="major",
            length=FigParams.tick_length,
            width=FigParams.tick_width,
            labelsize=tickfontsize,
        )


def build_weights_ax(
    ax_proximal: plt.Axes,
    ax_simple: plt.Axes,
    ax_complex: plt.Axes,
    weights: dict[str, np.ndarray],
    spacing: int = 2,
    vmax: float = 0.5,
    dpratio: int = 0,
    edge: int = 0,
    simulation: int = 0,
    neuron: int = 0,
    gabor_width: float = 0.6,
    gabor_envelope: float = 0.4,
    gabor_gamma: float = 1.5,
    gabor_halfsize: int = 25,
    gabor_phase: float = 0,
    gabor_x_extent_fraction: float = 0.5,
    fontsize: float = FigParams.smallfontsize,
    show_titles: bool = True,
):
    gabor_params = dict(
        width=gabor_width,
        envelope=gabor_envelope,
        gamma=gabor_gamma,
        halfsize=gabor_halfsize,
        phase=gabor_phase,
    )
    proximal_gabor = weights_to_gabor(
        weights["proximal"][dpratio, edge, simulation, neuron],
        SourcePopulationGabor.orientations,
        spacing=spacing,
        **gabor_params,
    )
    simple_gabor = weights_to_gabor(
        weights["distal-simple"][dpratio, edge, simulation, neuron],
        SourcePopulationGabor.orientations,
        spacing=spacing,
        **gabor_params,
    )
    complex_gabor = weights_to_gabor(
        weights["distal-complex"][dpratio, edge, simulation, neuron],
        SourcePopulationGabor.orientations,
        spacing=spacing,
        **gabor_params,
    )

    ax_proximal.imshow(proximal_gabor, vmin=-vmax, vmax=vmax, cmap="bwr")
    ax_simple.imshow(simple_gabor, vmin=-vmax, vmax=vmax, cmap="bwr")
    ax_complex.imshow(complex_gabor, vmin=-vmax, vmax=vmax, cmap="bwr")
    if show_titles:
        ax_proximal.set_title("Proximal", fontsize=fontsize)
        ax_simple.set_title("Distal-Simple", fontsize=fontsize)
        ax_complex.set_title("Distal-Complex", fontsize=fontsize)

    for ax in [ax_proximal, ax_simple, ax_complex]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("k")
            spine.set_linewidth(FigParams.thinlinewidth)

    xs_gabor = create_gabor_grid(np.random.randn(3, 3), spacing=spacing, gabor_params=gabor_params, center_only=True)[1]
    overlay_empty_pixels_with_x(
        ax_proximal,
        gabors=xs_gabor,
        spacing=spacing,
        x_extent_fraction=gabor_x_extent_fraction,
        color="black",
        linewidth=FigParams.thinlinewidth,
        linestyle="-",
    )

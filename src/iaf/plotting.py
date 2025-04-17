from typing import Optional, List
import numpy as np
from scipy.signal import filtfilt
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import colors as mcolors
from .analysis import get_norm_factor, get_groupnames, get_sigmoid_params, sigmoid
from ..plotting import FigParams, beeswarm
from ..conductance import NMDAR, VGCC


def create_dpratio_colors(num_ratios: int, cmap: str = "plasma_r", cmap_pinch: float = 0.25):
    cmap = colormaps[cmap]
    colors = [cmap(ii) for ii in np.linspace(cmap_pinch, 1 - cmap_pinch, num_ratios)]
    return colors, cmap


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

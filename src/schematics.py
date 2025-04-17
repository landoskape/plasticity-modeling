import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType

from .plotting import FigParams, Proximal, DistalSimple, DistalComplex, add_dpratio_inset
from .iaf.plotting import create_dpratio_colors


class Neuron:
    def __init__(
        self,
        linewidth=2.5,
        soma_size=1.0,
        soma_filled=True,
        soma_color=Proximal.color,
        trunk_height=1.75,
        tuft_height=3.0,
        trunk_color=Proximal.color,
        simple_tuft_angle=20,
        simple_tuft_color=DistalSimple.color,
        complex_tuft_angle=20,
        complex_tuft_color=DistalComplex.color,
        complex_tuft_branches=5,
        complex_tuft_branch_fraction=0.3,
        trunk_text: str | None = None,
        trunk_text_color: ColorType = Proximal.color,
        trunk_text_offset: float = 0.2,
        simple_tuft_text: str | None = None,
        simple_tuft_text_color: ColorType = DistalSimple.color,
        simple_tuft_text_offset: float = 0.2,
        complex_tuft_text: str | None = None,
        complex_tuft_text_color: ColorType = DistalComplex.color,
        complex_tuft_text_offset: float = 0.2,
        fontsize: float = 8,
    ):
        """
        Initialize a neuron schematic with idealized dendrites.

        Parameters:
        -----------
        linewidth : float
            Line width for all dendrites
        soma_size : float
            Size of the soma (cell body) circle
        soma_filled : bool
            Whether the soma is filled or empty
        soma_color : str
            Color of the soma
        trunk_height : float
            Height of the apical trunk dendrite
        tuft_height : float
            Height of the tuft dendrites (both simple and complex)
        trunk_color : str
            Color of the apical trunk dendrite
        simple_tuft_angle : float
            Angle (in degrees) of the simple apical tuft dendrite
        simple_tuft_color : str
            Color of the simple apical tuft dendrite
        complex_tuft_angle : float
            Angle (in degrees) of the complex apical tuft dendrite
        complex_tuft_color : str
            Color of the complex apical tuft dendrite
        complex_tuft_branches : int
            Number of branches in the complex tuft
        complex_tuft_branch_fraction : float
            Fraction of the tuft height that the branches will occupy
        trunk_text : str | None
            Text to display on the trunk
        trunk_text_color : ColorType
            Color of the trunk text
        trunk_text_offset : float
            Offset of the trunk text from the trunk
        simple_tuft_text : str | None
            Text to display on the simple tuft
        simple_tuft_text_color : ColorType
            Color of the simple tuft text
        complex_tuft_text : str | None
            Text to display on the complex tuft
        complex_tuft_text_color : ColorType
            Color of the complex tuft text
        complex_tuft_text_offset : float
            Offset of the complex tuft text from the complex tuft
        fontsize : float
            Font size for all text
        """
        self.linewidth = linewidth
        self.soma_size = soma_size
        self.soma_filled = soma_filled
        self.soma_color = soma_color
        self.trunk_height = trunk_height
        self.tuft_height = tuft_height
        self.trunk_color = trunk_color
        self.simple_tuft_angle = simple_tuft_angle
        self.simple_tuft_color = simple_tuft_color
        self.complex_tuft_angle = complex_tuft_angle
        self.complex_tuft_color = complex_tuft_color
        self.complex_tuft_branches = complex_tuft_branches
        self.complex_tuft_branch_fraction = complex_tuft_branch_fraction
        self.trunk_text = trunk_text
        self.trunk_text_color = trunk_text_color
        self.trunk_text_offset = trunk_text_offset
        self.simple_tuft_text = simple_tuft_text
        self.simple_tuft_text_color = simple_tuft_text_color
        self.simple_tuft_text_offset = simple_tuft_text_offset
        self.complex_tuft_text = complex_tuft_text
        self.complex_tuft_text_color = complex_tuft_text_color
        self.complex_tuft_text_offset = complex_tuft_text_offset
        self.fontsize = fontsize

    def plot(self, ax: plt.Axes, origin=(0, 0), **kwargs):
        """
        Create the neuron schematic on the given matplotlib axis.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis on which to draw the neuron
        origin : tuple of float
            The (x, y) coordinates for the center of the soma
        **kwargs : dict
            Additional parameters that can override instance attributes

        Returns:
        --------
        elements : dict
            Dictionary containing the drawn elements of the neuron
        """
        # Elements created during rendering
        elements = {
            "soma": None,
            "trunk": None,
            "simple_tuft": None,
            "complex_tuft": [],
            "complex_branches": [],
            "text": {"trunk": None, "simple_tuft": None, "complex_tuft": None},
        }

        # Update attributes with any provided kwargs
        original_attributes = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                original_attributes[key] = getattr(self, key)
                setattr(self, key, value)

        x0, y0 = origin

        # Calculate the radius of the soma
        soma_radius = self.soma_size / 2

        # Calculate the lengths for the tufts so they reach the same height
        simple_angle_rad = np.radians(90 - self.simple_tuft_angle)
        complex_angle_rad = np.radians(90 - self.complex_tuft_angle)

        # Calculate length needed to reach tuft_height at each angle
        simple_length = self.tuft_height / np.sin(simple_angle_rad)
        complex_length = self.tuft_height / np.sin(complex_angle_rad)

        # Calculate trunk line
        trunk_length = self.trunk_height
        x1, y1 = x0, y0 + soma_radius  # Start at top of soma
        x2, y2 = x0, y1 + trunk_length  # End point

        # Calculate complex tuft base (first half)
        complex_dx = complex_length * np.cos(complex_angle_rad)
        complex_dy = complex_length * np.sin(complex_angle_rad)
        complex_x_mid = x2 + complex_dx * self.complex_tuft_branch_fraction  # Right side, half length
        complex_y_mid = y2 + complex_dy * self.complex_tuft_branch_fraction  # Half height

        # Draw complex tuft base
        complex_base = Line2D(
            [x2, complex_x_mid],
            [y2, complex_y_mid],
            linewidth=self.linewidth,
            color=self.complex_tuft_color,
        )
        elements["complex_tuft"].append(complex_base)
        ax.add_line(complex_base)

        # Draw complex tuft branches
        branch_angles = np.linspace(
            np.radians(self.complex_tuft_angle),
            -np.radians(self.complex_tuft_angle),
            self.complex_tuft_branches,
        )

        # Calculate remaining height to reach tuft_height
        remaining_height = self.tuft_height - complex_dy * self.complex_tuft_branch_fraction

        for angle_offset in branch_angles:
            # Calculate length needed for the branch to reach the total tuft_height
            # Since we're using absolute angles, we use sine relative to vertical
            branch_length = remaining_height / np.cos(angle_offset) if np.cos(angle_offset) != 0 else remaining_height

            dx = branch_length * np.sin(angle_offset)  # Using sin for horizontal component (relative to vertical)
            dy = branch_length * np.cos(angle_offset)  # Using cos for vertical component (relative to vertical)
            x_branch_end = complex_x_mid + dx
            y_branch_end = complex_y_mid + dy

            branch = Line2D(
                [complex_x_mid, x_branch_end],
                [complex_y_mid, y_branch_end],
                linewidth=self.linewidth,
                color=self.complex_tuft_color,
            )
            elements["complex_branches"].append(branch)
            ax.add_line(branch)

        # Calculate simple tuft endpoints
        simple_dx = simple_length * np.cos(simple_angle_rad)
        simple_dy = simple_length * np.sin(simple_angle_rad)
        simple_x_end = x2 - simple_dx  # Left side
        simple_y_end = y2 + simple_dy  # Should be y2 + tuft_height * scale

        # Draw simple tuft
        elements["simple_tuft"] = Line2D(
            [x2, simple_x_end],
            [y2, simple_y_end],
            linewidth=self.linewidth,
            color=self.simple_tuft_color,
        )
        ax.add_line(elements["simple_tuft"])

        # Draw trunk
        elements["trunk"] = Line2D([x1, x2], [y1, y2], linewidth=self.linewidth, color=self.trunk_color)
        ax.add_line(elements["trunk"])

        # Draw soma (circle)
        fill = self.soma_filled
        elements["soma"] = Circle(
            origin,
            soma_radius,
            facecolor=self.soma_color if fill else "none",
            edgecolor=self.soma_color,
            linewidth=self.linewidth,
        )
        ax.add_patch(elements["soma"])

        # --- Add Text Labels ---
        # Trunk Text
        if self.trunk_text:
            trunk_mid_x = x0
            trunk_mid_y = (y1 + y2) / 2
            elements["text"]["trunk"] = ax.text(
                trunk_mid_x - self.trunk_text_offset,  # Offset to the left
                trunk_mid_y,
                self.trunk_text,
                color=self.trunk_text_color,
                fontsize=self.fontsize,
                rotation=90,
                ha="center",
                va="center",
            )

        # Simple Tuft Text
        if self.simple_tuft_text:
            simple_tuft_mid_x = (x2 + simple_x_end) / 2
            simple_tuft_mid_y = (y2 + simple_y_end) / 2
            # Calculate orthogonal vector for offset (left of line segment)
            angle_rad = np.arctan2(simple_y_end - y2, simple_x_end - x2)
            offset_angle = angle_rad + np.pi / 2  # Perpendicular, pointing "left"
            offset_x = self.simple_tuft_text_offset * np.cos(offset_angle)
            offset_y = self.simple_tuft_text_offset * np.sin(offset_angle)
            rotation_deg = np.degrees(angle_rad)  # Rotate parallel to line angle

            elements["text"]["simple_tuft"] = ax.text(
                simple_tuft_mid_x + offset_x,
                simple_tuft_mid_y + offset_y,
                self.simple_tuft_text,
                color=self.simple_tuft_text_color,
                fontsize=self.fontsize,
                rotation=rotation_deg,
                ha="center",
                va="center",
            )

        # Complex Tuft Text (on the base segment)
        if self.complex_tuft_text:
            # Calculate orthogonal vector for offset (right of line segment)
            angle_rad = np.arctan2(complex_y_mid - y2, complex_x_mid - x2)
            complex_text_y = y2 + self.tuft_height / 2  # Vertical midpoint of entire tuft
            x_on_line = x2 if angle_rad == 0 else x2 + (complex_text_y - y2) / np.tan(angle_rad)
            complex_text_x = x_on_line + self.complex_tuft_text_offset
            rotation_deg = np.degrees(angle_rad)  # Rotate parallel to base segment angle

            elements["text"]["complex_tuft"] = ax.text(
                complex_text_x,
                complex_text_y,  # Use the calculated vertical midpoint
                self.complex_tuft_text,
                color=self.complex_tuft_text_color,
                fontsize=self.fontsize,
                rotation=rotation_deg,
                ha="center",
                va="center",
            )

        # Return object to original attributes
        # (kwargs are just for this plot call)
        for key, value in original_attributes.items():
            setattr(self, key, value)

        return elements

    def get_bounds(self, elements):
        """
        Calculate the bounding box (xmin, xmax, ymin, ymax) of the neuron schematic.

        Parameters:
        -----------
        elements : dict
            Dictionary of elements created during plotting.

        Returns:
        --------
        bounds : tuple
            A tuple (xmin, xmax, ymin, ymax) representing the bounding box.
        """
        xmin, xmax = np.inf, -np.inf
        ymin, ymax = np.inf, -np.inf

        # Process each element type
        for key, value in elements.items():
            if isinstance(value, Line2D):  # Single line
                xdata, ydata = value.get_xdata(), value.get_ydata()
                xmin, xmax = min(xmin, np.min(xdata)), max(xmax, np.max(xdata))
                ymin, ymax = min(ymin, np.min(ydata)), max(ymax, np.max(ydata))
            elif isinstance(value, Circle):  # Circle
                center = value.center
                radius = value.radius
                xmin, xmax = min(xmin, center[0] - radius), max(xmax, center[0] + radius)
                ymin, ymax = min(ymin, center[1] - radius), max(ymax, center[1] + radius)
            elif isinstance(value, list):  # List of elements (e.g., branches)
                for item in value:
                    if isinstance(item, Line2D):
                        xdata, ydata = item.get_xdata(), item.get_ydata()
                        xmin, xmax = min(xmin, np.min(xdata)), max(xmax, np.max(xdata))
                        ymin, ymax = min(ymin, np.min(ydata)), max(ymax, np.max(ydata))

        return xmin, xmax, ymin, ymax


def neuron_color_kwargs(proximal_color: ColorType, distal_simple_color: ColorType, distal_complex_color: ColorType):
    """To easily set colors for a neuron schematic."""
    return {
        "soma_color": proximal_color,
        "trunk_color": proximal_color,
        "simple_tuft_color": distal_simple_color,
        "complex_tuft_color": distal_complex_color,
    }


class DPRatio:
    """
    A schematic of DPRatios.

    The schematic uses a horizontal line to represent the 0% extra depression value that's always colored black.
    Then, the ratios are represented as points on the line, with the y-axis representing the extra depression value.
    The points are colored according to the pointcolor. Below each point is a label which is colored according to
    the label color. If linecolor is not "none", will connect the points using a line with that color and the linewidth.
    On the left will be yticks and there labels if provided. Above the whole thing is a title if not None.
    """

    def __init__(
        self,
        ratios: list[float] = [0.1, 0.1, 0.1],
        labels: list[str] = [Proximal.tinylabel, DistalSimple.tinylabel, DistalComplex.tinylabel],
        pointcolors: list[ColorType] = [Proximal.color, DistalSimple.color, DistalComplex.color],
        labelcolors: list[ColorType] = [Proximal.color, DistalSimple.color, DistalComplex.color],
        width: float = 0.5,
        pinch: float = 0.125,
        baselinewidth: float = 1.0,
        linewidth: float = 2.5,
        linecolor: ColorType = "black",
        markersize: float = 10,
        yticks: list[float] | None = [0, 0.1],
        ylabels: list[str] | None = ["0", "10"],
        title: str | None = "Extra Depression",
        xlabel_yoffset: float = -0.12,
        ytitle_yoffset: float = 0.15,
        ytick_xpos: tuple[float, float] | None = (0.025, 0.05),
        label_fontsize: float = 8,
        title_fontsize: float = 8,
    ):
        """
        Initialize a depression-potentiation ratio schematic.

        Parameters:
        -----------
        ratios : list of float
            List of depression-potentiation ratios for each segment
        labels : list of str
            List of labels for each segment
        pointcolors : list of ColorType
            List of colors for the points
        labelcolors : list of ColorType
            List of colors for the labels
        width : float
            Width of the schematic (the horizontal 0% line).
        pinch : float
            How much to compress the points within the width.
        baselinewidth : float
            Width of the horizontal baseline (0% line)
        linewidth : float
            Line width for the segments
        linecolor : ColorType
            Color of the connecting line (use "none" for no line).
        markersize : float
            Size of the markers
        yticks : list of float
            Y-axis ticks for the plot
        ylabels : list of str
            Y-axis labels for the plot
        title : str
            Title for the plot
        xlabel_yoffset : float
            Y-offset for the x-axis labels
        ytitle_yoffset : float
            Y-offset for the y-axis title
        ytick_xpos : tuple[float] | None
            X-position of the y-axis ticks (start position and width)
        label_fontsize : float
            Font size for the x-axis & y-axis labels
        title_fontsize : float
            Font size for the title
        """
        self.ratios = ratios
        self.labels = labels
        self.pointcolors = pointcolors
        self.labelcolors = labelcolors
        self.width = width
        self.pinch = pinch
        self.baselinewidth = baselinewidth
        self.linewidth = linewidth
        self.linecolor = linecolor
        self.markersize = markersize
        self.yticks = yticks
        self.ylabels = ylabels
        self.title = title
        self.xlabel_yoffset = xlabel_yoffset
        self.ytitle_yoffset = ytitle_yoffset
        self.ytick_xpos = ytick_xpos
        self.label_fontsize = label_fontsize
        self.title_fontsize = title_fontsize

    def plot(self, ax: plt.Axes, origin=(0, 0), **kwargs):
        """
        Create the dpratio schematic on the given matplotlib axis.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis on which to draw the neuron
        origin : tuple of float
            The (x, y) coordinates for the center of the 0-line.
        **kwargs : dict
            Additional parameters that can override instance attributes

        Returns:
        --------
        elements : dict
            Dictionary of elements created during plotting.
        """
        # Elements created during rendering
        elements = {
            "bar": None,
            "points": [],
            "labels": [],
            "connecting_line": None,
            "ytick_labels": [],
            "title_text": None,
        }

        # Update attributes with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Compute positions
        x0, y0 = origin
        bar_extent = x0 - self.width / 2, x0 + self.width / 2
        num_points = len(self.ratios)

        # Calculate x positions for points, evenly spaced with pinch factor
        scaled_pinch = self.pinch * self.width
        points_xpos = np.linspace(bar_extent[0] + scaled_pinch, bar_extent[1] - scaled_pinch, num_points)

        # Y positions are directly from ratios
        points_ypos = np.array(self.ratios)

        # Draw the horizontal baseline (0% line)
        baseline = Line2D(
            [bar_extent[0], bar_extent[1]], [y0, y0], linewidth=self.baselinewidth, color="black", zorder=1
        )
        elements["bar"] = baseline
        ax.add_line(baseline)

        # Draw connecting line between points if requested
        if self.linecolor != "none":
            connecting_line = Line2D(
                points_xpos, y0 + points_ypos, linewidth=self.linewidth, color=self.linecolor, zorder=2
            )
            elements["connecting_line"] = connecting_line
            ax.add_line(connecting_line)

        # Draw points
        for i in range(num_points):
            x_point = points_xpos[i]
            y_point = y0 + points_ypos[i]

            # Create point
            point = ax.scatter(
                x_point,
                y_point,
                s=self.markersize,
                color=self.pointcolors[i],
                edgecolor="w",
                linewidth=0.5,
                zorder=3,
            )
            elements["points"].append(point)

            # Add label below point
            if self.labels is not None and i < len(self.labels):
                text = ax.text(
                    x_point,
                    y0 + self.xlabel_yoffset,
                    self.labels[i],
                    color=self.labelcolors[i],
                    ha="center",
                    va="top",
                    fontsize=self.label_fontsize,
                    zorder=3,
                )
                elements["labels"].append(text)

        # Add y-axis ticks and labels if provided
        if self.yticks is not None:
            if self.ylabels is None:
                self.ylabels = []

            ylabel_x = bar_extent[0] - 0.05  # Offset to the left of the bar

            for i, tick_value in enumerate(self.yticks):
                # Scale the tick position
                tick_y = y0 + tick_value

                # Add a short ytick line
                if self.ytick_xpos is not None and tick_value != 0:
                    ytick_x = (
                        bar_extent[0] + self.ytick_xpos[0],
                        bar_extent[0] + self.ytick_xpos[0] + self.ytick_xpos[1],
                    )
                    ytick_line = Line2D(
                        ytick_x,
                        [tick_y, tick_y],
                        linewidth=self.baselinewidth,
                        color="black",
                        zorder=1,
                    )
                    ax.add_line(ytick_line)

                # Add tick label if available
                if i < len(self.ylabels):
                    tick_label = ax.text(
                        ylabel_x,
                        tick_y,
                        self.ylabels[i],
                        ha="right",
                        va="center",
                        fontsize=self.label_fontsize,
                        zorder=1,
                    )
                    elements["ytick_labels"].append(tick_label)

        # Add title if provided
        if self.title is not None:
            title_y = y0 + max(self.ratios) + self.ytitle_yoffset
            title = ax.text(x0, title_y, self.title, ha="center", va="bottom", fontsize=self.title_fontsize, zorder=4)
            elements["title_text"] = title

        # Set the axis limits to crop tightly around the schematic
        padding = 0.2
        y_min = y0 - padding
        y_max = y0 + max(max(self.ratios) + padding, 0.15)
        x_min = bar_extent[0] - padding
        x_max = bar_extent[1] + padding
        limits = x_min, x_max, y_min, y_max

        return elements, limits


def build_integrated_schematic_axis(
    ax_schematic: plt.Axes,
    ax_table: plt.Axes,
    main_neuron_label_style: str = "label",
    main_neuron_label_offset: float = 0.27,
    main_neuron_label_fontsize: float = FigParams.fontsize,
    num_neurons: int = 5,
    neurons_xoffset: float = 1,
    neurons_yoffset: float = 0,
    neurons_xshift: float = 0.6,
    neurons_yshift: float = 0.85,
    small_neuron_soma_size: float = 0.5,
    small_neuron_trunk_height: float = 0.75,
    small_neuron_tuft_height: float = 0.75,
    small_neuron_linewidth: float = FigParams.linewidth,
    dp_max_ratio: float = 0.4,
    dp_width: float = 2,
    dp_pinch: float = 0.175,
    dp_xoffset: float = -0.4,
    dp_markersize: float = 12,
    dp_linewidth: float = FigParams.thicklinewidth,
    dp_xlabel_yoffset: float = -0.12,
    dp_ytitle_yoffset: float = 0.29,
    dp_yschema_shift: float = 2.6,
    dp_color_inset_position: tuple[float] = (0.55, 0.06, 0.07, 0.4),
    dp_color_label_padding: float = -1,
    dp_label_fontsize: float = FigParams.fontsize * 0.85,
    dp_title_fontsize: float = FigParams.fontsize,
    xrange_buffer: float = 0.05,
    yrange_buffer: float = 0.05,
):
    # Add neuron schematic
    dpratio_colors = create_dpratio_colors(num_neurons)[0]
    dpratios = np.linspace(0, 1, num_neurons)  # relative to maxratio yvalue
    group_colors = [Proximal.color, DistalSimple.color, DistalComplex.color]
    neuron_labels = dict(
        trunk_text=getattr(Proximal, main_neuron_label_style),
        simple_tuft_text=getattr(DistalSimple, main_neuron_label_style),
        complex_tuft_text=getattr(DistalComplex, main_neuron_label_style),
        trunk_text_offset=main_neuron_label_offset,
        simple_tuft_text_offset=main_neuron_label_offset,
        complex_tuft_text_offset=main_neuron_label_offset,
        fontsize=main_neuron_label_fontsize,
    )
    dpratio_labels = [Proximal.tinylabel, DistalSimple.tinylabel, DistalComplex.tinylabel]

    bounds = []
    neuron = Neuron(linewidth=FigParams.thicklinewidth)
    elements = neuron.plot(
        ax_schematic,
        origin=(0, 0),
        **neuron_color_kwargs(*group_colors),
        **neuron_labels,
    )
    bounds.append(neuron.get_bounds(elements))

    for icolor, complex_color in enumerate(reversed(dpratio_colors)):
        origin = (
            neurons_xoffset + icolor * neurons_xshift,
            neurons_yoffset + icolor * neurons_yshift,
        )
        c_colors = neuron_color_kwargs(dpratio_colors[-1], dpratio_colors[-1], complex_color)
        smaller_design = {
            "linewidth": small_neuron_linewidth,
            "soma_size": small_neuron_soma_size,
            "trunk_height": small_neuron_trunk_height,
            "tuft_height": small_neuron_tuft_height,
            "complex_tuft_branches": 1 + icolor,
        }
        elements = neuron.plot(
            ax_schematic,
            origin=origin,
            **c_colors,
            **smaller_design,
        )
        bounds.append(neuron.get_bounds(elements))

    # Create the DPRatio object
    yticks = [0, dp_max_ratio]
    dpratio = DPRatio(
        width=dp_width,
        baselinewidth=FigParams.linewidth,
        linewidth=dp_linewidth,
        pinch=dp_pinch,
        yticks=yticks,
        ylabels=["0", "10"],
        markersize=dp_markersize,
        xlabel_yoffset=dp_xlabel_yoffset,
        ytitle_yoffset=dp_ytitle_yoffset,
        label_fontsize=dp_label_fontsize,
        title_fontsize=dp_title_fontsize,
    )

    xmax = np.max([b[1] for b in bounds])
    for icolor, (complex_ratio, complex_color) in enumerate(zip(reversed(dpratios), reversed(dpratio_colors))):
        origin = (
            xmax + dp_width + dp_xoffset,
            icolor * dp_max_ratio * dp_yschema_shift,
        )
        cratios = np.array([1, 1, complex_ratio]) * dp_max_ratio
        _, limits = dpratio.plot(
            ax_schematic,
            origin=origin,
            ratios=cratios,
            linecolor=complex_color,
            labels=dpratio_labels if icolor == 0 else None,
            title="Extra LTD (%)" if icolor == num_neurons - 1 else None,
        )
        bounds.append(limits)

    # Set axis limits
    xmin = np.min([b[0] for b in bounds])
    xmax = np.max([b[1] for b in bounds])
    ymin = np.min([b[2] for b in bounds])
    ymax = np.max([b[3] for b in bounds])
    xrange = xmax - xmin
    yrange = ymax - ymin
    xmin = xmin - xrange * xrange_buffer
    xmax = xmax + xrange * xrange_buffer
    ymin = ymin - yrange * yrange_buffer
    ymax = ymax + yrange * yrange_buffer
    ax_schematic.set_xlim(xmin, xmax)
    ax_schematic.set_ylim(ymin, ymax)
    ax_schematic.set_aspect("equal")
    ax_schematic.set_xticks([])
    ax_schematic.set_yticks([])
    for spine in ax_schematic.spines.values():
        spine.set_visible(False)

    add_dpratio_inset(
        ax_schematic,
        dp_color_inset_position,
        dpratio_colors,
        dpratios,
        label="Extra LTD (%)",
        fontsize=FigParams.fontsize,
        label_padding=dp_color_label_padding,
        reverse=True,
    )

    # Table Data
    row_data = [["Strength", "Strong", "Weak", "Weak"], ["Extra LTD", "High", "High", "Variable"]]
    row_colors = ["black", *group_colors]
    x_positions = np.linspace(0, 1, 4 + 2)[1:-1]
    x_positions[0] *= 0.85
    y_positions = [0.75, 0.25]

    # Calculate bounding boxes for text items
    text_objects = []
    label_objects = []
    for row_idx, row_labels in enumerate(row_data):
        for col_idx, col_label in enumerate(row_labels):
            text = ax_table.text(
                x_positions[col_idx],
                y_positions[row_idx],
                col_label,
                ha="center",
                va="center",
                color=row_colors[col_idx],
                fontsize=main_neuron_label_fontsize,
            )
            if col_idx == 0:
                label_objects.append(text)
            else:
                text_objects.append(text)

    # Draw boundary lines based on bounding boxes
    ax_table.figure.canvas.draw()  # Draw the canvas to update text positions
    bboxes_text = [text.get_window_extent() for text in text_objects]
    bboxes_text_data = [bbox.transformed(ax_table.transData.inverted()) for bbox in bboxes_text]
    bboxes_label = [text.get_window_extent() for text in label_objects]
    bboxes_label_data = [bbox.transformed(ax_table.transData.inverted()) for bbox in bboxes_label]

    # Calculate horizontal line position
    left_most = min(bbox.x0 for bbox in bboxes_label_data)
    right_most = max(bbox.x1 for bbox in bboxes_text_data)
    ax_table.plot(
        [left_most, right_most],
        [0.5, 0.5],
        color="black",
        linewidth=FigParams.thinlinewidth,
    )

    # Calculate vertical line position
    first_col_right = max([bld.x1 for bld in bboxes_label_data])
    second_col_left = min([bld.x0 for bld in bboxes_text_data])
    vertical_line_x = (first_col_right + second_col_left) / 2
    ax_table.plot(
        [vertical_line_x, vertical_line_x],
        y_positions,
        color="black",
        linewidth=FigParams.thinlinewidth,
    )

    # Hide axes
    ax_table.set_xticks([])
    ax_table.set_yticks([])
    for spine in ax_table.spines.values():
        spine.set_visible(False)
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)

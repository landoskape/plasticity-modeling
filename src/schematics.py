import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from .plotting import Proximal, DistalSimple, DistalComplex


class Neuron:
    def __init__(
        self,
        linewidth=2.5,
        soma_size=1.0,
        soma_filled=True,
        trunk_height=1.75,
        tuft_height=3.0,
        trunk_color=Proximal.color,
        simple_tuft_angle=20,
        simple_tuft_color=DistalSimple.color,
        complex_tuft_angle=20,
        complex_tuft_color=DistalComplex.color,
        complex_tuft_branches=5,
        complex_tuft_branch_fraction=0.3,
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
        """
        self.linewidth = linewidth
        self.soma_size = soma_size
        self.soma_filled = soma_filled
        self.trunk_height = trunk_height
        self.tuft_height = tuft_height
        self.trunk_color = trunk_color
        self.simple_tuft_angle = simple_tuft_angle
        self.simple_tuft_color = simple_tuft_color
        self.complex_tuft_angle = complex_tuft_angle
        self.complex_tuft_color = complex_tuft_color
        self.complex_tuft_branches = complex_tuft_branches
        self.complex_tuft_branch_fraction = complex_tuft_branch_fraction

    def plot(self, ax, origin=(0, 0), scale=1.0, **kwargs):
        """
        Create the neuron schematic on the given matplotlib axis.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis on which to draw the neuron
        origin : tuple of float
            The (x, y) coordinates for the center of the soma
        scale : float
            Scale factor to adjust the overall size of the neuron
        **kwargs : dict
            Additional parameters that can override instance attributes

        Returns:
        --------
        self : for method chaining
        """
        # Elements created during rendering
        elements = {"soma": None, "trunk": None, "simple_tuft": None, "complex_tuft": [], "complex_branches": []}

        # Update attributes with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        x0, y0 = origin

        # Calculate the radius of the soma
        soma_radius = self.soma_size * scale / 2

        # Calculate the lengths for the tufts so they reach the same height
        simple_angle_rad = np.radians(90 - self.simple_tuft_angle)
        complex_angle_rad = np.radians(90 - self.complex_tuft_angle)

        # Calculate length needed to reach tuft_height at each angle
        simple_length = self.tuft_height * scale / np.sin(simple_angle_rad)
        complex_length = self.tuft_height * scale / np.sin(complex_angle_rad)

        # Calculate trunk line
        trunk_length = self.trunk_height * scale
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
            linewidth=self.linewidth * scale,
            color=self.complex_tuft_color,
        )
        elements["complex_tuft"].append(complex_base)
        ax.add_line(complex_base)

        # Draw complex tuft branches
        branch_angles = np.linspace(
            -np.radians(self.complex_tuft_angle),
            np.radians(self.complex_tuft_angle),
            self.complex_tuft_branches,
        )

        # Calculate remaining height to reach tuft_height
        remaining_height = self.tuft_height * scale - complex_dy * self.complex_tuft_branch_fraction

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
                linewidth=self.linewidth * scale,
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
            [x2, simple_x_end], [y2, simple_y_end], linewidth=self.linewidth * scale, color=self.simple_tuft_color
        )
        ax.add_line(elements["simple_tuft"])

        # Draw trunk
        elements["trunk"] = Line2D([x1, x2], [y1, y2], linewidth=self.linewidth * scale, color=self.trunk_color)
        ax.add_line(elements["trunk"])

        # Draw soma (circle)
        fill = self.soma_filled
        elements["soma"] = Circle(
            origin,
            soma_radius,
            facecolor="black" if fill else "white",
            edgecolor="black",
            linewidth=self.linewidth * scale,
        )
        ax.add_patch(elements["soma"])

        return elements

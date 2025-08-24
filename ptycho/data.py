from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
from ptychoep.backend.backend import np

IdxType = Optional[Tuple[slice, slice]]


@dataclass
class DiffractionData:
    """
    Container class representing a single diffraction pattern and its metadata.

    This class holds a 2D diffraction image along with associated position,
    optional index slicing information, and other auxiliary metadata such as gamma_w.

    Attributes
    ----------
    position : Tuple[int, int]
        The scan position (y, x) on the object grid.
    diffraction : np.ndarray
        The complex-valued diffraction pattern.
    meta : dict
        A dictionary to store auxiliary metadata (e.g., scan index, intensity stats).
    indices : Optional[Tuple[slice, slice]]
        Slice object indexing into the object array corresponding to this scan.
    gamma_w : Optional[float]
        Precision parameter used in uncertainty modeling (optional).
    """

    position: Tuple[int, int]
    diffraction: np().ndarray
    meta: dict = field(default_factory=dict)
    indices: IdxType = None
    gamma_w: Optional[float] = None

    def intensity(self) -> np().ndarray:
        """
        Return the intensity of the diffraction pattern (squared amplitude).

        Returns
        -------
        np.ndarray
            Real-valued intensity image.
        """
        return np().abs(self.diffraction) ** 2

    def get_gamma_w(self):
        """
        Return the precision value gamma_w if set.

        Raises
        ------
        ValueError
            If gamma_w is not set.
        """
        if self.gamma_w is None:
            raise ValueError("gamma_w is not set for this diffraction data.")
        return self.gamma_w

    def show(self, ax=None, log_scale=True, cmap="jet"):
        """
        Visualize the diffraction pattern using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axis to draw on. If None, a new figure is created.
        log_scale : bool
            Whether to apply log10 scaling for better visibility.
        cmap : str
            Colormap used for plotting.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the plot.
        """
        import matplotlib.pyplot as plt
        from ptychoep.backend.backend import np

        data = self.diffraction
        if log_scale:
            data = np().log10(np().abs(data) + 1e-8)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(data, cmap=cmap)
        ax.set_title(f"Pos: {self.position}")
        ax.axis("off")
        return ax

    def summary(self) -> str:
        """
        Return a string summarizing the diffraction data.

        Returns
        -------
        str
            Summary of position, shape, and metadata.
        """
        return f"Pos={self.position}, shape={self.diffraction.shape}, meta={self.meta}"

    def __hash__(self):
        """
        Hash function based on object identity.
        """
        return id(self)

    def __eq__(self, other):
        """
        Equality check based on object identity.
        """
        return self is other

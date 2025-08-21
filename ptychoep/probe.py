from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from .fft_channel import FFTChannel
from typing import Optional, Any
from PtychoEP.backend.backend import np
from PtychoEP.ptycho.data import DiffractionData

class Probe:
    def __init__(
        self,
        data: np().ndarray,
        parent: Optional["Object"] = None,
        diffraction: Optional[DiffractionData] = None
    ):
        """
        Initialize the Probe object.

        Parameters
        ----------
        data : np.ndarray
            The complex-valued probe pattern.
        parent : Object or None
            The parent Object node this probe belongs to.
        diffraction : DiffractionData or None
            The measurement node associated with this probe.
        """
        self.dtype = data.dtype
        self.set_data(data)

        self.shape = data.shape
        self.parent = parent
        self.diff = diffraction

        self.input_belief: Optional[UA] = None
        self.msg_to_object: Optional[UA] = None

        # Create FFTChannel child node and link back
        self.child = FFTChannel(parent_probe=self, diff = self.diff)



    def set_data(self, data: np().ndarray,
                 data_abs: np().ndarray | None = None,
                 data_inv: np().ndarray | None = None):
        """
        Set the probe data and optionally cache its derived quantities.

        Parameters
        ----------
        data : np.ndarray
            Complex-valued probe (2D). Must have ndim == 2.
        data_abs : np.ndarray or None
            Optional precomputed |data|^2 (squared magnitude). If not provided,
            it will be computed internally with numerical stabilization.
        data_inv : np.ndarray or None
            Optional precomputed conjugate(data) / |data|^2. If not provided,
            it will be computed from data and data_abs.

        Notes
        -----
        When multiple Probe instances share the same data (e.g., in ptychography),
        supplying shared precomputed data_abs and data_inv avoids redundant computation.
        """
        xp = np()
        arr = xp.asarray(data, dtype=self.dtype)
        if arr.ndim != 2:
            raise ValueError("Probe.data must be 2D.")
        self.data = arr

        if data_abs is None:
            self.abs2 = xp.maximum(xp.abs(arr) ** 2, 1e-8)
        else:
            self.abs2 = data_abs

        if data_inv is None:
            self.data_inv = self.data.conj() / self.abs2
        else:
            self.data_inv = data_inv


    def forward(self) -> None:
        """
        Forward message propagation from Probe to FFTChannel.

        This method applies a pixel-wise complex scaling (Hadamard product)
        to the input belief using the probe pattern, then stores the result 
        in the child FFTChannel as its `input_belief`.

        Raises
        ------
        RuntimeError
            If no input belief has been set.
        """
        xp = np()
        if self.input_belief is None:
            raise RuntimeError("Probe.forward : no input belief")
        #out = self.input_belief.scaled(gain=self.data)
        new_mean =  self.input_belief.mean * self.data
        new_prec = self.input_belief.precision / self.abs2
        self.child.input_belief = UA(mean = new_mean, precision = new_prec, dtype = self.dtype)


    def backward(self) -> None:
        """
        Backward message propagation from FFTChannel to Object.

        This method receives a message from the child FFTChannel (i.e., the 
        inverse FFT output), and scales it by the complex conjugate of the 
        probe field to produce the message to be sent back to the Object node.
        """
        xp = np()
        msg_from_fft = self.child.msg_to_probe
        #self.msg_to_object = msg_from_fft.scaled(self.data_inv)
        new_mean = msg_from_fft.mean * self.data_inv
        new_prec = msg_from_fft.precision * self.abs2
        self.msg_to_object = UA(mean = new_mean, precision = new_prec, dtype = self.dtype)

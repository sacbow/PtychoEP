from .uncertain_array import UncertainArray as UA, fft_ua, ifft_ua
from backend.backend import np
from typing import Optional
from ptycho.data import DiffractionData

from .uncertain_array import UncertainArray as UA
from backend.backend import np
from ptycho.data import DiffractionData
from typing import Optional
from .denoiser import Denoiser


class FFTChannel:
    """
    Represents the FFT node between a Probe and the Output Denoiser.

    This node receives the forward-transformed exit wave and sends it to 
    the denoiser, and also propagates backward messages to the Probe.
    """

    def __init__(self, parent_probe, diff: Optional[DiffractionData] = None):
        """
        Initialize the FFTChannel node.

        Parameters
        ----------
        parent_probe : Probe
            The parent Probe object that this channel is connected to.
        diff : DiffractionData or None
            The diffraction data node associated with this channel.
        """
        self.probe = parent_probe
        self.diff = diff
        self.denoiser = Denoiser(diff = diff, parent = self)
        self.input_belief: Optional[UA] = None    # From Probe (exit wave before FFT)
        self.msg_to_probe: Optional[UA] = None    # Backward message to Probe
        self.msg_from_denoiser: Optional[UA] = None  # Message from OutputDenoiser (z-domain)

        self.initialize_msg_from_denoiser()

    def initialize_msg_from_denoiser(self):
        """
        Populate the initial message from the denoiser.

        This is done by computing the FFT of the estimated exit wave, defined as:
            z0 = FFT(probe * object_patch)

        The resulting UncertainArray is stored as msg_from_denoiser, with 
        mean = z0 and precision = 1.0 (scalar).
        """
        xp = np()

        if self.probe is None or self.probe.parent is None or self.diff is None:
            raise RuntimeError("FFTChannel requires valid probe, object, and diffraction data.")

        obj = self.probe.parent
        indices = self.diff.indices
        patch = obj.object_init[indices]
        probe_data = self.probe.data

        exit_wave = probe_data * patch
        z0 = xp.fft.fft2(exit_wave, norm="ortho")
        self.msg_from_denoiser = UA(mean=z0, precision=1.0, dtype=z0.dtype)

    def forward(self) -> None:
        """
        Perform FFT forward message passing from Probe to OutputDenoiser.

        This computes:
            msg_to_output = FFT(input_belief) / msg_from_denoiser

        The result is stored in:
            denoiser.msg_from_fft ← UA (scalar precision)

        Notes
        -----
        - `input_belief` is the UA from the Probe (exit wave).
        - `msg_from_denoiser` acts as a prior message in the z-domain.
        - OutputDenoiser must have already been linked to `self.denoiser`.
        """
        if self.input_belief is None:
            raise RuntimeError("FFTChannel.forward: input_belief is None")

        # Forward FFT: exit wave → diffraction domain
        output_belief = fft_ua(self.input_belief)  # Always scalar precision

        # Send message to denoiser
        self.denoiser.msg_from_fft = output_belief / self.msg_from_denoiser

    def backward(self) -> None:
        """
        Perform backward message passing from OutputDenoiser to Probe.

        This applies inverse FFT to the denoiser's message, transforming it 
        back into the object domain and storing it as msg_to_probe.

        Notes
        -----
        - Assumes that msg_from_denoiser has already been updated
          by OutputDenoiser.backward().
        - msg_to_probe will be consumed by Probe.backward().
        """
        if self.msg_from_denoiser is None:
            raise RuntimeError("FFTChannel.backward: msg_from_denoiser is None")

        self.msg_to_probe = ifft_ua(self.msg_from_denoiser)


        

from __future__ import annotations
from .accumulative_uncertain_array import AccumulativeUncertainArray as AUA
from .uncertain_array import UncertainArray as UA
from backend.backend import np
from rng.rng_utils import get_rng, normal
from ptycho.data import DiffractionData
from .probe import Probe
from .prior import BasePrior, SparsePrior


class Object:
    """
    Object node in the EP-based ptychography graph.

    This class maintains the global estimate of the object as an 
    AccumulativeUncertainArray (AUA), tracks messages exchanged 
    with data nodes (via probes), and holds references to all 
    associated probe, channel, and denoiser objects.

    Attributes
    ----------
    shape : tuple
        Shape of the object image (typically square: H x W).
    dtype : np.dtype
        Data type of the internal arrays (usually np.complex64).
    rng : Generator
        Random number generator used for initializations.
    object_init : np.ndarray
        Initial complex-valued guess of the object. (randomly initialized if not given)
    probe_init : np.ndarray
        Initial complex-valued probe used for each measurement. 
    belief : AccumulativeUncertainArray
        Accumulated posterior (product form) of all incoming messages.
    msg_from_prior : UncertainArray
        Message from the prior node (can be zero or identity if not used).
    msg_from_data : dict[DiffractionData, UncertainArray]
        Messages from each data node (i.e., from probes via FFT → output).
    data_registry : dict[DiffractionData, tuple[slice, slice]]
        Patch location of each DiffractionData relative to the object image.
    probe_registry : dict[DiffractionData, Probe]
        Mapping from each DiffractionData to its associated Probe object.
    """

    def __init__(self, shape, rng, initial_probe: np().ndarray,
                 dtype=np().complex64, initial_object: np().ndarray | None = None):
        # Basic attributes
        self.shape = shape
        self.dtype = dtype

        # Random initialization
        self.rng = rng if rng is not None else get_rng()
        self.object_init = initial_object if initial_object is not None else normal(rng=self.rng, size=self.shape)
        self.probe_init = initial_probe

        self.prior = None

        # Belief and messages
        self.belief = AUA(shape=shape, dtype=dtype)
        self.msg_from_prior: UA = UA.zeros(shape=shape, scalar_precision=False)
        self.msg_from_data: dict[DiffractionData, UA] = {}

        # Pointers to external components
        self.data_registry: dict[DiffractionData, tuple[slice, slice]] = {}
        self.probe_registry: dict[DiffractionData, Probe] = {}
    
    def set_prior(self, prior_name = "gaussian", **prior_kwarg):
        if prior_name == "sparse":
            self.prior = SparsePrior(self, prior_kwarg)

    def register_data(self, diff: DiffractionData):
        """
        Register a new diffraction data object to the object node.

        This initializes:
        - The slice index mapping from object to this data.
        - A corresponding message from object to data.
        - A probe object associated with this data point.

        Parameters
        ----------
        data : DiffractionData
            The measurement node to associate.
        """
        if diff.indices is None:
            raise ValueError(f"indices not set for data at position {diff.position}")

        # Register slice location
        self.data_registry[diff] = diff.indices

        # Create and register corresponding Probe
        prb = Probe(data = self.probe_init, parent = self, diffraction = diff)
        self.probe_registry[diff] = prb

        # Initialize message and belief update
        init_msg = UA(mean=self.object_init[diff.indices]).to_array_precision()
        self.msg_from_data[diff] = init_msg
        self.belief.add(init_msg, diff.indices)

    
    def forward(self, data: DiffractionData) -> None:
        """
        Send a forward message from the object to the associated probe.

        This method extracts the relevant patch of the current belief (UA) 
        corresponding to the specified diffraction data, and sets it as the 
        input belief for the associated probe.

        Parameters
        ----------
        data : DiffractionData
            The data point whose corresponding probe should receive the message.

        Raises
        ------
        KeyError if the given data is not registered to the object.
        """

        ua_to_probe = self.get_patch_ua(data)
        prb = self.probe_registry[data]
        prb.input_belief = ua_to_probe


    def get_patch_ua(self, data: DiffractionData) -> UA:
        """
        Extract the belief patch corresponding to a specific data point.

        Returns the local UncertainArray slice of the object's belief 
        that corresponds to the patch indexed by the given data.

        Parameters
        ----------
        data : DiffractionData
            The data node whose patch should be extracted.

        Returns
        -------
        UA : UncertainArray
            The patch of the current belief corresponding to the data's slice.

        Raises
        ------
        ValueError if the given data is not registered to the object.
        """
        indices = self.data_registry.get(data)
        if indices is None:
            raise ValueError("Data not registered to object")
        return self.belief.get_ua(indices)
    
    def backward(self, data: DiffractionData) -> None:
        """
        Receive a backward message from the associated probe and update the object's belief.

        This method updates the object's accumulated belief by first subtracting the old
        message (from data) and then adding the new message obtained from the probe.

        Parameters
        ----------
        data : DiffractionData
            The data node whose associated probe has computed a new backward message.

        Raises
        ------
        KeyError if the given data is not registered or missing in msg_from_data.
        """
        # new and old msg_from_data
        prb = self.probe_registry[data]
        new_msg = prb.msg_to_object
        old_msg = self.msg_from_data[data]

        # update belief and msg_from_data
        indices = self.data_registry[data]
        self.belief.subtract(old_msg, indices)
        self.belief.add(new_msg, indices)
        self.msg_from_data[data] = new_msg

    def get_belief(self) -> UA:
        """
        Return the current global belief (posterior) of the object as a UncertainArray.

        This is typically used at the end of inference to extract the final estimated image.

        Returns
        -------
        UncertainArray
            The object's full belief over the entire field.
        """
        return self.belief.to_ua()


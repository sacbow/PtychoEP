from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from ptychoep.backend.backend import np


class BasePrior:
    """
    BasePrior represents a generic prior node in the EP-based ptychography framework.

    This class defines the interface and shared logic for all prior models. It handles the 
    message exchange between the object node and the prior, including belief updates and 
    message replacements in the global object belief.

    Subclasses should implement the `compute_belief()` method to define a specific prior 
    distribution (e.g., sparsity, total variation).

    Attributes
    ----------
    object : Object
        The object node to which this prior is attached.
    msg_from_object : UncertainArray
        The incoming message from the object (belief / previous prior message).
    belief : UncertainArray
        The approximate posterior under the prior.
    msg_to_object : UncertainArray
        The outgoing message to be sent back to the object.
    """

    def __init__(self, obj: "Object", prior_kwargs=None):
        self.dtype = obj.dtype
        self.shape = obj.shape
        self.object = obj

        self.msg_from_object: UA | None = None
        self.belief: UA | None = None
        self.msg_to_object: UA | None = None

    def forward(self):
        """
        Perform a full EP update between the object and the prior.

        This involves:
        1. Receiving the current belief from the object.
        2. Computing the approximate posterior belief under the prior model.
        3. Sending a new message back to the object and updating its belief accordingly.
        """
        self.msg_from_object = self.object.get_belief() / self.object.msg_from_prior
        self.compute_belief()
        self.msg_to_object = self.belief / self.msg_from_object
        self.object.belief.subtract(self.object.msg_from_prior)
        self.object.belief.add(self.msg_to_object)
        self.object.msg_from_prior = self.msg_to_object

    def compute_belief(self):
        """
        Compute the posterior approximation under the prior distribution.

        This method must be implemented by subclasses to define the actual prior behavior.
        """
        raise NotImplementedError("compute_belief() must be implemented in subclass")


class SparsePrior(BasePrior):
    """
    SparsePrior implements a spike-and-slab prior model promoting sparsity in the object.

    The model assumes a mixture of zero-valued (spike) and Gaussian (slab) components, and 
    applies a pixel-wise update based on the EP message from the object.

    Parameters
    ----------
    rho : float
        The sparsity parameter (0 < rho < 1), controlling the weight of the slab component.
    """

    def __init__(self, obj: "Object", prior_kwargs=None):
        super().__init__(obj, prior_kwargs)
        if prior_kwargs is None:
            prior_kwargs = {}
        self.rho = prior_kwargs.get("sparsity", 0.1)

    def compute_belief(self):
        """
        Compute the posterior belief under a spike-and-slab prior using closed-form updates.

        This follows standard expressions for Gaussian mixture models, using the current
        message from the object to compute a pixel-wise posterior mean and variance.
        """
        if self.msg_from_object is None:
            raise RuntimeError("SparsePrior: msg_from_object is None")

        m = self.msg_from_object.mean
        v = 1.0 / self.msg_from_object.precision

        v_post = 1.0 / (1.0 + 1.0 / v)
        m_post = v_post * (m / v)

        slab = self.rho * np().exp(-np().abs(m) ** 2 / (1.0 + v)) / (1.0 + v)
        spike = (1 - self.rho) * np().exp(-np().abs(m) ** 2 / v) / v
        Z = slab + spike + 1e-8

        mu = (slab / Z) * m_post
        e_x2 = (slab / Z) * (np().abs(m_post) ** 2 + v_post)
        var = np().maximum(e_x2 - np().abs(mu) ** 2, 1e-8)

        precision = 1.0 / var
        self.belief = UA(mean=mu, precision=precision, dtype=self.dtype)

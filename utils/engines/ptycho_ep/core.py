# utils/engines/ptycho_ep/core.py

from ...backend import np
from ...rng_utils import get_rng
from ...ptycho.data import DiffractionData
from .object import Object
from .uncertain_array import UncertainArray as UA

class PtychoEP:
    """
    Expectation Propagation (EP)-based ptychographic solver.
    """

    def __init__(self, ptycho, damping=0.7, seed : int | None = None, obj_init = None, prb_init=None, callback=None):
        """
        Parameters
        ----------
        ptycho : Ptycho
            Ptycho object holding object/probe/diffraction geometry.
        damping : float
            Damping coefficient used in denoiser backward pass.
        prb_init : np.ndarray or None
            Optional probe initialization. If None, uses ptycho.prb.
        callback : callable or None
            Function to be called after each iteration: callback(iter, error, object_est).
        """
        self.xp = np()
        self.ptycho = ptycho
        self.damping = damping
        self.callback = callback
        self.prior = None  # Optional prior denoiser node (set externally if needed)

        rng = get_rng(seed)

        # --- Initialize object node ---
        self.obj_node = Object(
            shape=(ptycho.obj_len, ptycho.obj_len),
            rng=rng,
            initial_probe=prb_init if prb_init is not None else ptycho.prb,
            initial_object=obj_init if obj_init is not None else None
        )

        # Register each diffraction data to the object
        for diff in ptycho._diff_data:
            self.obj_node.register_data(diff)
            # damping is passed into each denoiser internally via FFTChannel
            self.obj_node.probe_registry[diff].child.denoiser.damping = damping

    def run(self, n_iter=100):
        """
        Run the EP update loop for a given number of iterations.

        Parameters
        ----------
        n_iter : int
            Number of EP iterations.

        Returns
        -------
        object_estimate : np.ndarray
            Final estimated object (complex-valued image).
        probe_estimate : np.ndarray
            Probe pattern (unchanged if not updated during inference).
        """
        xp = self.xp
        for it in range(n_iter):
            for diff in self.ptycho._diff_data:
                # Expectation propagation
                self.obj_node.forward(diff)
                probe = self.obj_node.probe_registry[diff]
                probe.forward()
                probe.child.forward()
                probe.child.denoiser.backward()
                probe.child.backward()
                probe.backward()
                self.obj_node.backward(diff)

            # Update prior (if applicable)
            if self.prior is not None:
                self.prior.receive_msg(self.obj_node.send_msg_to_prior())
                msg_prior = self.prior.forward_msg()
                self.obj_node.receive_msg_from_prior(msg_prior)

            # Optional callback
            if self.callback:
                mean_err = xp.mean(xp.array([p.child.denoiser.error for p in self.obj_node.probe_registry.values()]))
                self.callback(it, float(mean_err), self.obj_node.get_belief().mean)

        return self.obj_node.get_belief().mean, self.obj_node.probe_registry[diff].data

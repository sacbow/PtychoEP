from ptychoep.backend.backend import np
from ptychoep.rng.rng_utils import get_rng
from ptychoep.ptycho.data import DiffractionData
from .object import Object
from .uncertain_array import UncertainArray as UA

class PtychoEP:
    """
    Expectation Propagation (EP)-based ptychographic solver.
    """

    def __init__(self, ptycho, damping=0.7, seed: int | None = None,
                 obj_init=None, prb_init=None, prior_name="gaussian",
                 callback=None, n_probe_update : int = 0, **prior_kwargs):
        """
        Parameters
        ----------
        ptycho : Ptycho
            Ptycho object holding object/probe/diffraction geometry.
        damping : float
            Damping coefficient used in Likelihood backward pass.
        obj_init : np.ndarray or None
            Optional object initialization.
        prb_init : np.ndarray or None
            Optional probe initialization.
        prior_name : str
            Name of prior to use ("gaussian" implies no prior).
        callback : callable or None
            Function to call after each iteration: callback(iter, error, object_est).
        """
        self.xp = np()
        self.ptycho = ptycho
        self.damping = damping
        self.callback = callback

        rng = get_rng(seed)

        # --- Initialize object node ---
        self.obj_node = Object(
            shape=(ptycho.obj_len, ptycho.obj_len),
            rng=rng,
            initial_probe = prb_init if prb_init is not None else ptycho.prb,
            initial_object = obj_init
        )
        self.obj_node.set_prior(prior_name, **prior_kwargs)

        # --- Register diffraction data and assign Likelihood damping ---
        for diff in ptycho._diff_data:
            self.obj_node.register_data(diff)
            self.obj_node.probe_registry[diff].child.likelihood.damping = damping
        
        # initialize probe update (optional)
        self.n_probe_update = n_probe_update
        if n_probe_update > 0:
            from .probe_updater import ProbeUpdater
            self.probe_updater = ProbeUpdater(self.obj_node) 


    def run(self, n_iter=100):
        """
        Run the EP update loop for a given number of iterations.

        Parameters
        ----------
        n_iter : int
            Number of EP iterations.
        n_probe_update : int
            Number of probe EM updates per iteration. Set to 0 to disable.

        Returns
        -------
        object_estimate : np.ndarray
            Final estimated object (complex-valued image).
        precision_estimate : np.ndarray
            Estimated posterior precision.
        """
        xp = self.xp
        for it in range(n_iter):
            # Optional prior update (if not gaussian)
            if self.obj_node.prior:
                self.obj_node.prior.forward()

            for diff in self.ptycho._diff_data:
                self.obj_node.forward(diff)
                probe = self.obj_node.probe_registry[diff]
                probe.forward()
                probe.child.forward()
                probe.child.likelihood.backward()
                probe.child.backward()
                probe.backward()
                self.obj_node.backward(diff)
            
            # --- Probe EM update ---
            if self.n_probe_update > 0:
                self.probe_updater.update(n_iter=self.n_probe_update)

            # Optional callback
            if self.callback:
                mean_err = xp.mean(xp.array([
                    p.child.likelihood.error for p in self.obj_node.probe_registry.values()
                ]))
                self.callback(it, float(mean_err), self.obj_node.get_belief().mean)
        # output results
        obj_estimate = self.obj_node.get_belief() # Uncertain Array
        if self.n_probe_update == 0:
            return obj_estimate.mean, obj_estimate.precision
        else:
            probe_estimate = self.obj_node.probe_registry[self.ptycho._diff_data[0]].data
            return obj_estimate.mean, obj_estimate.precision, probe_estimate

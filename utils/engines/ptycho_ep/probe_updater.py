from __future__ import annotations
from .object import Object
from ...backend import np

class ProbeUpdater:
    """
    EM-based probe update engine.

    This class implements the update of the global probe field based on the 
    current object belief and the messages received from each FFTChannel node.

    Optionally, it also updates the noise precision term (adaptive EM).
    """

    def __init__(self, obj_node: Object):
        """
        Parameters
        ----------
        obj_node : Object
            The object node containing current belief and all probe instances.
        """
        self.obj_node = obj_node
        self.xp = np()

    def update(self, n_iter: int = 1):
        """
        Perform multiple EM updates of the probe and its associated precision parameters.

        Parameters
        ----------
        n_iter : int
            Number of EM update steps to run. The object belief is fixed during these steps.
        """
        # --- Collect patches ---
        xp = self.xp
        full_belief = self.obj_node.belief.to_ua()
        O_mu_list = []
        O_var_list = []
        Phi_list = []
        gamma_list = []
        for diff, probe in self.obj_node.probe_registry.items():
                indices = self.obj_node.data_registry[diff]
                O_mu = full_belief.mean[indices]
                O_var = 1.0 / full_belief.precision[indices]
                Phi = probe.child.msg_to_probe.mean
                gamma = probe.child.msg_from_denoiser.precision
                O_mu_list.append(O_mu)
                O_var_list.append(O_var)
                Phi_list.append(Phi)
                gamma_list.append(gamma)

        # --- Stack into arrays ---
        O_mu_all = xp.stack(O_mu_list, axis=0)            # (N, H, W)
        O_var_all = xp.stack(O_var_list, axis=0)          # (N, H, W)
        Phi_all = xp.stack(Phi_list, axis=0)              # (N, H, W)
        gamma_all = xp.array(gamma_list).reshape(-1, 1, 1)

        # --- precompute constant terms ---
        numerator_terms = xp.conj(O_mu_all) * Phi_all
        denominator_terms = xp.abs(O_mu_all)**2 + O_var_all

        for _ in range(n_iter):
            # --- EM Update of probe ---
            P1 = xp.sum(gamma_all * numerator_terms, axis=0)
            P2 = xp.sum(gamma_all * denominator_terms, axis=0)
            P_est = P1 / P2

            # --- Adaptive EM: update precision ---
            gamma_all = 1 / xp.mean(xp.abs(Phi_all - O_mu_all * P_est)**2, axis = (1,2))

        # Assigns all probes
        P_abs2 = xp.maximum(xp.abs(P_est) ** 2, 1e-8)
        P_inv = xp.conj(P_est) / P_abs2
        for probe in self.obj_node.probe_registry.values():
            probe.set_data(P_est, data_abs=P_abs2, data_inv=P_inv)

        #Assign to all precisions
        for i, probe in enumerate(self.obj_node.probe_registry.values()):
            probe.child.msg_from_denoiser.precision = gamma_all[i].item()



from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from ...backend import np
from ...rng_utils import get_rng, normal

class BasePrior:
    def __init__(self, obj : "Object", prior_kwargs = None):
        self.dtype = obj.dtype
        self.shape = obj.shape
        self.object = obj
        self.msg_from_object: UA | None = None
        self.belief: UA | None = None
        self.msg_to_object: UA | None = None
    
    def forward(self):
        if self.msg_from_object:
            self.compute_belief()
            self.msg_to_object = self.belief / self.msg_from_object
    
    def compute_belief(self):
        raise NotImplementedError

class SparsePrior(BasePrior):
    def __init__(self, obj : "Object", **prior_kwargs):
        super().__init__(self, obj)
        self.rho = prior_kwargs["sparsity"]

    def compute_belief(self):
        m = self.msg_from_object.data
        v = 1.0 / self.msg_from_object.precision(raw=True)

        prec_post = 1.0 + 1.0 / v
        v_post = 1.0 / prec_post
        m_post = v_post * (m / v)

        slab = self.rho * np().exp(-np().abs(m)**2 / (1 + v)) / ((1 + v))
        spike = (1 - self.rho) * np().exp(-np().abs(m)**2 / v) / v

        Z = slab + spike + 1e-8  # normalization constant

        mu = (slab / Z) * m_post
        e_x2 = (slab / Z) * (np().abs(m_post) ** 2 + v_post)
        var = np().maximum(e_x2 - np().abs(mu) ** 2, 1e-8)
        precision = 1.0 / var

        self.belief = UA(mu, dtype=self.dtype, precision=precision)
    

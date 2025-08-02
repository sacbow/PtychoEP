#PIE
class config_PIE():
    def __init__(self, alpha, object_init, probe, num_iter):
        self.alpha = alpha
        self.obj_init = object_init
        self.prb = probe
        self.num_iter = num_iter


#ePIE
class config_ePIE():
    def __init__(self, alpha, beta, object_init, probe_init, num_iter):
        self.alpha = alpha
        self.beta = beta
        self.obj_init = object_init
        self.prb_init = probe_init
        self.num_iter = num_iter
#DM
class config_DM():
    def __init__(self, object_init, probe_init, num_iter):
        self.obj_init = object_init
        self.prb_init = probe_init
        self.num_iter = num_iter

#PtychoEP
class config_PtychoEP():
    def __init__(self, object_init, probe_init, num_iter, num_prb, damping):
        self.object_init = object_init
        self.probe_init = probe_init
        self.num_iter = num_iter
        self.num_prb = num_prb
        self.damping = damping

class config_PtychoEP_Sparse():
    def __init__(self, object_init, probe_init, num_iter, num_prb, damping, rho):
        self.object_init = object_init
        self.probe_init = probe_init
        self.num_iter = num_iter
        self.damping = damping
        self.num_prb = num_prb
        self.rho = rho
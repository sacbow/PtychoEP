#import modules
import numpy as np
from numpy import linalg as LA
from numpy import exp

# known probe
def error_metric(x1, x2):
    theta = np.angle(np.sum(x1.conj()*x2))
    return LA.norm(x1 - x2*exp(-1j*theta))**2/LA.norm(x1)**2

#unknown probe
def error_metric_normalize(x1, x2):
    x2_norm = x2 * LA.norm(x1)/LA.norm(x2)
    theta = np.angle(np.sum(x1.conj()*x2))
    return LA.norm(x1 - x2_norm*exp(-1j*theta))**2/LA.norm(x1)**2


#imports
import numpy as np

def Message_Passing(r_pre, gamma_pre, x, eta):
    gamma_new = np.maximum(eta - gamma_pre, 1)
    r_new = (eta*x - gamma_pre * r_pre)/gamma_new
    return r_new, gamma_new

def Damping(r_raw, gamma_raw, r_old, gamma_old, damping):
    r = damping*r_raw + (1-damping)*r_old
    gamma = 1/(damping/(gamma_raw**0.5) + (1-damping)/(gamma_old**0.5))**2
    return r, gamma

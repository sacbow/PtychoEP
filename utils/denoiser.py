#imports
import numpy as np
from numpy import exp, log, pi

#Gaussian prior N(0,1)
# r, gamma, x, eta are 2d-array
def gaussian_complex_input_denoiser(r, gamma):
    x = r*gamma/(1+gamma)
    eta = gamma + 1
    return x, eta

#Bernoulli-Gaussian prior (with sparsity estimation)
def sparse_complex_input_denoiser(r, gamma, rho):
    A = np.maximum(rho*exp(-(gamma/(gamma+1))*np.abs(r)**2)/(1 + gamma), 1e-20)
    B = (1-rho)*exp(-gamma*np.abs(r)**2)
    pi = A/(A + B)
    one_minus_pi = B/(A+B)
    g = pi*(gamma/(gamma+1))*r
    dg = pi*(gamma/(gamma+1))*(1 + gamma*one_minus_pi*(gamma/(gamma+1))*np.abs(r)**2)
    return g, gamma/dg

#PR denoiser; y = |z| + N(0, 1/gamma_w)
def PR_output_denoiser(p, tau, y, gamma_w):
    p_abs = np.abs(p)
    p_ang = np.angle(p)
    y_p = np.mean(y/np.maximum(p_abs, 1e-12))
    z = np.exp(1j * p_ang) * (tau*p_abs + (gamma_w/2)*y)/(tau + (gamma_w/2))
    lam = 2*(gamma_w + 2*tau)/(gamma_w*y_p/tau + 4)
    return z, lam
#imports
import numpy as np
from scipy import stats
from numpy import linalg as LA


#(N, N) 2d array with entries i.i.d. drawn from circular complex Gaussian distribution
def complex_gaussian_matrix(N, var):
    return np.random.normal(loc = 0, scale = (var/2)**0.5, size = (N, N)) + 1j * np.random.normal(loc = 0, scale = (var/2)**0.5, size = (N, N))

# (N,N) 2d array with i.i.d. Gaussian-Bernoulli entries
def complex_sparse_matrix(N, rho, var):
    return complex_gaussian_matrix(N, var) * stats.bernoulli.rvs(p = rho, size = (N, N))
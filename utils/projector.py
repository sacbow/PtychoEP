#import modules
import numpy as np
from numpy.random import randint
from numpy import exp, pi
from numpy import linalg as LA
from numpy.fft import fft2, ifft2, fftshift

#fourier space constraint
def Fourier_projector(exit_wave, diff):
    return ifft2(diff*exp(1j*np.angle(fft2(exit_wave))))*len(exit_wave)

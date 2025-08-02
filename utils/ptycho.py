#import modules
import numpy as np
from numpy.random import randint
from numpy.fft import fft2, ifft2, fftshift
from matplotlib import pyplot as plt

from utils import matrix as mt
from utils import projector

#Base class
class Ptycho():
    def __init__(self, object, probe, scan_positions, noise):
        self.obj = object
        self.prb = probe
        self.obj_len = len(object) # obj is (obj_len * pbj_len) 2d-array
        self.prb_len = len(probe) # prb is (prb_len * prb_len) 2d-array
        self.scan_pos = scan_positions
        self.noise = noise 
        self.diffs = self.generate_diffraction()
        self.scan_img, self.sampling_ratio = self.illuminate()
    
    #forward model of ptychography
    def generate_diffraction(self):
        diffs = []
        for pos in self.scan_pos:
            obj_pos = self.obj[pos[0] - self.prb_len//2 : pos[0] + self.prb_len//2, pos[1] - self.prb_len//2 : pos[1] + self.prb_len//2]
            diff_pos = np.abs(fft2(obj_pos * self.prb)/self.prb_len) + np.random.normal(loc = 0, scale = self.noise, size = (self.prb_len, self.prb_len))
            diffs.append(diff_pos)
        return diffs
    
    #multiply object with illuminated areas
    def illuminate(self):
        prb_abs = np.abs(self.prb)
        scan_img = np.zeros((self.obj_len,self.obj_len), dtype = float)
        for pos in self.scan_pos:
            scan_img[pos[0] - self.prb_len//2 : pos[0] + self.prb_len//2, pos[1] - self.prb_len//2 : pos[1] + self.prb_len//2] += prb_abs**2
        sampling_number = np.sum(scan_img > 0.1 * np.max(scan_img))
        alpha = (self.prb_len**2 * len(self.scan_pos))/sampling_number
        return scan_img, alpha
    
    #show data
    def show(self):
        L = 4 # show L diffraction patterns
        fig, ax = plt.subplots(1,L+1, figsize=(3*(L+1), 3))
        ax[0].axis("off")
        ax[0].imshow(self.scan_img, cmap = "gray")
        ax[0].set_title("Scanned positions", fontsize = 15)
        for l in range(1,L+1):
            ax[l].axis("off")
            ax[l].imshow(np.log10(fftshift(self.diffs[l-1])), cmap = "jet")
            ax[l].set_title("Diffraction pattern "+str(l), fontsize = 15)
        plt.show()
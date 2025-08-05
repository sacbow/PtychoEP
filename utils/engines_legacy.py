#import modules
import numpy as np
from numpy.random import randint
from numpy import exp, pi
from numpy import linalg as LA
from numpy.fft import fft2, ifft2, fftshift
import copy
#import files
from utils import projector


#PIE
def PIE(ptycho_data, config):
    #"ptycho" : Ptycho object, "config" : config_ePIE object
    #load data
    prb_len = ptycho_data.prb_len
    diffs = ptycho_data.diffs
    scan_pos = ptycho_data.scan_pos
    alpha = config.alpha
    #Initialization
    error_list = []
    obj_PIE = config.obj_init
    prb_PIE = config.prb
    update_schedule = np.arange(len(ptycho_data.scan_pos))
    #ePIE iteration
    prb_max = np.max(np.abs(prb_PIE))
    prb_abs = np.abs(prb_PIE)
    for iter in range(config.num_iter):
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            error += np.mean((diffs[j] - np.abs(fft2(prb_PIE * obj_PIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])/prb_len))**2)
        error_list.append(error/len(scan_pos))
        for j in update_schedule:
            pos = scan_pos[j]
            diff = diffs[j]
            obj_pos = obj_PIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            #Fourier update
            ew_old = prb_PIE * obj_pos
            ew_proj = projector.Fourier_projector(ew_old, diff)
            #object update
            obj_PIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += prb_abs * prb_PIE.conj() * (ew_proj - ew_old) / (prb_max * ( prb_abs**2 + alpha * prb_max**2))       
    return obj_PIE, error_list

#ePIE
def ePIE(ptycho_data, config):
    #"ptycho" : Ptycho object, "config" : config_ePIE object
    #load data
    prb_len = ptycho_data.prb_len
    diffs = ptycho_data.diffs
    scan_pos = ptycho_data.scan_pos
    alpha = config.alpha
    beta = config.beta
    #Initialization
    error_list = []
    obj_ePIE = config.obj_init
    prb_ePIE = config.prb_init
    update_schedule = np.arange(len(ptycho_data.scan_pos))
    #ePIE iteration
    for iter in range(config.num_iter):
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            error += np.mean((diffs[j] - np.abs(fft2(prb_ePIE * obj_ePIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])/prb_len))**2)
        error_list.append(error/len(scan_pos))
        for j in update_schedule:
            pos = scan_pos[j]
            diff = diffs[j]
            obj_prev = copy.copy(obj_ePIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])
            prb_prev = copy.copy(prb_ePIE)
            obj_max = np.max(np.abs(obj_prev))
            prb_max = np.max(np.abs(prb_prev))
            #Fourier update
            ew_old = prb_ePIE * obj_prev
            ew_proj = projector.Fourier_projector(ew_old, diff)
            #object update
            obj_ePIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += alpha * prb_prev.conj() * (ew_proj - ew_old) / (prb_max**2)
            #probe update
            prb_ePIE += beta * obj_prev.conj() * (ew_proj - ew_old) / (obj_max**2)
    return obj_ePIE, prb_ePIE, error_list

def rPIE(ptycho_data, config):
    #"ptycho" : Ptycho object, "config" : config_rPIE object
    #load data
    prb_len = ptycho_data.prb_len
    diffs = ptycho_data.diffs
    scan_pos = ptycho_data.scan_pos
    alpha = config.alpha
    beta = config.beta
    #Initialization
    error_list = []
    obj_rPIE = config.obj_init
    prb_rPIE = config.prb_init
    update_schedule = np.arange(len(ptycho_data.scan_pos))
    #rPIE iteration
    for iter in range(config.num_iter):
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            error += np.mean((diffs[j] - np.abs(fft2(prb_rPIE * obj_rPIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])/prb_len))**2)
        error_list.append(error/len(scan_pos))
        for j in update_schedule:
            pos = scan_pos[j]
            diff = diffs[j]
            obj_prev = copy.copy(obj_rPIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])
            prb_prev = copy.copy(prb_rPIE)
            obj_max = np.max(np.abs(obj_prev))
            prb_max = np.max(np.abs(prb_prev))
            #Fourier update
            ew_old = prb_rPIE * obj_prev
            ew_proj = projector.Fourier_projector(ew_old, diff)
            #object update
            obj_rPIE[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += prb_prev.conj() * (ew_proj - ew_old) / (alpha * prb_max**2 + (1-alpha) * np.abs(prb_prev)**2)
            #probe update
            prb_rPIE += beta * obj_prev.conj() * (ew_proj - ew_old) / (obj_max**2)
    return obj_rPIE, prb_rPIE, error_list






#Difference Map
def DifferenceMap(ptycho_data, config):
    #"ptycho" : Ptycho object, "config" : config_DM object
    #load data
    prb_len = ptycho_data.prb_len
    diffs = ptycho_data.diffs
    scan_pos = ptycho_data.scan_pos
    #load config
    obj_DM = config.obj_init
    prb_DM = config.prb_init
    #Initialize exit_wave and Phi
    exit_wave_list, Phi_list = [], []
    for j, pos in enumerate(scan_pos):
        exit_wave = obj_DM[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] * prb_DM
        exit_wave_list.append(exit_wave)
        Phi = projector.Fourier_projector(exit_wave, diffs[j])
        Phi_list.append(Phi)
    #iteration
    error_list = []
    for iter in range(config.num_iter):
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            error += np.mean((diffs[j] - np.abs(fft2(exit_wave_list[j])/prb_len))**2)
        error_list.append(error/len(scan_pos))
        #object update
        den_obj = np.zeros_like(ptycho_data.obj) + 1e-10
        num_obj = np.zeros_like(ptycho_data.obj)
        for j, pos in enumerate(scan_pos):
            den_obj[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += np.abs(prb_DM)**2
            num_obj[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += prb_DM.conj() * Phi_list[j]
        obj_DM = num_obj / den_obj
        #probe update
        den_prb = np.zeros_like(ptycho_data.prb) + 1e-10
        num_prb = np.zeros_like(ptycho_data.prb)
        for j, pos in enumerate(scan_pos):
            den_prb += np.abs(obj_DM[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])**2
            num_prb += obj_DM[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2].conj() * Phi_list[j]
        prb_DM = num_prb / den_prb
        #exit_wave update
        for j, pos in enumerate(scan_pos):
            exit_wave_list[j] = prb_DM * obj_DM[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
        #difference map
        for j in range(len(scan_pos)):
            Phi_list[j] += projector.Fourier_projector(2 * exit_wave_list[j] - Phi_list[j], diffs[j]) - exit_wave_list[j]
    return obj_DM, prb_DM, error_list
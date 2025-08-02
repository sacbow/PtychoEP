#imports
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

#Linear Least-Mean-Square estimator

def parallel_llmse(r_2, gamma_2, exit_wave_list, tau_2_list, ptycho_data, probe):
    #load ptycho_data
    scan_pos = ptycho_data.scan_pos
    probe_int = np.abs(probe)**2
    prb_len = ptycho_data.prb_len
    #calculate x_2 and eta_2
    x_2_num, eta_2 = gamma_2 * r_2, np.zeros_like(gamma_2)
    eta_2 += gamma_2
    for j, pos in enumerate(scan_pos):
        eta_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += tau_2_list[j] * probe_int
        x_2_num[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += tau_2_list[j] * probe.conj() * exit_wave_list[j]
    x_2 = x_2_num / eta_2
    #EM update of probe
    probe_num, probe_den = np.zeros_like(probe), np.zeros((prb_len, prb_len), dtype = float)
    for j, pos in enumerate(scan_pos):
        probe_num += tau_2_list[j] * x_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2].conj() * exit_wave_list[j]
        probe_den += tau_2_list[j] * (np.abs(x_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])**2 +\
                                       eta_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]**(-1))
    probe_EM = probe_num / probe_den
    probe_int = np.abs(probe_EM)**2
    #update z_2_list and lam_2_list
    z_2_list, lam_2_list = [], []
    for j, pos in enumerate(scan_pos):
        lam_2_list.append(1/np.mean(probe_int/eta_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]))
        z_2_list.append(fft2(probe_EM * x_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])/prb_len)
    return x_2, eta_2, z_2_list, lam_2_list, probe_EM

def llmse(r_2, gamma_2, exit_wave_list, tau_2_list, ptycho_data, probe):
    #load ptycho_data
    scan_pos = ptycho_data.scan_pos
    probe_int = np.abs(probe)**2
    prb_len = ptycho_data.prb_len
    #calculate x_2 and eta_2
    x_2_num, eta_2 = gamma_2 * r_2, np.zeros_like(gamma_2)
    eta_2 += gamma_2
    for j, pos in enumerate(scan_pos):
        eta_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += tau_2_list[j] * probe_int
        x_2_num[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += tau_2_list[j] * probe.conj() * exit_wave_list[j]
    x_2 = x_2_num / eta_2
    #EM update of probe
    probe_num, probe_den = np.zeros_like(probe), np.zeros((prb_len, prb_len), dtype = float)
    for j, pos in enumerate(scan_pos):
        probe_num += tau_2_list[j] * x_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2].conj() * exit_wave_list[j]
        probe_den += tau_2_list[j] * (np.abs(x_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])**2 +\
                                       eta_2[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]**(-1))
    probe_EM = probe_num / probe_den
    probe_int = np.abs(probe_EM)**2
    return x_2, eta_2, probe_EM


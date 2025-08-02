#imports
import numpy as np
from numpy.fft import fft2, ifft2
from utils import message_passing as mp
from utils import denoiser

# known probe, uninformative prior CN(0,1)
def PtychoEP_Gaussian(ptycho_data, config):
    #load ptycho_data
    obj_len = ptycho_data.obj_len
    prb_len = ptycho_data.prb_len
    scan_pos = ptycho_data.scan_pos
    gamma_w = (1/ptycho_data.noise)**2
    diffs = ptycho_data.diffs
    #initialize some variables
    error_list = []
    O_int = config.object_init
    gamma_int_O = np.ones((obj_len, obj_len))
    P = config.probe_init
    P_intensity = np.abs(P)**2
    gamma_int_Phi = np.ones(len(scan_pos))
    gamma_int_O_hat = np.ones((obj_len, obj_len))
    beta_int_hat = np.zeros_like(O_int)
    Phi_int, Phi_int_ift = [], []
    for j, pos in enumerate(scan_pos):
        O_j = O_int[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
        exit_wave_j = P * O_j
        Phi_int_ift.append(exit_wave_j)
        Phi_int.append(fft2(exit_wave_j)/prb_len)
        gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += P_intensity
        beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += P_intensity * O_j
    #main iteration
    for iter in range(config.num_iter):
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            error += np.mean((diffs[j] - np.abs(fft2(O_j * P)/prb_len))**2)
        error_list.append(error/len(scan_pos))
        for j, pos in enumerate(scan_pos):
            #LLMSE
            O_int_hat_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            Phi_int_hat_j = fft2(P * O_int_hat_j)/prb_len
            gamma_int_hat_j = 1/np.mean(P_intensity/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])
            #Message Passing
            Phi_ext_j, gamma_ext_Phi_j = mp.Message_Passing(Phi_int[j], gamma_int_Phi[j], Phi_int_hat_j, gamma_int_hat_j)
            #denoising by diffraction pattern
            Phi_ext_j_hat, gamma_ext_Phi_j_hat = denoiser.PR_output_denoiser(Phi_ext_j, gamma_ext_Phi_j, diffs[j], gamma_w)
            #Message Passing with damping
            Phi_int_j_raw, gamma_int_Phi_raw = mp.Message_Passing(Phi_ext_j, gamma_ext_Phi_j, Phi_ext_j_hat, gamma_ext_Phi_j_hat)
            Phi_int_j_new, gamma_int_Phi_new = mp.Damping(Phi_int_j_raw, gamma_int_Phi_raw, Phi_int[j], gamma_int_Phi[j], config.damping)
            Phi_int_j_new_ift = ifft2(Phi_int_j_new)*prb_len
            #update beta_int_hat, gamma_o_int_hat, Phi_int, Phi_int_ift, gamma_int_Phi
            gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += (gamma_int_Phi_new - gamma_int_Phi[j]) * P_intensity
            beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += \
                  P.conj() * (gamma_int_Phi_new * Phi_int_j_new_ift - gamma_int_Phi[j] * Phi_int_ift[j])
            gamma_int_Phi[j] = gamma_int_Phi_new
            Phi_int[j] = Phi_int_j_new
            Phi_int_ift[j] = Phi_int_j_new_ift
        #denoising by prior
        O_int_hat = beta_int_hat/gamma_int_O_hat
        O_ext, gamma_ext_O = mp.Message_Passing(O_int, gamma_int_O, O_int_hat, gamma_int_O_hat)
        O_ext_hat, gamma_ext_O_hat = denoiser.gaussian_complex_input_denoiser(O_ext, gamma_ext_O)
        O_int_new, gamma_int_O_new = mp.Message_Passing(O_ext, gamma_ext_O, O_ext_hat, gamma_ext_O_hat)
        #update beta_int_hat, gamma_int_O_hat
        gamma_int_O_hat += gamma_int_O_new - gamma_int_O
        beta_int_hat += gamma_int_O_new * O_int_new - gamma_int_O * O_int
        gamma_int_O, O_int = gamma_int_O_new, O_int_new
    return O_int_hat, gamma_int_O_hat, error_list


#known probe, sparse prior
def PtychoEP_Sparse(ptycho_data, config):
    #load ptycho_data
    obj_len = ptycho_data.obj_len
    prb_len = ptycho_data.prb_len
    scan_pos = ptycho_data.scan_pos
    gamma_w = (1/ptycho_data.noise)**2
    diffs = ptycho_data.diffs
    #initialize some variables
    error_list = []
    O_int = config.object_init
    gamma_int_O = np.ones((obj_len, obj_len))
    P = config.probe_init
    P_intensity = np.abs(P)**2
    gamma_int_Phi = np.ones(len(scan_pos))
    gamma_int_O_hat = np.ones((obj_len, obj_len))
    beta_int_hat = np.zeros_like(O_int)
    Phi_int, Phi_int_ift = [], []
    for j, pos in enumerate(scan_pos):
        O_j = O_int[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
        exit_wave_j = P * O_j
        Phi_int_ift.append(exit_wave_j)
        Phi_int.append(fft2(exit_wave_j)/prb_len)
        gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += P_intensity
        beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += P_intensity * O_j
    #main iteration
    for iter in range(config.num_iter):
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            error += np.mean((diffs[j] - np.abs(fft2(O_j * P)/prb_len))**2)
        error_list.append(error/len(scan_pos))
        for j, pos in enumerate(scan_pos):
            #LLMSE
            O_int_hat_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            Phi_int_hat_j = fft2(P * O_int_hat_j)/prb_len
            gamma_int_hat_j = 1/np.mean(P_intensity/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])
            #Message Passing
            Phi_ext_j, gamma_ext_Phi_j = mp.Message_Passing(Phi_int[j], gamma_int_Phi[j], Phi_int_hat_j, gamma_int_hat_j)
            #denoising by diffraction pattern
            Phi_ext_j_hat, gamma_ext_Phi_j_hat = denoiser.PR_output_denoiser(Phi_ext_j, gamma_ext_Phi_j, diffs[j], gamma_w)
            #Message Passing with damping
            Phi_int_j_raw, gamma_int_Phi_raw = mp.Message_Passing(Phi_ext_j, gamma_ext_Phi_j, Phi_ext_j_hat, gamma_ext_Phi_j_hat)
            Phi_int_j_new, gamma_int_Phi_new = mp.Damping(Phi_int_j_raw, gamma_int_Phi_raw, Phi_int[j], gamma_int_Phi[j], config.damping)
            Phi_int_j_new_ift = ifft2(Phi_int_j_new)*prb_len
            #update beta_int_hat, gamma_o_int_hat, Phi_int, Phi_int_ift, gamma_int_Phi
            gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += (gamma_int_Phi_new - gamma_int_Phi[j]) * P_intensity
            beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += \
                  P.conj() * (gamma_int_Phi_new * Phi_int_j_new_ift - gamma_int_Phi[j] * Phi_int_ift[j])
            gamma_int_Phi[j] = gamma_int_Phi_new
            Phi_int[j] = Phi_int_j_new
            Phi_int_ift[j] = Phi_int_j_new_ift
        #denoising by prior
        O_int_hat = beta_int_hat/gamma_int_O_hat
        O_ext, gamma_ext_O = mp.Message_Passing(O_int, gamma_int_O, O_int_hat, gamma_int_O_hat)
        O_ext_hat, gamma_ext_O_hat = denoiser.sparse_complex_input_denoiser(O_ext, gamma_ext_O, config.rho)
        O_int_new, gamma_int_O_new = mp.Message_Passing(O_ext, gamma_ext_O, O_ext_hat, gamma_ext_O_hat)
        #update beta_int_hat, gamma_int_O_hat
        gamma_int_O_hat += gamma_int_O_new - gamma_int_O
        beta_int_hat += gamma_int_O_new * O_int_new - gamma_int_O * O_int
        gamma_int_O, O_int = gamma_int_O_new, O_int_new
    return O_int_hat, gamma_int_O_hat, error_list

# unknown probe, uninformative prior CN(0,1)
def PtychoEP_Gaussian_EM(ptycho_data, config):
    #load ptycho_data
    obj_len = ptycho_data.obj_len
    prb_len = ptycho_data.prb_len
    scan_pos = ptycho_data.scan_pos
    gamma_w = (1/ptycho_data.noise)**2
    diffs = ptycho_data.diffs
    #initialize some variables
    error_list = []
    O_int = config.object_init
    gamma_int_O = np.ones((obj_len, obj_len))
    P = config.probe_init
    P_intensity = np.abs(P)**2
    gamma_int_Phi = np.ones(len(scan_pos))
    Phi_int, Phi_int_ift = [], []
    for j, pos in enumerate(scan_pos):
        O_j = O_int[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
        exit_wave_j = P * O_j
        Phi_int_ift.append(exit_wave_j)
        Phi_int.append(fft2(exit_wave_j)/prb_len)
    #main iteration
    for iter in range(config.num_iter):
        #calculate beta_int_hat and gamma_int_O_hat
        gamma_int_O_hat, beta_int_hat = np.zeros((obj_len, obj_len)), np.zeros_like(O_int)
        gamma_int_O_hat += gamma_int_O
        beta_int_hat += gamma_int_O * O_int
        for j, pos in enumerate(scan_pos):
            gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += gamma_int_Phi[j] * P_intensity
            beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += gamma_int_Phi[j] * P.conj() * Phi_int_ift[j]
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            error += np.mean((diffs[j] - np.abs(fft2(O_j * P)/prb_len))**2)
        error_list.append(error/len(scan_pos))
        #exploit diffraction patterns
        for j, pos in enumerate(scan_pos):
            #LLMSE
            O_int_hat_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            Phi_int_hat_j = fft2(P * O_int_hat_j)/prb_len
            gamma_int_hat_j = 1/np.mean(P_intensity/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])
            #Message Passing
            Phi_ext_j, gamma_ext_Phi_j = mp.Message_Passing(Phi_int[j], gamma_int_Phi[j], Phi_int_hat_j, gamma_int_hat_j)
            #denoising by diffraction pattern
            Phi_ext_j_hat, gamma_ext_Phi_j_hat = denoiser.PR_output_denoiser(Phi_ext_j, gamma_ext_Phi_j, diffs[j], gamma_w)
            #Message Passing with damping
            Phi_int_j_raw, gamma_int_Phi_raw = mp.Message_Passing(Phi_ext_j, gamma_ext_Phi_j, Phi_ext_j_hat, gamma_ext_Phi_j_hat)
            Phi_int_j_new, gamma_int_Phi_new = mp.Damping(Phi_int_j_raw, gamma_int_Phi_raw, Phi_int[j], gamma_int_Phi[j], config.damping)
            Phi_int_j_new_ift = ifft2(Phi_int_j_new)*prb_len
            #update beta_int_hat, gamma_o_int_hat, Phi_int, Phi_int_ift, gamma_int_Phi
            gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += (gamma_int_Phi_new - gamma_int_Phi[j]) * P_intensity
            beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += \
                  P.conj() * (gamma_int_Phi_new * Phi_int_j_new_ift - gamma_int_Phi[j] * Phi_int_ift[j])
            gamma_int_Phi[j] = gamma_int_Phi_new
            Phi_int[j] = Phi_int_j_new
            Phi_int_ift[j] = Phi_int_j_new_ift
        #denoising by prior
        O_int_hat = beta_int_hat/gamma_int_O_hat
        O_ext, gamma_ext_O = mp.Message_Passing(O_int, gamma_int_O, O_int_hat, gamma_int_O_hat)
        O_ext_hat, gamma_ext_O_hat = denoiser.gaussian_complex_input_denoiser(O_ext, gamma_ext_O)
        O_int_new, gamma_int_O_new = mp.Message_Passing(O_ext, gamma_ext_O, O_ext_hat, gamma_ext_O_hat)
        #update beta_int_hat, gamma_int_O_hat
        gamma_int_O_hat += gamma_int_O_new - gamma_int_O
        beta_int_hat += gamma_int_O_new * O_int_new - gamma_int_O * O_int
        gamma_int_O, O_int = gamma_int_O_new, O_int_new
        #Adaptive EM update of probe
        for t in range(config.num_prb):
            #probe update
            P_1, P_2 = np.zeros_like(P), np.zeros((prb_len, prb_len), dtype = float)
            for j, pos in enumerate(scan_pos):
                O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                gamma_O_j = gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                P_1 += gamma_int_Phi[j] * O_j.conj() * Phi_int_ift[j]
                P_2 += gamma_int_Phi[j] * (np.abs(O_j)**2 + 1/gamma_O_j)
            P = P_1/P_2
            P_intensity = np.abs(P)**2
            #adaptive update of gamma_int_Phi
            for j, pos in enumerate(scan_pos):
                O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                gamma_j = gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                gamma_int_Phi[j] = 1/np.mean(np.abs(Phi_int_ift[j] - P * O_j)**2 + P_intensity/gamma_j)
    return O_int_hat, gamma_int_O_hat, P, error_list

# unknown probe, sprase prior
def PtychoEP_Sparse_EM(ptycho_data, config):
    #load ptycho_data
    obj_len = ptycho_data.obj_len
    prb_len = ptycho_data.prb_len
    scan_pos = ptycho_data.scan_pos
    gamma_w = (1/ptycho_data.noise)**2
    diffs = ptycho_data.diffs
    #initialize some variables
    error_list = []
    O_int = config.object_init
    gamma_int_O = np.ones((obj_len, obj_len))
    P = config.probe_init
    P_intensity = np.abs(P)**2
    gamma_int_Phi = np.ones(len(scan_pos))
    Phi_int, Phi_int_ift = [], []
    for j, pos in enumerate(scan_pos):
        O_j = O_int[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
        exit_wave_j = P * O_j
        Phi_int_ift.append(exit_wave_j)
        Phi_int.append(fft2(exit_wave_j)/prb_len)
    #main iteration
    for iter in range(config.num_iter):
        #calculate beta_int_hat and gamma_int_O_hat
        gamma_int_O_hat, beta_int_hat = np.zeros((obj_len, obj_len)), np.zeros_like(O_int)
        gamma_int_O_hat += gamma_int_O
        beta_int_hat += gamma_int_O * O_int
        for j, pos in enumerate(scan_pos):
            gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += gamma_int_Phi[j] * P_intensity
            beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += gamma_int_Phi[j] * P.conj() * Phi_int_ift[j]
        #measure error
        error = 0
        for j, pos in enumerate(scan_pos):
            O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            error += np.mean((diffs[j] - np.abs(fft2(O_j * P)/prb_len))**2)
        error_list.append(error/len(scan_pos))
        #exploit diffraction patterns
        for j, pos in enumerate(scan_pos):
            #LLMSE
            O_int_hat_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
            Phi_int_hat_j = fft2(P * O_int_hat_j)/prb_len
            gamma_int_hat_j = 1/np.mean(P_intensity/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2])
            #Message Passing
            Phi_ext_j, gamma_ext_Phi_j = mp.Message_Passing(Phi_int[j], gamma_int_Phi[j], Phi_int_hat_j, gamma_int_hat_j)
            #denoising by diffraction pattern
            Phi_ext_j_hat, gamma_ext_Phi_j_hat = denoiser.PR_output_denoiser(Phi_ext_j, gamma_ext_Phi_j, diffs[j], gamma_w)
            #Message Passing with damping
            Phi_int_j_raw, gamma_int_Phi_raw = mp.Message_Passing(Phi_ext_j, gamma_ext_Phi_j, Phi_ext_j_hat, gamma_ext_Phi_j_hat)
            Phi_int_j_new, gamma_int_Phi_new = mp.Damping(Phi_int_j_raw, gamma_int_Phi_raw, Phi_int[j], gamma_int_Phi[j], config.damping)
            Phi_int_j_new_ift = ifft2(Phi_int_j_new)*prb_len
            #update beta_int_hat, gamma_o_int_hat, Phi_int, Phi_int_ift, gamma_int_Phi
            gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += (gamma_int_Phi_new - gamma_int_Phi[j]) * P_intensity
            beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2] += \
                  P.conj() * (gamma_int_Phi_new * Phi_int_j_new_ift - gamma_int_Phi[j] * Phi_int_ift[j])
            gamma_int_Phi[j] = gamma_int_Phi_new
            Phi_int[j] = Phi_int_j_new
            Phi_int_ift[j] = Phi_int_j_new_ift
        #denoising by prior
        O_int_hat = beta_int_hat/gamma_int_O_hat
        O_ext, gamma_ext_O = mp.Message_Passing(O_int, gamma_int_O, O_int_hat, gamma_int_O_hat)
        O_ext_hat, gamma_ext_O_hat = denoiser.sparse_complex_input_denoiser(O_ext, gamma_ext_O, config.rho)
        O_int_new, gamma_int_O_new = mp.Message_Passing(O_ext, gamma_ext_O, O_ext_hat, gamma_ext_O_hat)
        #update beta_int_hat, gamma_int_O_hat
        gamma_int_O_hat += gamma_int_O_new - gamma_int_O
        beta_int_hat += gamma_int_O_new * O_int_new - gamma_int_O * O_int
        gamma_int_O, O_int = gamma_int_O_new, O_int_new
        #Adaptive EM update of probe
        for t in range(config.num_prb):
            #probe update
            P_1, P_2 = np.zeros_like(P), np.zeros((prb_len, prb_len), dtype = float)
            for j, pos in enumerate(scan_pos):
                O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                gamma_O_j = gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                P_1 += gamma_int_Phi[j] * O_j.conj() * Phi_int_ift[j]
                P_2 += gamma_int_Phi[j] * (np.abs(O_j)**2 + 1/gamma_O_j)
            P = P_1/P_2
            P_intensity = np.abs(P)**2
            #adaptive update of gamma_int_Phi
            for j, pos in enumerate(scan_pos):
                O_j = beta_int_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]/gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                gamma_j = gamma_int_O_hat[pos[0] - prb_len//2 : pos[0] + prb_len//2, pos[1] - prb_len//2 : pos[1] + prb_len//2]
                gamma_int_Phi[j] = 1/np.mean(np.abs(Phi_int_ift[j] - P * O_j)**2 + P_intensity/gamma_j)
    return O_int_hat, gamma_int_O_hat, P, error_list
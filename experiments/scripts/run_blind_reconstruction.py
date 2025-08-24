import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import matplotlib.pyplot as plt
from ptychoep.backend.backend import set_backend, np
set_backend("numpy")

from ptychoep.ptycho.core import Ptycho
from ptychoep.utils.io_utils import load_data_image
from ptychoep.ptycho.aperture_utils import circular_aperture
from ptychoep.ptycho.scan_utils import generate_centered_grid_positions
from ptychoep.ptycho.noise import GaussianNoise, PoissonNoise
from ptychoep.rng.rng_utils import get_rng
from ptychoep.classic_engines.pie import PIE
from ptychoep.classic_engines.epie import ePIE
from ptychoep.classic_engines.rpie import rPIE
from ptychoep.classic_engines.difference_map import DifferenceMap
from ptychoep.ptychoep.core import PtychoEP


def generate_data(image_size, probe_size, step, num_scans, jitter, noise_type, noise_param, seed):
    amp = np().asarray(load_data_image("cameraman.png"), dtype=np().complex64)
    phase = np().asarray(load_data_image("eagle.png"), dtype=np().complex64)
    obj = amp * np().exp(1j * np().pi / 2 * phase)

    probe_gt = np().array(load_data_image("probe.png"), dtype=np().complex64)
    probe_init = np().array(load_data_image("probe_init.png"), dtype=np().complex64)

    positions = generate_centered_grid_positions(
        image_size=image_size,
        probe_size=probe_size,
        step=step,
        num_points_y=num_scans,
        num_points_x=num_scans,
        jitter=jitter,
        seed=90
    )

    ptycho = Ptycho()
    ptycho.set_object(obj)
    ptycho.set_probe(probe_gt)
    ptycho.forward_and_set_diffraction(positions)
    ptycho.sort_diffraction_data()

    if noise_type == 'gaussian':
        GaussianNoise(var=noise_param, seed=seed) @ ptycho
    elif noise_type == 'poisson':
        PoissonNoise(scale=noise_param, seed=seed) @ ptycho
        max_count = [np().max(diff.diffraction) for diff in ptycho._diff_data]
        photons = np().max(np().array(max_count)) ** 2 * noise_param - 3. / 8.
        print(f"maximum number of photons :  {int(round(photons))}")
    else:
        raise ValueError("Unsupported noise type")

    return ptycho, probe_init


def run_all_algorithms(ptycho, probe_init, seed):
    results = {}

    for EngineClass, name, kwargs in [
        (PtychoEP, "Ptycho-EP(proposed)", dict(damping=0.5, n_probe_update=2)),
        (ePIE, "ePIE", dict(alpha=1.0, beta=1.0)),
        (rPIE, "rPIE", dict(alpha=1.0, beta=1.0)),
        (DifferenceMap, "DM", dict())
    ]:
        errors = []
        engine = EngineClass(
            ptycho=ptycho,
            prb_init=probe_init,
            seed=seed,
            callback=lambda it, err, obj: errors.append(err),
            **kwargs
        )
        result = engine.run(n_iter=100 if name != "ePIE" and name != "rPIE" else 200)
        results[name] = (np().asarray(errors), result[0])

    return results


def plot_convergence(all_errors_dict, ptycho, outpath):
    diff_power = np().mean([np().mean(np().abs(d.diffraction)**2) for d in ptycho._diff_data])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.tick_params(direction="in", length=3, width=2, labelsize=20)
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(2)

    color_map = {
        "ePIE": ("blue", "+", "navy"),
        "DM": ("orange", "o", "red"),
        "Ptycho-EP(proposed)": ("purple", "s", "magenta"),
        "rPIE": ("green", "^", "darkgreen")
    }

    for name, errors_list in all_errors_dict.items():
        errors_array = np().array(errors_list) / diff_power  # shape: (n_repeats, n_iters)
        median_error = np().median(errors_array, axis=0)
        color, marker, edge = color_map.get(name, ("black", "x", "black"))
        ax.plot(median_error, label=name, color=color, marker=marker,
                markeredgecolor=edge, markerfacecolor="white", markersize=5)

    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("NMSE", fontsize=20)
    ax.set_ylim(1e-3, 1)
    ax.set_yscale('log')
    ax.legend(fontsize=14)
    ax.grid(which='major', color='black', linestyle='-', alpha=0.3)
    ax.grid(which='minor', color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def crop_center(img, crop_size):
    H, W = img.shape
    ch = crop_size // 2
    center_y, center_x = H // 2, W // 2
    return img[center_y - ch:center_y + ch, center_x - ch:center_x + ch]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_repeats', type=int, default=10)
    parser.add_argument('--noise_type', type=str, choices=['gaussian', 'poisson'], default='gaussian')
    parser.add_argument('--noise_param', type=float, default=1e-4)  # var or scale
    parser.add_argument('--step', type=int, default=18)
    parser.add_argument('--num_scans', type=int, default=11)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(script_dir), "result")
    os.makedirs(out_dir, exist_ok=True)

    all_errors = {"Ptycho-EP(proposed)": [], "ePIE": [], "rPIE": [], "DM": []}

    for i in range(args.n_repeats):
        print(f"Running seed {i}...")
        ptycho, probe_init = generate_data(
            image_size=512,
            probe_size=128,
            step=args.step,
            num_scans=args.num_scans,
            jitter=2,
            noise_type=args.noise_type,
            noise_param=args.noise_param,
            seed=9 + i
        )
        results = run_all_algorithms(ptycho, probe_init, seed=i)

        for name, (errors, obj) in results.items():
            all_errors[name].append(errors)
            if i == 0:
                cropped = crop_center(np().abs(obj), 256)
                plt.imsave(
                    os.path.join(out_dir, f"recon_{name}_seed_{i}_step{args.step}.png"),
                    cropped, cmap="gray"
                )



    plot_convergence(
        all_errors,
        ptycho,
        outpath=os.path.join(out_dir, f"convergence_median_step{args.step}.png")
    )

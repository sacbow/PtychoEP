import argparse
from functools import partial

from PtychoEP.backend.backend import set_backend, np
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.ptycho.noise import GaussianNoise
from PtychoEP.ptychoep.core import PtychoEP

from .profile_utils import time_execution, profile_execution


def main():
    parser = argparse.ArgumentParser(description="PtychoEP profiling script")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy"],
                        help="Backend to use (numpy or cupy)")
    parser.add_argument("--niter", type=int, default=20, help="Number of EP iterations")
    parser.add_argument("--num_points", type=int, default=200, help="Number of scan positions")
    parser.add_argument("--use_noise", action="store_true", help="Add Gaussian noise to diffraction data")
    parser.add_argument("--use_prior", action="store_true", help="Use sparse prior (default: Gaussian)")
    parser.add_argument("--n_probe_update", type=int, default=0, help="Number of EM probe updates per iteration")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile profiling")
    parser.add_argument("--profile_sort", type=str, default="cumulative",
                        choices=["time", "cumulative", "calls"], help="Sort key for cProfile results")
    parser.add_argument("--profile_limit", type=int, default=30, help="Number of profile lines to display")
    parser.add_argument("--profile_output", type=str, default=None, help="File to save cProfile results")
    args = parser.parse_args()

    # --- Set backend ---
    set_backend(args.backend)
    print(f"[INFO] Backend: {args.backend}")

    # --- Setup Ptycho object ---
    ptycho = Ptycho()
    obj = np().array(load_data_image("cameraman.png")) * np().exp(1j * np().pi * np().array(load_data_image("eagle.png")) ,dtype = np().complex64)
    probe = np().array(load_data_image("probe.png"), dtype = np().complex64)
    ptycho.set_object(obj)
    ptycho.set_probe(probe)

    # --- Scan positions and forward model ---
    positions = generate_spiral_scan_positions(image_size=512, probe_size=128, num_points=args.num_points)
    ptycho.forward_and_set_diffraction(positions)

    # --- Add noise if specified ---
    if args.use_noise:
        GaussianNoise(var=1e-3) @ ptycho
        print("[INFO] Gaussian noise added (var=1e-3)")

    # --- Create PtychoEP instance ---
    ep = PtychoEP(
        ptycho,
        damping=0.7,
        prior_name="sparse" if args.use_prior else "gaussian",
        n_probe_update=args.n_probe_update
    )
    run_fn = partial(ep.run, n_iter=args.niter)

    # --- Run profiling or timing ---
    if args.profile:
        print(f"[INFO] Running PtychoEP with cProfile (n_iter={args.niter})...")
        profile_execution(run_fn, sort_key=args.profile_sort,
                          limit=args.profile_limit, output_file=args.profile_output)
    else:
        print(f"[INFO] Running PtychoEP timing (n_iter={args.niter})...")
        elapsed = time_execution(run_fn, backend=args.backend)
        print(f"[RESULT] PtychoEP execution time ({args.backend}): {elapsed:.3f} sec")


if __name__ == "__main__":
    main()

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
from ptychoep.ptycho.scan_utils import generate_spiral_scan_positions
from ptychoep.ptycho.noise import GaussianNoise
from ptychoep.ptycho.visualize import compute_illumination
from ptychoep.rng.rng_utils import get_rng
from ptychoep.classic_engines.pie import PIE
from ptychoep.ptychoep.core import PtychoEP

# directory to save results
output_dir = os.path.join(os.path.dirname(__file__), '../result')
os.makedirs(output_dir, exist_ok=True)

def phase_align(x_est, x_true):
    inner_product = np().vdot(x_true, x_est)
    phase = np().angle(inner_product)
    return x_est * np().exp(-1j * phase)

def error(x_est, x_true):
    return np().sum(np().abs(x_est - x_true) ** 2)

def pmse(x_est, x_true, crop=None):
    if crop is not None:
        y0, y1, x0, x1 = crop
        x_est = x_est[y0:y1, x0:x1]
        x_true = x_true[y0:y1, x0:x1]
    error_power = error(phase_align(x_est, x_true), x_true)
    signal_power = np().sum(np().abs(x_true)**2)
    return 10 * np().log10(signal_power / error_power)

def crop_center(img, crop_size):
    H, W = img.shape
    ch = crop_size // 2
    center_y, center_x = H // 2, W // 2
    return img[center_y - ch:center_y + ch, center_x - ch:center_x + ch]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=16.93, help="Scan step size")
    parser.add_argument("--noise", type=float, default=3.4, help="noise variance(1e-5)")
    parser.add_argument("--trials", type=int, default=10, help="Number of repeated trials")
    parser.add_argument("--object", type=str, choices=["lily", "cameraman"], default="lily",
                    help="Object image set to use (lily/moon or cameraman/eagle)")
    parser.add_argument("--prior", type=str, choices=["gaussian", "sparse"], default="gaussian",
                    help="Prior type for Ptycho-EP (gaussian or sparse)")

    args = parser.parse_args()

    # 固定条件
    if args.object == "lily":
        amp = load_data_image("lily.png")
        phs = load_data_image("moon.png")
    elif args.object == "cameraman":
        amp = load_data_image("cameraman.png")
        phs = load_data_image("eagle.png")
    else:
        raise ValueError(f"Unknown object name: {args.object}")

    obj_true = amp * np().exp(1j * np().pi / 2 * phs)
    probe = circular_aperture(size=64, r=0.47)
    center_crop = (128, 384, 128, 384)
    noise_var = args.noise * 1e-5
    n_iter = 400

    # 結果格納
    pmse_list_pie = []
    pmse_list_ep = []
    all_pie_errors = []
    all_ep_errors = []

    for trial in range(args.trials):
        print(f"\n--- Trial {trial+1}/{args.trials} ---")
        data_seed = trial
        init_seed = trial + 1000
        rng_data = get_rng(data_seed)
        rng_init = get_rng(init_seed)

        # 初期オブジェクト
        obj_init = rng_init.normal(size=obj_true.shape) + 1j * rng_init.normal(size=obj_true.shape)
        obj_init = obj_init.astype(np().complex64)

        # Ptychoデータ生成
        ptycho = Ptycho()
        ptycho.set_object(obj_true)
        ptycho.set_probe(probe)
        positions = generate_spiral_scan_positions(
            image_size=512,
            probe_size=64,
            step=args.step,
            num_points=1000
        )
        ptycho.forward_and_set_diffraction(positions)
        GaussianNoise(var=noise_var, seed=data_seed) @ ptycho

        # 観測率・ノイズ情報出力
        _, alpha = compute_illumination(ptycho)
        print(f"[INFO] Sampling ratio (alpha): {alpha:.2f}")
        print(f"[INFO] J = {len(ptycho._diff_data)}")
        print(f"[INFO] SNR: {ptycho.noise_stats['snr_mean_db']:.2f} dB")

        # PIE 実行
        pie_errors = []
        pie_solver = PIE(
            ptycho=ptycho,
            alpha=0.1,
            obj_init=obj_init.copy(),
            callback=lambda i, err, est: pie_errors.append(err) if i % 10 == 0 else None
        )
        obj_est_pie = pie_solver.run(n_iter=n_iter)
        all_pie_errors.append(pie_errors)
        pmse_pie = pmse(obj_est_pie, obj_true, crop=center_crop)
        pmse_list_pie.append(pmse_pie)

        # Ptycho-EP 実行
        ep_errors = []
        ep_solver = PtychoEP(
            ptycho=ptycho,
            prb_init=probe.copy(),
            obj_init=obj_init.copy(),
            damping=0.9,
            callback=lambda i, err, est: ep_errors.append(err) if i % 10 == 0 else None, 
            prior_name=args.prior,
            n_probe_update=0
        )

        obj_est_ep, _ = ep_solver.run(n_iter=n_iter)
        all_ep_errors.append(ep_errors)
        pmse_ep = pmse(obj_est_ep, obj_true, crop=center_crop)
        pmse_list_ep.append(pmse_ep)

        print(f"[RESULT] PIE       PMSE: {pmse_pie:.4f} dB")
        print(f"[RESULT] Ptycho-EP PMSE: {pmse_ep:.4f} dB")

        # 画像保存（初回だけ）
        if trial == 0:
            out_img_pie = crop_center(np().abs(obj_est_pie), 128)
            out_img_ep  = crop_center(np().abs(obj_est_ep), 128)
            plt.imsave(os.path.join(output_dir, f"recon_pie_object_{args.object}_step_{args.step}.png"), out_img_pie, cmap="gray")
            plt.imsave(os.path.join(output_dir, f"recon_ep_object_{args.object}_step_{args.step}.png"), out_img_ep, cmap="gray")

    # メディアンと四分位数を出力
    median_pie = np().median(pmse_list_pie)
    q1_pie = np().percentile(pmse_list_pie, 25)
    q3_pie = np().percentile(pmse_list_pie, 75)

    median_ep = np().median(pmse_list_ep)
    q1_ep = np().percentile(pmse_list_ep, 25)
    q3_ep = np().percentile(pmse_list_ep, 75)

    print("\n========== Summary ==========")
    print(f"[PIE]       PMSE Median: {median_pie:.4f} dB  (Q1={q1_pie:.4f}, Q3={q3_pie:.4f})")
    print(f"[Ptycho-EP] PMSE Median: {median_ep:.4f} dB  (Q1={q1_ep:.4f}, Q3={q3_ep:.4f})")

if __name__ == "__main__":
    main()

import sys 
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ptychoep.backend.backend import set_backend, np
set_backend("numpy")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from ptychoep.ptycho.core import Ptycho
from ptychoep.utils.io_utils import load_data_image
from ptychoep.ptycho.aperture_utils import circular_aperture
from ptychoep.ptycho.scan_utils import generate_centered_grid_positions
from ptychoep.ptycho.noise import GaussianNoise
from ptychoep.ptychoep.core import PtychoEP
import os

os.makedirs("experiment/results", exist_ok=True)

# === データ生成 ===
image_size = 512
probe_size = 128
step = 12
num_scans = 16
jitter = 2

amp = np().asarray(load_data_image("cameraman.png"), dtype=np().complex64)
phase = np().asarray(load_data_image("eagle.png"), dtype=np().complex64)
obj = amp * np().exp(1j * np().pi/2 * phase)

probe_gt = np().array(load_data_image("probe.png") , dtype=np().complex64)
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

var = 1e-3
GaussianNoise(var=var, seed=124) @ ptycho

ep_list = []
def record_callback(it, err, current_obj):
    ep_list.append(err)

engine = PtychoEP(ptycho=ptycho, damping=0.5, seed=1, callback=record_callback)
result_ep = engine.run(n_iter=100)
reconstruction = result_ep[0]
precision_map = result_ep[1]

def phase_align(x_est, x_true):
    inner_product = np().vdot(x_true, x_est)
    phase = np().angle(inner_product)
    return x_est * np().exp(-1j * phase)

reconstruction_aligned = phase_align(reconstruction, obj)
amplitude_error = np().abs(reconstruction_aligned - obj)
standard_deviation = 1.0 / np().sqrt(precision_map)

crop = slice(128, 384)
normalized_error = (amplitude_error[crop, crop] / standard_deviation[crop, crop]).flatten()
ratio_in_range = ((normalized_error > 0.5) & (normalized_error < 2.0)).mean()
print(f"Ratio of pixels within [0.5σ, 2σ]: {ratio_in_range:.2%}")

# === 可視化と保存 ===
plt.imsave("../result/reconstruction_amplitude.png",
           np().abs(reconstruction_aligned[crop, crop]), cmap="gray")
# カラーバー付きで標準偏差マップを保存
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(standard_deviation[crop, crop], cmap="magma", norm=LogNorm(vmin=1e-2, vmax=1e0))
ax.set_title("Estimated Standard Deviation")

# 軸の目盛りを非表示にする
ax.axis("off")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Std. Dev. (1 / sqrt(Γ_int))")
plt.tight_layout()
plt.savefig("../result/estimated_stddev_with_colorbar.png")



plt.figure(figsize=(6, 4))
plt.hist(normalized_error, bins=100, range=(0, 3), density=True, alpha=0.7, label='Normalized Error')
plt.axvline(0.5, color='red', linestyle='--', label='Bounds 0.5 / 2.0')
plt.axvline(2.0, color='red', linestyle='--')
plt.title("Distribution of Normalized Reconstruction Error")
plt.xlabel(r"$|x_{\mathrm{est}} - x_{\mathrm{true}}| / \sigma$")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../result/uncertainty_histogram.png")

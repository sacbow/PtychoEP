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


# === データ生成 ===
image_size = 512
probe_size = 128
step = 12
num_scans = 16
jitter = 2
data_seed = 999
algorithm_seed = 123

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
    seed=data_seed
)

ptycho = Ptycho()
ptycho.set_object(obj)
ptycho.set_probe(probe_gt)
ptycho.forward_and_set_diffraction(positions)
ptycho.sort_diffraction_data()

var = 1e-3
GaussianNoise(var=var, seed=data_seed) @ ptycho

ep_list = []
def record_callback(it, err, current_obj):
    ep_list.append(err)

engine = PtychoEP(ptycho=ptycho, damping=0.5, seed=algorithm_seed, callback=record_callback)
result_ep = engine.run(n_iter=200)
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
fig1, ax1 = plt.subplots(figsize=(6, 5))
im1 = ax1.imshow(standard_deviation[crop, crop], cmap="magma", norm=LogNorm(vmin=2e-2, vmax=1e0))
ax1.set_title("Estimated Standard Deviation", fontsize = 20)
# 軸の目盛りを非表示にする
ax1.axis("off")
cbar1 = fig1.colorbar(im1, ax=ax1)
plt.tight_layout()
plt.savefig("../result/estimated_stddev_with_colorbar.png")

#-------  追加 ------
# カラーバー付きで誤差マップを保存
fig2, ax2 = plt.subplots(figsize=(6, 5))
im2 = ax2.imshow(amplitude_error[crop, crop], cmap="magma", norm=LogNorm(vmin=2e-2, vmax=1e0))
ax2.set_title("Absolute Error", fontsize = 20)
# 軸の目盛りを非表示にする
ax2.axis("off")
cbar2 = fig2.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.savefig("../result/absolute_error.png")





plt.figure(figsize=(6, 4))
plt.hist(normalized_error, bins=100, range=(0, 3), density=True, alpha=0.7, label='Normalized Error')
plt.axvline(0.5, color='red', linestyle='--', label='Bounds 0.5 / 2.0')
plt.axvline(2.0, color='red', linestyle='--')
plt.title("Distribution of Normalized Reconstruction Error", fontsize = 15)
plt.xlabel(r"$|O_{\mathrm{est}} - O_{\mathrm{true}}| / \sigma$", fontsize = 15)
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../result/uncertainty_histogram.png")

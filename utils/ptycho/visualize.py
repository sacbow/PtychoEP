import matplotlib.pyplot as plt
from ..backend import np

def compute_illumination(ptycho):
    """
    照明分布(scan_img)とサンプリング率(alpha)を計算する。
    forward生成時に計算されたindicesを利用。
    """
    prb_sq = np().abs(ptycho.prb) ** 2
    scan_img = np().zeros((ptycho.obj_len, ptycho.obj_len), dtype=float)
    
    for d in ptycho._diff_data:
        Y, X = d.indices  # forward時に保存したindicesを利用
        np().add.at(scan_img, (Y, X), prb_sq)

    sampling_number = np().sum(scan_img > 0.1 * np().max(scan_img))
    alpha = (ptycho.prb_len ** 2 * len(ptycho.scan_pos)) / sampling_number
    return scan_img, alpha



def show_scan_and_diffs(ptycho, num_patterns=4, log_scale=True, cmap_obj="gray", cmap_diff="jet"):
    """
    スキャン照明分布と回折像の可視化。
    """
    from numpy.fft import fftshift

    scan_img, alpha = compute_illumination(ptycho)
    fig, axes = plt.subplots(1, num_patterns + 1, figsize=(3 * (num_patterns + 1), 3))

    # スキャン照明分布
    axes[0].imshow(scan_img, cmap=cmap_obj)
    axes[0].axis("off")
    axes[0].set_title(f"Scanned positions\n(alpha={alpha:.2f})", fontsize=12)

    # 回折像 (先頭からnum_patterns枚)
    for i in range(num_patterns):
        ax = axes[i + 1]
        ax.axis("off")
        diff = ptycho.diffs[i]
        if log_scale:
            diff = np().log10(np().abs(fftshift(diff)) + 1e-8)
        else:
            diff = np().abs(fftshift(diff))
        ax.imshow(diff, cmap=cmap_diff)
        ax.set_title(f"Diffraction {i+1}", fontsize=12)

    plt.tight_layout()
    plt.show()

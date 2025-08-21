# test/test_engines_pie.py
import numpy as np
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.ptycho.forward import generate_diffraction
from PtychoEP.classic_engines.pie import PIE

def test_pie_runs_and_reduces_error():
    # 1. データ準備
    ptycho = Ptycho()
    obj = load_data_image("cameraman.png")  # 正規化済み
    prb = load_data_image("probe.png")
    ptycho.set_object(obj.astype(np.complex64))
    ptycho.set_probe(prb.astype(np.complex64))

    # スキャン座標生成 & 回折データ計算
    positions = generate_spiral_scan_positions(image_size=ptycho.obj_len,
                                               probe_size=ptycho.prb_len,
                                               num_points=10)
    ptycho.forward_and_set_diffraction(positions)

    # 2. PIEアルゴリズム実行
    errors = []
    def callback(iter_idx, err, obj_est):
        errors.append(err)

    pie = PIE(ptycho, alpha=0.1, callback=callback)
    obj_est = pie.run(n_iter=5)

    # 3. テスト: 誤差が減少しているか
    assert len(errors) == 5
    assert errors[-1] <= errors[0]  # 初期より減少しているはず
    assert obj_est.shape == obj.shape  # 出力形状が一致

import pytest
from PtychoEP.utils.backend import set_backend, np
from PtychoEP.utils.ptycho.core import Ptycho
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.utils.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.utils.engines.rpie import rPIE

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_rpie_runs_and_reduces_error(backend):
    # バックエンド設定
    set_backend(backend)
    ptycho = Ptycho()

    # オブジェクトとプローブ設定（複素数dtypeを統一）
    obj = np().array(load_data_image("cameraman.png"), dtype=np().float32)
    obj_phase = np().array(load_data_image("eagle.png"), dtype=np().float32)
    obj = obj * np().exp(1j * np().pi * obj_phase).astype(np().complex64)
    probe = np().array(load_data_image("probe.png"), dtype=np().complex64)

    ptycho.set_object(obj)
    ptycho.set_probe(probe)

    # スキャン座標生成＆回折像生成
    positions = generate_spiral_scan_positions(image_size=512, probe_size=128, num_points=50)
    ptycho.forward_and_set_diffraction(positions)

    # エラー記録用コールバック
    errors = []
    def callback(iter_idx, err, obj_est):
        errors.append(err)

    # rPIE実行
    rpie = rPIE(ptycho, alpha=0.1, beta=0.1, callback=callback)
    obj_est, prb_est = rpie.run(n_iter=10)

    # --- テスト ---
    assert len(errors) == 10, "コールバックが全イテレーションで呼ばれていません"
    assert errors[0] > errors[-1], "誤差が収束していません"
    assert obj_est.shape == obj.shape, "オブジェクト推定の形状が不一致"
    assert prb_est.shape == probe.shape, "プローブ推定の形状が不一致"

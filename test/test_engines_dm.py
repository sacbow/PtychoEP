import pytest
from PtychoEP.backend.backend import set_backend, np
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.classic_engines.difference_map import DifferenceMap

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_difference_map_runs_and_reduces_error(backend):
    """DifferenceMapが正常に動作し、エラーが減少することを確認"""
    set_backend(backend)

    # --- Ptychoデータ生成 ---
    ptycho = Ptycho()
    xp = np()
    obj = xp.array(load_data_image("cameraman.png")) * xp.exp(1j * xp.pi * xp.array(load_data_image("eagle.png")))
    probe = xp.array(load_data_image("probe.png"), dtype=xp.complex64)

    ptycho.set_object(obj)
    ptycho.set_probe(probe)
    positions = generate_spiral_scan_positions(image_size=512, probe_size=128, num_points=50)
    ptycho.forward_and_set_diffraction(positions)

    # --- DifferenceMapの実行 ---
    errors = []

    def callback(iter_idx, err, obj_est):
        errors.append(err)

    dm = DifferenceMap(ptycho, beta=1.0, callback=callback)
    obj_est, prb_est = dm.run(n_iter=10)

    # --- テスト: エラーが反復ごとに記録されているか ---
    assert len(errors) == 10, "各イテレーションでエラーが記録されるはず"

    # --- テスト: エラーが減少傾向にあるか (初期値 > 終了値) ---
    assert errors[0] > errors[-1], f"誤差が減少していません: {errors[0]} → {errors[-1]}"

    # --- テスト: 出力オブジェクト/プローブのサイズ確認 ---
    assert obj_est.shape == obj.shape
    assert prb_est.shape == probe.shape
    assert xp.iscomplexobj(obj_est) and xp.iscomplexobj(prb_est), "推定結果は複素数配列である必要がある"

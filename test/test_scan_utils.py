import pytest
from numbers import Integral
from ptychoep.ptycho.scan_utils import (
    generate_spiral_scan_positions,
    generate_grid_scan_positions,
)
from ptychoep.backend.backend import set_backend

@pytest.fixture(autouse=True)
def setup_backend():
    """テストはnumpyバックエンドで固定"""
    set_backend("numpy")

# --- Spiral scan tests ---
def test_spiral_scan_basic():
    """スパイラルスキャンが指定数の点を生成し、範囲内に収まるかを確認"""
    positions = generate_spiral_scan_positions(
        image_size=128, probe_size=16, num_points=200, step=5.0
    )
    assert len(positions) <= 200  # bounds_checkにより減る場合がある

    for y, x in positions:
        # 整数型であること
        assert isinstance(y, Integral)
        assert isinstance(x, Integral)
        # 画像範囲からプローブがはみ出さない
        assert 0 <= y - 8 and y + 8 <= 128
        assert 0 <= x - 8 and x + 8 <= 128

def test_spiral_scan_custom_center():
    """中心座標を指定した場合のスパイラルスキャン"""
    center = (32, 64)
    positions = generate_spiral_scan_positions(
        image_size=128, probe_size=16, num_points=50, step=4.0, center=center
    )
    # 最初の点は中心に近いはず
    y0, x0 = positions[0]
    assert abs(y0 - center[0]) <= 1
    assert abs(x0 - center[1]) <= 1

def test_spiral_scan_bounds_off():
    """bounds_check=Falseで画像外座標が許容されること"""
    positions = generate_spiral_scan_positions(
        image_size=64, probe_size=16, num_points=100, step=20.0, bounds_check=False
    )
    assert any(y < 0 or x < 0 or y >= 64 or x >= 64 for y, x in positions)

# --- Grid scan tests ---
def test_grid_scan_no_jitter():
    """ジッターなし格子スキャンの基本"""
    positions = generate_grid_scan_positions(
        image_size=64, probe_size=16, step=16, jitter=0
    )
    # 格子座標はすべてstep間隔
    ys = sorted(set(y for y, _ in positions))
    xs = sorted(set(x for _, x in positions))
    assert ys[1] - ys[0] == 16
    assert xs[1] - xs[0] == 16

def test_grid_scan_with_jitter():
    """ジッター付き格子スキャンで格子座標からずれが生じる"""
    positions = generate_grid_scan_positions(
        image_size=64, probe_size=16, step=16, jitter=3
    )
    assert len(positions) > 0
    # 少なくとも1点は格子からずれている
    assert any((y % 16 != 0 or x % 16 != 0) for y, x in positions)

def test_grid_scan_bounds_check():
    """bounds_checkの有効・無効で結果が変わる"""
    # 有効時
    positions_bc = generate_grid_scan_positions(
        image_size=32, probe_size=16, step=8, jitter=0, bounds_check=True
    )
    assert all(0 <= y - 8 and y + 8 <= 32 and 0 <= x - 8 and x + 8 <= 32 for y, x in positions_bc)

    # 無効時
    positions_no_bc = generate_grid_scan_positions(
        image_size=32, probe_size=16, step=8, jitter=0, bounds_check=False
    )
    assert any(y - 8 < 0 or y + 8 > 32 or x - 8 < 0 or x + 8 > 32 for y, x in positions_no_bc)

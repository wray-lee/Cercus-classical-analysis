"""T1 golden test: select_vmax_threshold cascade (KDE valley → fallback)."""
import numpy as np

from cercus.analysis import vmax_threshold as vt


def test_bimodal_picks_kde_valley(monkeypatch):
    monkeypatch.setattr(vt, "USE_ADAPTIVE_THRESHOLD", True)
    rng = np.random.default_rng(0)
    vmax = np.concatenate([rng.normal(20, 5, 400), rng.normal(200, 30, 200)])
    thr, method, info = vt.select_vmax_threshold(vmax)
    assert method == "KDE valley"
    assert 40 < thr < 160, thr  # valley between quiet/burst clusters
    assert "gmm_escape_threshold" in info


def test_single_peak_falls_back(monkeypatch):
    monkeypatch.setattr(vt, "USE_ADAPTIVE_THRESHOLD", True)
    monkeypatch.setattr(vt, "FALLBACK_VMAX_THRESHOLD", 120.0)
    # 单峰高斯（>10 样本，KDE 无第二峰；Log-GMM 因样本少/分离差返回 None 或仍失败时
    # 走 fallback）——用极少量样本直接触发各方法 skip
    vmax = np.array([50.0] * 12)
    thr, method, _ = vt.select_vmax_threshold(vmax)
    assert method == "hardcoded fallback"
    assert thr == 120.0


def test_fixed_config_mode(monkeypatch):
    monkeypatch.setattr(vt, "USE_ADAPTIVE_THRESHOLD", False)
    monkeypatch.setattr(vt, "ESCAPE_VMAX_THRESHOLD", 98.0)
    thr, method, _ = vt.select_vmax_threshold(np.linspace(1, 300, 100))
    assert method == "config (fixed)"
    assert thr == 98.0

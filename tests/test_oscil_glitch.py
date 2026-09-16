"""stim_state 单帧毛刺（如 967）不得撑爆示波器 wind fill 多边形（顶刊图 bug 回归）。"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.visualization._core import draw_oscilloscope_channels


def _df_with_glitch(glitch: float = 967.0) -> pd.DataFrame:
    t = np.linspace(-1500, 1500, 300)
    stim = np.where((t >= -373) & (t <= -200), 1, 0).astype(float)
    stim[150] = glitch
    return pd.DataFrame({
        "global_trial_id": 0,
        "type": "looming_wind",
        "t_rel": t,
        "stim_state": stim,
    })


def test_glitch_stim_state_clipped():
    fig, ax = plt.subplots()
    draw_oscilloscope_channels(ax, _df_with_glitch(), "looming_wind")
    verts = np.concatenate(
        [p.vertices for c in ax.collections for p in c.get_paths()]
    )
    # y 必须在 [0, 5] 轴域内（钳制后 fill 高度 ≤ wind_baseline+1）
    assert verts[:, 1].min() > 0.0, "polygon escapes below axis (glitch spike)"
    assert verts[:, 1].max() < 5.0, "polygon escapes above axis (glitch spike)"
    plt.close(fig)


def test_clean_signal_unchanged():
    fig, ax = plt.subplots()
    draw_oscilloscope_channels(ax, _df_with_glitch(glitch=1.0), "looming_wind")
    tops = np.concatenate(
        [p.vertices for c in ax.collections for p in c.get_paths()]
    )
    # 正常 wind 块顶边 = wind_baseline + 1 = 4
    assert np.isclose(tops[:, 1].max(), 4.0)
    plt.close(fig)

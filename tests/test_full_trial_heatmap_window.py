"""Full-trial heatmaps cover the observed times after each alignment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cercus.visualization import plot_spaghetti_kinetics_heatmap, plot_trial_stacked_heatmap
from population_analysis import _full_trial_heatmap_windows


def test_full_trial_windows_include_all_samples_after_alignment():
    df = pd.DataFrame({
        "subject_id": ["subj"] * 6,
        "global_trial_id": [0] * 3 + [1] * 3,
        "response_type": ["Escape"] * 6,
        "t_rel": [-3000.0, 0.0, 2600.0, -1400.0, 500.0, 4500.0],
        "interval_onset_ms": [np.nan] * 3 + [500.0] * 3,
        "speed": [20.0] * 3 + [30.0] * 3,
    })

    ttc, onset, density = _full_trial_heatmap_windows(df, t_bin_s=0.005, dt_ms=2.0)

    assert ttc == pytest.approx((-3.0, 4.505))
    assert onset == pytest.approx((-3.0, 4.005))
    assert density == pytest.approx((-3000.0, 4002.0))

    for align, window, latest in (("ttc", ttc, 4.5), ("onset", onset, 4.0)):
        fig = plot_trial_stacked_heatmap(
            df, align=align, t_window=window, t_bin_s=0.005,
            conditions=["Escape"],
        )
        image = fig.axes[0].images[0]
        times = np.arange(*window, 0.005)
        last = np.argmin(abs(times - latest))
        assert image.get_extent()[1] >= latest
        assert image.get_array()[:, last].max() == pytest.approx(30.0)
        plt.close(fig)

    fig = plot_spaghetti_kinetics_heatmap(df, t_window=density, dt=2.0)
    assert max(fig.axes[0].lines[0].get_xdata()) >= 4000.0
    plt.close(fig)

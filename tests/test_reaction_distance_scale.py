"""Population distance panels use one shared axis across datasets."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.config import get_visualization
from cercus.visualization.behavior import plot_reaction_distance_panel


def test_distance_axis_is_comparable_across_datasets():
    for distance in (100.0, 340.0):
        df = pd.DataFrame({
            "subject_id": ["a", "b"],
            "global_trial_id": [1, 1],
            "response_type": ["Escape", "Escape"],
            "reaction_time_ms": [50.0, 60.0],
            "distance_mm": [distance, distance + 1.0],
        })
        fig = plot_reaction_distance_panel(df)
        assert fig.axes[1].get_ylim() == (0.0, get_visualization().reaction_distance_max_mm)
        plt.close(fig)


def test_prewalk_rt_uses_escape_and_pairs_only_complete_wind_endpoints():
    df = pd.DataFrame({
        "subject_id": ["a", "a", "b", "c"],
        "global_trial_id": [1, 2, 1, 1],
        "response_type": ["PreWalk"] * 4,
        "type": ["baseline_wind", "baseline_wind", "looming_wind", "baseline_visual"],
        "reaction_time_ms": [30.0, 300.0, 40.0, 900.0],
        "escape_reaction_time_ms": [120.0, np.nan, 140.0, np.nan],
        "stillness_reaction_time_ms": [30.0, 300.0, 40.0, 900.0],
        "stop_to_escape_interval_ms": [90.0, np.nan, 100.0, 900.0],
        "distance_mm": [10.0] * 4,
    })
    before = df.copy(deep=True)
    fig = plot_reaction_distance_panel(df)
    assert len(fig.axes) == 3
    assert sorted(fig.axes[0].collections[0].get_offsets()[:, 1]) == [120.0, 140.0]
    assert [c.get_offsets()[0, 1] for c in fig.axes[2].collections] == [30.0, 90.0, 40.0, 100.0]
    assert "paired: 2 trials / 2 subjects" in [t.get_text() for t in fig.axes[2].texts]
    pd.testing.assert_frame_equal(df, before)
    plt.close(fig)


def test_rt_and_t1_t2_prefer_same_causal_pair_without_imputation():
    df = pd.DataFrame({
        "subject_id": ["a", "b"], "global_trial_id": [1, 1],
        "response_type": ["PreWalk"] * 2, "type": ["baseline_wind"] * 2,
        "reaction_time_ms": [30.0] * 2, "escape_reaction_time_ms": [90.0] * 2,
        "stillness_reaction_time_ms": [30.0] * 2, "stop_to_escape_interval_ms": [60.0] * 2,
        "pause_stopping_time_ms": [70.0] * 2,
        "pause_to_escape_time_ms": [50.0, np.nan],
        "pause_reaction_time_ms": [120.0, np.nan],
        "pause_status": ["observed"] * 2,
        "pause_baseline_status": ["continuous_moving", "intermittent_moving"],
        "distance_mm": [10.0] * 2,
    })
    fig = plot_reaction_distance_panel(df)
    assert list(fig.axes[0].collections[0].get_offsets()[:, 1]) == [120.0]
    assert [c.get_offsets()[0, 1] for c in fig.axes[2].collections] == [70.0, 50.0]
    assert "Pure: moving 1/2; paired 1/1" in [t.get_text() for t in fig.axes[2].texts]
    assert "RT observed — PreWalk: 1/2" in [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)


def test_t1_t2_lines_are_actual_trial_pairs_with_subject_level_summaries():
    df = pd.DataFrame({
        "subject_id": ["a", "a"], "global_trial_id": [1, 2],
        "response_type": ["PreWalk"] * 2, "type": ["baseline_wind"] * 2,
        "escape_reaction_time_ms": [120.0, 230.0], "distance_mm": [10.0] * 2,
        "pause_stopping_time_ms": [70.0, 200.0],
        "pause_to_escape_time_ms": [50.0, 30.0],
        "pause_reaction_time_ms": [120.0, 230.0], "pause_status": ["observed"] * 2,
    })
    fig = plot_reaction_distance_panel(df, pause_cohort="all_local")
    lines = [tuple(line.get_ydata()) for line in fig.axes[2].lines
             if len(line.get_xdata()) == 2 and list(line.get_xdata()) == [0, 1]]
    assert (70.0, 50.0) in lines
    assert (200.0, 30.0) in lines
    assert (135.0, 40.0) not in lines
    assert [c.get_offsets()[0, 1] for c in fig.axes[2].collections] == [135.0, 40.0]

    # Actual trial pair lines have marker dots that coincide with line endpoints
    trial_lines = [l for l in fig.axes[2].lines
                   if len(l.get_xdata()) == 2 and list(l.get_xdata()) == [0, 1]
                   and l.get_marker() == "o"]
    assert len(trial_lines) == 2
    assert all(line.get_markevery() is None for line in trial_lines)

    # Coverage and legend text must not obscure the data.
    for txt in fig.axes[2].texts:
        x, y = txt.get_position()
        assert y < 0.0 or y > 1.0, f"Annotation {txt.get_text()!r} is inside data area (y={y})"

    plt.close(fig)


def test_missing_escape_rt_does_not_fall_back_to_stopping_rt():
    df = pd.DataFrame({
        "subject_id": ["a"], "global_trial_id": [1],
        "response_type": ["PreWalk"], "type": ["baseline_wind"],
        "reaction_time_ms": [30.0], "escape_reaction_time_ms": [np.nan],
        "distance_mm": [10.0],
    })
    fig = plot_reaction_distance_panel(df)
    assert not fig.axes[0].collections
    assert "no data" in [t.get_text() for t in fig.axes[0].texts]
    assert "no paired wind PreWalk endpoints" in [t.get_text() for t in fig.axes[2].texts]
    plt.close(fig)


def test_causal_t1_t2_default_is_strict_moving_with_explicit_local_view():
    df = pd.DataFrame({
        "subject_id": ["a", "b"], "global_trial_id": [1, 1],
        "response_type": ["PreWalk"] * 2, "type": ["baseline_wind"] * 2,
        "escape_reaction_time_ms": [120.0, 230.0], "distance_mm": [10.0] * 2,
        "pause_stopping_time_ms": [70.0, 200.0],
        "pause_to_escape_time_ms": [50.0, 30.0],
        "pause_reaction_time_ms": [120.0, 230.0], "pause_status": ["observed"] * 2,
        "pause_baseline_status": ["continuous_moving", "intermittent_moving"],
    })
    fig = plot_reaction_distance_panel(df)
    assert [c.get_offsets()[0, 1] for c in fig.axes[2].collections] == [70.0, 50.0]
    assert "Strict moving" in fig.axes[2].get_title()
    plt.close(fig)
    fig = plot_reaction_distance_panel(df, pause_cohort="all_local")
    assert [list(c.get_offsets()[:, 1]) for c in fig.axes[2].collections] == [[70.0, 200.0], [50.0, 30.0]]
    assert "Local transitions" in fig.axes[2].get_title()
    plt.close(fig)


def test_strict_moving_panel_never_borrows_intermittent_pairs():
    df = pd.DataFrame({
        "subject_id": ["a"], "global_trial_id": [1], "response_type": ["PreWalk"],
        "type": ["baseline_wind"], "escape_reaction_time_ms": [120.0],
        "distance_mm": [10.0], "pause_stopping_time_ms": [70.0],
        "pause_to_escape_time_ms": [50.0], "pause_reaction_time_ms": [120.0],
        "pause_status": ["observed"], "pause_baseline_status": ["intermittent_moving"],
    })
    fig = plot_reaction_distance_panel(df)
    assert not fig.axes[2].collections
    assert "no paired strict-moving endpoints" in [t.get_text() for t in fig.axes[2].texts]
    assert "Pure: moving 0/1; paired 0/0" in [t.get_text() for t in fig.axes[2].texts]
    plt.close(fig)
    df = df.drop(columns="pause_baseline_status")
    fig = plot_reaction_distance_panel(df)
    assert not fig.axes[2].collections  # missing cohort is not eligibility
    plt.close(fig)


def test_strict_moving_cohort_is_independent_of_legacy_response_class():
    df = pd.DataFrame({
        "subject_id": ["a"], "global_trial_id": [1], "response_type": ["NoResponse"],
        "type": ["baseline_wind"], "escape_reaction_time_ms": [np.nan],
        "distance_mm": [np.nan], "pause_stopping_time_ms": [100.0],
        "pause_to_escape_time_ms": [220.0], "pause_reaction_time_ms": [320.0],
        "pause_status": ["observed"], "pause_baseline_status": ["continuous_moving"],
    })
    fig = plot_reaction_distance_panel(df)
    assert [c.get_offsets()[0, 1] for c in fig.axes[2].collections] == [100.0, 220.0]
    plt.close(fig)

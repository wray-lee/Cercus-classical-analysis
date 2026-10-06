"""Population behavior plots: raw split view vs derived merged grouping."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cercus.visualization.behavior import plot_population_behavior_probability


def _labels(fig):
    return [tick.get_text() for tick in fig.axes[0].get_xticklabels()]


def test_population_behavior_omits_missing_preescape():
    """Split view (merge off) keeps raw labels and omits absent PreEscape."""
    for types, expected in (
        (["Escape", "PreWalk", "NoResponse"], False),
        (["Escape", "PreEscape", "PreWalk"], True),
    ):
        df = pd.DataFrame({
            "subject_id": ["s1", "s2", "s3"],
            "global_trial_index": [0, 0, 0],
            "response_type": types,
        })
        fig = plot_population_behavior_probability(df, merge_prewalk=False)
        ax = fig.axes[0]
        assert ("PreEscape" in _labels(fig)) == expected
        assert "PreWalk" in _labels(fig)
        assert len(ax.collections) == (4 if expected else 3)
        plt.close(fig)


def test_population_behavior_merges_prewalk_into_escape():
    """Merged view (default) folds raw PreWalk into the derived Escape bar."""
    df = pd.DataFrame({
        "subject_id": ["s1", "s2", "s3"],
        "global_trial_index": [0, 0, 0],
        "response_type": ["Escape", "PreWalk", "NoResponse"],
    })
    fig = plot_population_behavior_probability(df, merge_prewalk=True)
    labels = _labels(fig)
    assert "PreWalk" not in labels
    assert labels.count("Escape") == 1
    # Two classes (Escape, NoResponse); per-subject scatter collections match.
    assert len(fig.axes[0].collections) == 2
    plt.close(fig)

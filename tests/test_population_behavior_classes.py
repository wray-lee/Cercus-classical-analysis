"""Population behavior plots omit nonexistent PreEscape observations."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cercus.visualization.behavior import plot_population_behavior_probability


def test_population_behavior_omits_missing_preescape():
    for types, expected in (
        (["Escape", "PreWalk", "NoResponse"], False),
        (["Escape", "PreEscape", "PreWalk"], True),
    ):
        df = pd.DataFrame({
            "subject_id": ["s1", "s2", "s3"],
            "global_trial_index": [0, 0, 0],
            "response_type": types,
        })
        fig = plot_population_behavior_probability(df)
        ax = fig.axes[0]
        assert ("PreEscape" in [tick.get_text() for tick in ax.get_xticklabels()]) == expected
        assert len(ax.collections) == (4 if expected else 3)
        plt.close(fig)

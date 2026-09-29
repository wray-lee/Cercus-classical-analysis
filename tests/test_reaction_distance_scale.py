"""Population distance panels use one shared axis across datasets."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

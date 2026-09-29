"""Optional paradigm inputs retain the comparison's existing mirror directions."""

from itertools import combinations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

# The legacy entry point applies publication rcParams at import time.
with matplotlib.rc_context():
    import plot_multisensory_trajectories as command
from cercus.cli.app import app
from cercus.visualization import trajectories


PARADIGMS = ("bv", "bw", "ms")
SUBSETS = [subset for n in (1, 2, 3) for subset in combinations(PARADIGMS, n)]
TITLES = {"bv": "Baseline Visual", "bw": "Baseline Wind", "ms": "Multisensory"}


@pytest.mark.parametrize("present", SUBSETS)
def test_supplied_paradigms(monkeypatch, tmp_path, present):
    df = pd.DataFrame({
        "global_trial_id": [0, 0, 1, 1],
        "t_rel": [0.0, 10.0, 0.0, 10.0],
        "screen_side": ["left", "left", "right", "right"],
    })
    offsets = []

    def trajectory(grp, *_args, **kwargs):
        offsets.append(kwargs["wind_offset_deg_override"])
        sign = -1 if grp["screen_side"].iloc[0] == "left" else 1
        return np.array([0.0, sign * 80.0]), np.array([0.0, 120.0])

    monkeypatch.setattr(trajectories, "compute_trajectory_masks", trajectory)
    fig = trajectories.plot_multisensory_trajectory_comparison(
        df, df, df, present=present, wind_offset_deg=12.5,
    )
    try:
        ax = fig.axes[0]
        lines = ax.lines[:2 * len(present)]
        colors = {"bv": trajectories.COLOR_LEFT, "bw": trajectories.COLOR_RIGHT,
                  "ms": trajectories.COLOR_MULTISENSORY}
        for index, label in enumerate(present):
            for line in lines[2 * index:2 * index + 2]:
                assert line.get_xdata()[-1] == (-80 if label == "ms" else 80)
                assert line.get_color() == colors[label]
        assert offsets == [12.5 if label == "bw" else None
                           for label in present for _ in range(2)]
        if len(present) == 1:
            label = present[0]
            assert ax.get_xlim() == (-200, 200)
            assert ax.get_title() == f"{TITLES[label]} (n=2)"
            assert ax.get_legend() is None
            assert fig.get_size_inches()[0] == 6
            headings = {text.get_text(): text.get_position()[0] for text in ax.texts}
            assert headings["Multisensory" if label == "ms" else "Baseline"] == (
                -100 if label != "ms" else 100
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            assert ax.title.get_window_extent(renderer).y0 >= ax.get_window_extent().y1
            assert ax.get_window_extent().width / ax.get_window_extent().height == pytest.approx(1.0)
        else:
            assert ax.get_xlim() == (-200, 200)
            assert [text.get_text() for text in ax.get_legend().get_texts()] == [
                f"{TITLES[label]} (n=2)" for label in present
            ]
            headings = {text.get_text(): text.get_position()[0] for text in ax.texts}
            assert headings["Baseline"] < 0
            if "ms" in present:
                assert headings["Multisensory"] > 0
            else:
                assert "Multisensory" not in headings
        fig.savefig(tmp_path / "preview.png", bbox_inches="tight")
        fig.savefig(tmp_path / "preview.svg", bbox_inches="tight")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("present", SUBSETS)
def test_cli_loads_only_supplied_inputs(monkeypatch, tmp_path, present):
    loaded = []

    def load(path, escape_only=False):
        loaded.append((path.name, escape_only))
        return pd.DataFrame()

    monkeypatch.setattr(command, "_load_dir", load)
    output = tmp_path / "result.svg"
    argv = ["multisensory-traj", "--output", str(output)]
    for label in present:
        path = tmp_path / label
        path.mkdir()
        argv.extend([f"--{label}", str(path)])
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert loaded == [(label, False) for label in present]
    svg = output.read_text()
    for label in PARADIGMS:
        assert (f"{TITLES[label]} (n=0)" in svg) == (label in present)


def test_cli_requires_input_and_validates_before_loading(monkeypatch, tmp_path, capsys):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid arguments must fail before loading data")

    monkeypatch.setattr(command, "_load_dir", unexpected_load)
    output = tmp_path / "result.svg"
    result = CliRunner().invoke(app, ["multisensory-traj", "-o", str(output)])
    assert result.exit_code == 2
    assert "Provide at least one of --bv, --bw, or --ms" in result.output
    with pytest.raises(SystemExit) as error:
        command.main(["--bv", str(tmp_path), "--ms", str(tmp_path / "missing"),
                      "--output", str(output)])
    assert error.value.code == 2
    assert "--ms is not a directory" in capsys.readouterr().err
    assert not output.exists()


def test_legacy_escape_only(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(command, "_load_dir", lambda path, escape_only:
                        loaded.append(escape_only) or pd.DataFrame())
    command.main(["--bw", str(tmp_path), "--escape-only",
                  "--output", str(tmp_path / "result.svg")])
    assert loaded == [True]


def test_default_comparison_and_invalid_selection():
    empty = pd.DataFrame()
    fig = trajectories.plot_multisensory_trajectory_comparison(empty, empty, empty)
    try:
        assert fig.axes[0].get_xlim() == (-200, 200)
        assert len(fig.axes[0].get_legend().get_texts()) == 3
    finally:
        plt.close(fig)
    for present in ((), ("invalid",), ("bv", "invalid")):
        with pytest.raises(ValueError, match="present must contain"):
            trajectories.plot_multisensory_trajectory_comparison(empty, empty, empty, present=present)

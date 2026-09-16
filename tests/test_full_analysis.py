"""T2: full-mode 数据侧 — 范式发现/排序/跳过 + 聚合表结构与全局阈值。"""
import pandas as pd
import pytest

from cercus.analysis import full as fullmod
from cercus.analysis.full import discover_paradigms, aggregate_paradigm_table


def _write_pair(d, stem):
    """最小合法 events/kinematics 配对（scan 只看文件名；聚合测试不读内容）。"""
    (d / f"{stem}_session_1_events.csv").write_text(
        "event_name,timestamp,global_trial_id,details\ntrial_start,1.0,0,'{}'\n"
    )
    pd.DataFrame({
        "sys_time": [1.0, 1.01, 1.02],
        "dx": 0.1, "dy": 0.0, "dz": 0.0,
        "stim_state": 0, "global_trial_id": 0,
    }).to_csv(d / f"{stem}_session_1_kinematics.csv", index=False)


def test_discover_skips_empty_and_sorts(tmp_path):
    for name in ("+200", "bv", "-373 30°", "-308 36°", "Results"):
        (tmp_path / name).mkdir()
    _write_pair(tmp_path / "bv", "0.7cricket_001")
    _write_pair(tmp_path / "-373 30°", "0.8cricket_001")
    _write_pair(tmp_path / "-308 36°", "0.9cricket_001")
    (tmp_path / "Results" / "dummy_kinematics.csv").write_text("x\n1\n")

    names = [n for n, _ in discover_paradigms(tmp_path)]
    # 空 +200 无配对 → warning 跳过；bv（无数字=视觉-only）最左；首整数升序；Results 排除
    assert names == ["bv", "-373 30°", "-308 36°"]


def test_aggregate_table_and_global_threshold(tmp_path, monkeypatch):
    fake = {
        "bv": {"s1": [], "s2": []},
        "-373 30°": {"s3": []},
    }
    monkeypatch.setattr(fullmod, "discover_paradigms", lambda _d: list(fake.items()))

    def _fake_process(task):
        paradigm, subj, _sess = task
        return pd.DataFrame({
            "paradigm": paradigm, "subject_id": subj,
            "global_trial_index": [0, 1],
            "v_max": [10.0, 300.0] if subj != "s3" else [10.0, 40.0],
            "response_type": ["NoResponse", "Escape"],
        })

    monkeypatch.setattr(fullmod, "_process_subject", _fake_process)
    df, meta = aggregate_paradigm_table(tmp_path, workers=1)

    assert {"paradigm", "subject_id", "is_valid_escape"} <= set(df.columns)
    assert meta["paradigms"] == ["bv", "-373 30°"]
    assert meta["n_subjects"] == {"bv": 2, "-373 30°": 1}
    # 全局单一阈值（非各范式独立）：只有两条 300 mm/s 的过线
    assert df["is_valid_escape"].sum() == 2
    assert 40 < meta["vmax_threshold"] < 300

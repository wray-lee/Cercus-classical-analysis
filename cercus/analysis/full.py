"""
Cercus Framework — Full (Cross-Paradigm) Mode
=============================================
输入 = 范式父目录（每个子目录一个范式）；输出 = 带 ``paradigm`` 列的全量 trial 表
+ 跨范式统一 V_max 阈值 meta（供连线图使用）。

Between-subject by design：动物不跨范式复用（subject id 仅在范式内唯一，
全局键为 ``(paradigm, subject_id)``）。空/无有效配对的范式目录 warning 跳过。
"""

from __future__ import annotations

import logging
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from cercus.analysis.vmax_threshold import select_vmax_threshold
from pipeline.classifier import label_trials
from pipeline.io import load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess

log = logging.getLogger(__name__)

# 排除目录：pipeline 输出 / 测试残留 / full 自身的输出目录约定
_EXCLUDE_EXTRA = {"full", "output", "figures", "results", "test", "train"}


def _paradigm_sort_key(name: str) -> tuple[int, float, str]:
    """非数字（bv/bw 视觉-only）排最左；其余按目录名首个带符号整数升序。"""
    m = re.search(r"[-+]?\d+", name)
    if m is None:
        return (0, 0.0, name)
    return (1, float(m.group()), name)


def discover_paradigms(input_dir: Path | str) -> list[tuple[str, dict[str, list[dict]]]]:
    """每个含有效 (events, kinematics) 配对的子目录 = 一个范式。空目录 warning 跳过。"""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir does not exist: {input_dir}")
    found: list[tuple[str, dict]] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.lower() in _EXCLUDE_EXTRA or child.name.startswith((".", "_")):
            continue
        subjects = scan_and_pair_sessions(child)
        if not subjects:
            log.warning("Paradigm %r has no valid (events, kinematics) pairs — skipped.", child.name)
            continue
        found.append((child.name, subjects))
    found.sort(key=lambda t: _paradigm_sort_key(t[0]))
    return found


def _process_subject(task: tuple[str, str, list]) -> pd.DataFrame:
    """(paradigm, subject_name, sessions) → labeled per-frame DataFrame."""
    paradigm, subject_name, sessions = task
    try:
        all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)
        df["global_trial_index"] = df["global_trial_id"]
        df = label_trials(df)
        df["subject_id"] = subject_name
        df["paradigm"] = paradigm
        return df
    except Exception as exc:  # noqa: BLE001 — 范式级隔离，坏目录不拖垮全局
        log.error("Paradigm %s subject %s failed: %s", paradigm, subject_name, exc)
        return pd.DataFrame()


def aggregate_paradigm_table(
    input_dir: Path | str,
    workers: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline per paradigm, pool one global V_max threshold.

    Returns
    -------
    (df, meta)
        ``df`` = per-frame rows with ``paradigm`` / ``is_valid_escape`` columns;
        ``meta`` = {"vmax_threshold", "vmax_method", "paradigms", "n_subjects"}.
    """
    paradigms = discover_paradigms(input_dir)
    if not paradigms:
        raise ValueError(f"No valid paradigm dirs under {input_dir}")

    tasks = [
        (pname, subj, sess)
        for pname, subjects in paradigms
        for subj, sess in subjects.items()
    ]
    log.info(
        "Full mode: %d paradigms, %d subject-tasks, workers=%s",
        len(paradigms), len(tasks), workers or cpu_count(),
    )
    n = workers or min(cpu_count(), len(tasks))
    if n == 1:
        parts = [_process_subject(t) for t in tasks]
    else:
        with Pool(processes=n) as pool:
            parts = pool.map(_process_subject, tasks)
    parts = [d for d in parts if not d.empty]
    if not parts:
        raise ValueError("No data processed across paradigms.")

    df = pd.concat(parts, ignore_index=True)

    # ── 跨范式统一阈值：池化所有范式的 trial 级 v_max 后只算一次 ──
    trial_vmax = (
        df.groupby(["paradigm", "subject_id", "global_trial_index"])["v_max"]
        .first()
        .dropna()
        .values
    )
    threshold, method, _info = select_vmax_threshold(np.asarray(trial_vmax))
    df["is_valid_escape"] = df["v_max"] >= threshold

    meta = {
        "vmax_threshold": float(threshold),
        "vmax_method": method,
        "paradigms": [p for p, _s in paradigms],
        "n_subjects": {p: len(s) for p, s in paradigms},
    }
    log.info(
        "Global V_max threshold: %.1f mm/s [%s]; paradigms: %s",
        threshold, method, meta["paradigms"],
    )
    return df, meta


__all__ = ["discover_paradigms", "aggregate_paradigm_table"]

"""
Cercus Framework — Fixed Unified Trajectory Overlay (All Paradigms)
===================================================================
Scans all subjects, runs the full preprocessing and classification pipeline,
and produces a single high-density figure with every trial drawn on one
unified physical-coordinate grid.

This variant uses **local coordinate reconstruction** to eliminate the radial
distortion caused by global cumulative heading drift in the pre-computed
x/y columns.

Usage:
    python plot_all_trajectories_fixed.py --input-dir path/to/data/ --save fixed_trajectories.svg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保从 test/ 子目录运行时也能找到顶层 pipeline 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pipeline.classifier import label_trials
from pipeline.constants import (
    COLOR_CONTROL,
    COLOR_ESCAPE,
    COLOR_LEFT,
    COLOR_NO_RESPONSE,
    COLOR_OSCI_HW,
    COLOR_OSCI_VIS,
    COLOR_PREWALK,
    COLOR_RIGHT,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    RADIUS_MM,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    _get_unified_side,
    _apply_publication_style
)
from pipeline.io import load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import (preprocess, compute_escape_latency)
from pipeline.visualization import (_draw_standardized_grid, _draw_side_arrows)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_apply_publication_style()

def plot_global_trajectory_overlay_fixed(
    df: pd.DataFrame,
    TRAJECTORY_MAX_RADIUS_MM: float = 120.0,
    TRAJECTORY_STEP_MM: float = 10.0,
    figsize: tuple[float, float] = (5.0, 5.0),
    alpha: float = 1.0,
    lw: float = 0.3,
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
    #------------- Trajectory Drawing Options -------------
    # USE_Z_DEGREE_TO_DRAW_TRAJECTORY = False
    USE_Z_DEGREE_TO_DRAW_TRAJECTORY: bool = True,
    USE_ESCAPE_ONSET_ONLY: bool = False
    # -------------------------------------------------
) -> plt.Figure:
    """
    Unified trajectory overlay — all paradigms on one axes, with left/right
    stimulus-side colouring and local heading alignment.

    [修复版]:
    1. 引入 subject_id 隔离跨动物的 global_trial_id 碰撞（修复放射状色块）。
    2. 丢弃全局 x/y 累计坐标，每次起步强制使用 dx, dy, dz 重新积分，确保局部 0 rad 起步。
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 修复1: 加入 subject_id 分组条件，防止不同动物的同名 Trial 互相串台导致坐标剧烈跳跃
    group_cols = ["subject_id", "global_trial_id"] if "subject_id" in df.columns else ["global_trial_id"]

    for _keys, grp in df.groupby(group_cols):
        grp = grp.sort_values("t_rel")
        t_vals = grp["t_rel"].values
        speed_vals = grp["speed"].values

        # 获取 TTC 以对齐窗口
        _ttc = grp["target_ttc_ms"].iloc[0] if "target_ttc_ms" in grp.columns else np.nan
        esc = compute_escape_latency(
            t_vals, speed_vals,
            stim_onset_t_rel=float(_ttc) if pd.notna(_ttc) else None,
        )

        # ── 读取当前试次的分类信息 ──
        _latency_ms = esc["latency_ms"]
        _response_type = grp["response_type"].iloc[0] if "response_type" in grp.columns else ""

        # ── 确定有效数据窗口 ──
        if (USE_ESCAPE_ONSET_ONLY
                and _response_type == "Escape"
                and not np.isnan(_latency_ms)):
            # Escape-onset-only mode: slice to the local escape interval
            lat_idx = int(np.argmin(np.abs(t_vals - _latency_ms)))
            # Search forward from latency onset for speed dropping below 10 mm/s
            post_onset_speed = speed_vals[lat_idx:]
            below_mask = post_onset_speed < ESCAPE_START_THRESHOLD
            if np.any(below_mask):
                first_below_local = int(np.argmax(below_mask))
                end_idx = lat_idx + first_below_local
            else:
                end_idx = len(speed_vals) - 1  # fallback: take to end of array

            # Guard: ensure the slice is valid
            if lat_idx >= end_idx:
                end_idx = min(lat_idx + 1, len(speed_vals) - 1)

            burst_mask = np.zeros(len(t_vals), dtype=bool)
            burst_mask[lat_idx:end_idx] = True
        elif not np.isnan(_latency_ms):
            render_start_ms = float(t_vals.min())
            render_start_idx = int(np.argmin(np.abs(t_vals - render_start_ms)))
            actual_start_ms = t_vals[render_start_idx]
            burst_end_ms = float(t_vals.max())
            burst_mask = (t_vals >= actual_start_ms) & (t_vals <= burst_end_ms)
        else:
            render_start_idx = int(np.argmin(np.abs(t_vals - 0.0)))
            actual_start_ms = 0.0
            burst_end_ms = float(t_vals.max())
            burst_mask = (t_vals >= actual_start_ms) & (t_vals <= burst_end_ms)

        if not np.any(burst_mask):
            continue

        burst = grp[burst_mask]

        # 修复2: 绝对丢弃 DataFrame 中自带的全局 x 和 y
        # 提取身体坐标系下的微小位移（根据现有逻辑，原始dx,dy需取反）
        dx_body = -np.nan_to_num(burst["dx"].values)
        dy_body = -np.nan_to_num(burst["dy"].values)
        dz_body = np.nan_to_num(burst["dz"].values)

        if USE_Z_DEGREE_TO_DRAW_TRAJECTORY:
            # 独立航向校准：局部累加 dz，强制第一帧朝向 0 rad
            local_heading = np.cumsum(dz_body) / RADIUS_MM
            local_heading -= local_heading[0]

            # 局部坐标系旋转重构
            dx_global = dx_body * np.cos(local_heading) - dy_body * np.sin(local_heading)
            dy_global = dx_body * np.sin(local_heading) + dy_body * np.cos(local_heading)

            traj_x = np.cumsum(dx_global)
            traj_y = np.cumsum(dy_global)
        else:
            # 若不开启动态 dz 校准，直接对 body 进行干净积分（同样不会产生全局偏移）
            traj_x = np.cumsum(dx_body)
            traj_y = np.cumsum(dy_body)

        traj_x -= traj_x[0]
        traj_y -= traj_y[0]

        # ── 颜色分配与绘图 ──
        ss = _get_unified_side(burst)
        if ss == "left":
            color = left_color
        elif ss == "right":
            color = right_color
        else:
            color = COLOR_CONTROL

        ax.plot(traj_x, traj_y, color=color, alpha=alpha, lw=lw)

    # 绘制背景和参考图例
    _draw_standardized_grid(ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM)
    _draw_side_arrows(ax)

    n_trials = df.groupby(group_cols).ngroups
    ax.set_title(
        f"Unified Trajectory Overlay (fixed)  (n={n_trials} trials)",
        fontweight="bold",
    )

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CLI Parser
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus Fixed Unified Trajectory Overlay — all paradigms, local coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Root directory containing per-subject session CSVs.",
    )
    p.add_argument(
        "--save", required=True,
        help="Path to save the output figure (e.g. fixed_trajectories.svg).",
    )
    return p


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    save_path = Path(args.save)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    # ── Discover subjects ──
    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    # ── Per-subject processing ──
    population_parts: list[pd.DataFrame] = []

    for subject_name, sessions in subjects.items():
        log.info("Processing subject: %s", subject_name)

        # 1. Load & timestamp alignment
        all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)
        df["global_trial_index"] = df["global_trial_id"]

        # 2. Ternary state routing
        df = label_trials(df)

        # 3. Inject subject_id for cross-animal key isolation
        df["subject_id"] = subject_name
        population_parts.append(df)

        n_trials = df["global_trial_index"].nunique()
        log.info("  %s: %d trials classified", subject_name, n_trials)

    if not population_parts:
        log.error("No data processed.")
        return

    # ── Global concatenation ──
    all_data = pd.concat(population_parts, ignore_index=True)

    n_subjects = all_data["subject_id"].nunique()
    n_trials = all_data.groupby(["subject_id", "global_trial_index"]).ngroups
    log.info("Population assembled: %d subjects, %d trials", n_subjects, n_trials)

    # ── Generate fixed unified trajectory overlay ──
    log.info("Generating fixed unified trajectory overlay...")
    fig = plot_global_trajectory_overlay_fixed(all_data)

    # ── Save ──
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved to %s", save_path)


if __name__ == "__main__":
    main()

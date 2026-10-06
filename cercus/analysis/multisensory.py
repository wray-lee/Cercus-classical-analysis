"""Multisensory parent-directory analysis with explicit incomplete-data coverage.

Standard paired source CSVs only. A legacy ``all`` export is not a separate
paradigm and is never silently converted or counted alongside its source data.
Raw classifier endpoints precede derived grouping; uncertainty resamples animals
within each independent paradigm, not frames or pooled trials.
"""
from __future__ import annotations

import csv
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from cercus.analysis.full import _paradigm_sort_key, _process_subject
from cercus.analysis.reaction_time import select_escape_latency
from cercus.analysis.response_groups import read_merge_prewalk, response_group_series
from cercus.analysis.vmax_threshold import select_vmax_threshold
from cercus.config import get_config
from cercus.constants.response_types import BURST_CLASSES, RESPONSE_TYPES
from pipeline.io import scan_and_pair_sessions

log = logging.getLogger(__name__)
_KEYS = ["paradigm", "subject_id", "global_trial_index"]
_CONDITIONS = ["paradigm", "subject_id", "type", "target_ttc_ms", "lv_ratio_ms", "init_half_angle_deg"]
_REQUIRED = {
    "events": {"event_name", "timestamp", "global_trial_id", "details"},
    "kinematics": {"sys_time", "dx", "dy", "dz", "stim_state", "global_trial_id"},
}


def _settings():
    return get_config().analysis.multisensory


def discover_ms_inputs(input_dir: Path | str) -> tuple[list[tuple], pd.DataFrame, list[dict]]:
    """Keep empty/absent expected patterns and reject unsupported session schemas.

    Discovery is per immediate pattern directory. Paths listed in YAML as
    aggregate/output directories are reported separately, never analyzed twice.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input dir does not exist: {root}")
    cfg = _settings()
    excluded = {str(name).lower() for name in cfg.exclude_directories}
    children = {p.name: p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))}
    ignored = [{"paradigm": name, "reason": "aggregate/output directory excluded by analysis.multisensory"}
               for name in children if name.lower() in excluded]
    names = set(cfg.expected_patterns) | {name for name in children if name.lower() not in excluded}
    tasks, rows = [], []
    for name in sorted(names, key=_paradigm_sort_key):
        row = dict(paradigm=name, status="missing", n_subjects_discovered=0,
                   n_subjects=0, n_subjects_failed=0, n_sessions=0, n_trials=0, message="directory absent")
        child = children.get(name)
        if child is not None:
            duplicates: dict[str, list[Path]] = {}
            subjects = scan_and_pair_sessions(child, duplicate_files=duplicates)
            row.update(n_subjects_discovered=len(subjects), n_sessions=sum(map(len, subjects.values())),
                       status="empty", message="no paired sessions")
            errors = []
            for subject, sessions in subjects.items():
                # Use the scanner's normalized keys and excluded-directory rules.
                if subject in duplicates:
                    collisions = sorted({str(path.relative_to(child)) for path in duplicates[subject]})
                    errors.append(f"{subject}: duplicate session keys: {collisions}")
                    row["n_subjects_failed"] += 1
                    continue
                try:
                    for session in sessions:
                        for kind, required in _REQUIRED.items():
                            with session[kind].open(newline="", encoding="utf-8-sig") as stream:
                                header = next(csv.reader(stream), [])
                            missing = required - set(header)
                            if missing:
                                raise ValueError(f"{session[kind].name}: missing {sorted(missing)}")
                    tasks.append((name, subject, sessions))
                except (OSError, ValueError) as exc:
                    errors.append(f"{subject}: {exc}")
                    row["n_subjects_failed"] += 1
            if subjects:
                row["status"] = "pending" if row["n_subjects_failed"] < len(subjects) else "unsupported"
                row["message"] = "; ".join(errors)
        rows.append(row)
    return tasks, pd.DataFrame(rows), ignored


def _trial_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Discard per-frame payload in the worker; retain all trial diagnostics.

    Trial constants are selected from one row, not groupby.first(), which can
    synthesize a row by taking different nonmissing values from different frames.
    """
    if frame.empty:
        return frame.copy()
    columns = [c for c in frame.columns if c in _KEYS or c in {
        "session_id", "global_trial_id", "type", "target_ttc_ms", "lv_ratio_ms", "init_half_angle_deg",
        "wind_dir", "screen_side", "direction", "side", "response_type", "response_group",
        "v_max", "latency_ms", "escape_interval_ms", "interval_onset_ms", "interval_offset_ms",
        "reaction_time_ms", "escape_onset_ms", "escape_reaction_time_ms", "distance_mm", "distance_500ms_mm",
        "short_rt", "stop_to_escape_interval_ms", "stillness_presence", "pooled_vmax_pass",
    } or c.startswith(("pause_", "stillness_", "prestim_"))]
    return frame.drop_duplicates(_KEYS).loc[:, columns].reset_index(drop=True).copy()


def _process_ms_subject(task: tuple) -> pd.DataFrame:
    return _trial_table(_process_subject(task))


def aggregate_ms_trials(input_dir: Path | str, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the established pipeline once per animal, returning only trial rows."""
    cfg = _settings()
    n_workers = int(cfg.workers) if workers is None else workers
    if n_workers < 1:
        raise ValueError("workers must be at least 1")
    tasks, coverage, ignored = discover_ms_inputs(input_dir)
    n_workers = min(n_workers, max(1, len(tasks)))
    if n_workers == 1:
        parts = [_process_ms_subject(task) for task in tasks]
    else:
        with Pool(n_workers) as pool:
            parts = pool.map(_process_ms_subject, tasks)
    failed = [{"paradigm": task[0], "subject_id": task[1]} for task, part in zip(tasks, parts) if part.empty]
    parts = [part for part in parts if not part.empty]
    trials = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_CONDITIONS + [
        "global_trial_index", "response_type", "v_max", "escape_reaction_time_ms", "distance_mm", "prestim_status"])
    if trials.duplicated(_KEYS).any():
        raise ValueError("Duplicate trial keys after MS aggregation")
    if trials.empty:
        threshold, method = None, "unavailable: no processed trials"
    else:
        threshold, method, _ = select_vmax_threshold(pd.to_numeric(trials.v_max, errors="coerce").to_numpy())
    # This is a pooled Vmax diagnostic, not another response/class eligibility gate.
    trials["pooled_vmax_pass"] = trials.v_max >= threshold if threshold is not None else pd.Series(dtype=bool)
    for index, row in coverage.iterrows():
        local = trials[trials.paradigm == row.paradigm]
        n_failed = row.n_subjects_failed + sum(f["paradigm"] == row.paradigm for f in failed)
        coverage.loc[index, ["n_subjects", "n_subjects_failed", "n_trials"]] = [local.subject_id.nunique(), n_failed, len(local)]
        if not local.empty:
            coverage.loc[index, "status"] = "partial" if n_failed else "available"
        elif row.status == "pending":
            coverage.loc[index, "status"] = "failed"
        if n_failed and row.status == "pending":
            coverage.loc[index, "message"] = (str(row.message) + "; " if row.message else "") + f"{n_failed} subject(s) unsupported or failed"
    meta = dict(input=str(Path(input_dir).resolve()), workers=n_workers,
                merge_prewalk=read_merge_prewalk(), vmax_threshold=threshold, vmax_method=method,
                status="available" if not trials.empty else "no_data",
                n_trials=len(trials), n_subjects=int(trials[_KEYS[:2]].drop_duplicates().shape[0]),
                ignored_directories=ignored, failed_subjects=failed,
                uncertainty_unit="independent animals within each paradigm",
                probability_endpoint="overview: derived Escape group; enhancement: raw any_burst including PreEscape",
                timing_endpoint="raw class selects endpoint before grouping; visual TTC-relative, wind calibrated-reference-relative",
                config=get_config().to_dict())
    return trials, coverage, meta


def _modality(types: pd.Series) -> pd.Series:
    names = types.fillna("").astype(str).str.lower()
    result = pd.Series("unknown", index=types.index)
    result.loc[names.str.startswith("baseline_visual") | names.isin(["visual_only", "visual", "looming"])] = "visual"
    result.loc[names.str.startswith("baseline_wind") | names.isin(["wind_only", "wind"])] = "wind"
    result.loc[names.str.startswith("looming_wind")] = "multisensory"
    return result


def summarize_ms_trials(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export classifier N separately from observed/missing timing and baseline."""
    trials = _trial_table(frame)
    for column in _CONDITIONS:
        if column not in trials:
            trials[column] = np.nan
    trials["selected_rt_ms"] = select_escape_latency(trials)
    trials["response_group"] = response_group_series(trials)
    trials["modality"] = _modality(trials.type)
    trials["escape_group_member"] = trials.response_group.eq("Escape")
    trials["burst_member"] = trials.response_type.isin(BURST_CLASSES)
    trials["selected_distance_mm"] = pd.to_numeric(trials.distance_mm, errors="coerce").where(trials.burst_member)
    trials["rt_observed"] = trials.burst_member & trials.selected_rt_ms.notna()
    trials["prestim_observed"] = trials.get("prestim_status", pd.Series("unobserved", index=trials.index)).isin(
        ["stationary", "intermittent_moving", "continuous_moving"])
    subjects = trials.groupby(_CONDITIONS, dropna=False, sort=False).agg(
        modality=("modality", "first"), n_trials=("global_trial_index", "size"),
        escape_probability=("escape_group_member", "mean"), burst_probability=("burst_member", "mean"),
        rt_trials=("burst_member", "sum"), rt_observed=("rt_observed", "sum"),
        rt_median_ms=("selected_rt_ms", "median"), distance_median_mm=("selected_distance_mm", "median"),
        prestim_observed=("prestim_observed", "sum"),
    ).reset_index()
    subjects["rt_missing"] = subjects.rt_trials - subjects.rt_observed
    counts = trials.groupby(_CONDITIONS + ["response_type", "response_group"], dropna=False, sort=False).size().rename("n_trials").reset_index()
    totals = trials.groupby(_CONDITIONS, dropna=False, sort=False).size().rename("n_all_trials").reset_index()
    responses = counts.merge(totals, on=_CONDITIONS, validate="many_to_one")
    responses["probability"] = responses.n_trials / responses.n_all_trials
    return trials, subjects, responses


def _matching_visual(subjects: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    """Do not compare MS with a visual control at a different looming setting."""
    visual = subjects[subjects.modality == "visual"]
    value = condition["lv_ratio_ms"]
    if pd.isna(value):
        return visual.iloc[:0]
    visual = visual[pd.to_numeric(visual.lv_ratio_ms, errors="coerce") == float(value)]
    angle = condition["init_half_angle_deg"]
    # Current source files do not record the initial visual angle. Match only
    # equally unrecorded controls, and expose this limitation in the comparison.
    angles = pd.to_numeric(visual.init_half_angle_deg, errors="coerce")
    return visual[angles.isna() if pd.isna(angle) else angles == float(angle)]


def compare_ms_probabilities(subjects: pd.DataFrame) -> pd.DataFrame:
    """Descriptive MS enhancement with independent, equal-animal bootstrap CIs.

    A common any-burst endpoint (including PreEscape) is used for V, W and MS:
    excluding MS visual-triggered PreEscape while counting visual-control bursts
    would compare different outcomes. P(V or W)=P(V)+P(W)-P(V)P(W) is a
    probability independence reference, not a reaction-time race-model bound or
    a mechanistic test of neural integration.
    """
    cfg = _settings()
    iterations, minimum = int(cfg.bootstrap_iterations), int(cfg.min_subjects)
    confidence = float(cfg.confidence_level)
    if iterations < 1 or minimum < 2 or not 0 < confidence < 1:
        raise ValueError("MS bootstrap requires iterations >= 1, min_subjects >= 2, and 0 < confidence_level < 1")
    rng = np.random.default_rng(int(cfg.random_seed))
    columns = ["paradigm", "type", "target_ttc_ms", "lv_ratio_ms", "init_half_angle_deg"]
    rows = []
    for key, local in subjects[subjects.modality == "multisensory"].groupby(columns, dropna=False, sort=False):
        condition = pd.Series(dict(zip(columns, key)))
        visual = _matching_visual(subjects, condition)
        wind = subjects[subjects.modality == "wind"]
        row = dict(condition, n_subjects=local.subject_id.nunique(), n_visual_subjects=visual[_KEYS[:2]].drop_duplicates().shape[0],
                   n_wind_subjects=wind[_KEYS[:2]].drop_duplicates().shape[0], status="missing_baseline",
                   probability_endpoint="any_burst", visual_match_status=("lv_ratio_and_angle" if pd.notna(condition.init_half_angle_deg) else "lv_ratio_only_angle_unrecorded"))
        for name in ("ms_probability", "visual_probability", "wind_probability", "independence_probability",
                     "delta_best", "delta_best_ci_low", "delta_best_ci_high", "delta_independence",
                     "delta_independence_ci_low", "delta_independence_ci_high"):
            row[name] = np.nan
        # If conditions repeat for one animal, combine its trials before assigning
        # one animal weight; do not weight its multiple sessions as independent.
        values = []
        for group in (local, visual, wind):
            numerator = group.burst_probability * group.n_trials
            grouped = group.assign(_successes=numerator).groupby(_KEYS[:2], sort=True).agg(
                successes=("_successes", "sum"), trials=("n_trials", "sum"))
            values.append((grouped.successes / grouped.trials).dropna().to_numpy())
        row["ms_probability"] = float(np.mean(values[0])) if len(values[0]) else np.nan
        if any(len(v) == 0 for v in values[1:]):
            rows.append(row)
            continue
        m, v, w = (float(np.mean(value)) for value in values)
        independent = v + w - v * w
        row.update(ms_probability=m, visual_probability=v, wind_probability=w,
                   independence_probability=independent, delta_best=m - max(v, w), delta_independence=m - independent,
                   status="insufficient_subjects")
        if all(len(value) >= minimum for value in values):
            draws = [rng.choice(value, size=(iterations, len(value)), replace=True).mean(axis=1) for value in values]
            deltas = (draws[0] - np.maximum(draws[1], draws[2]), draws[0] - (draws[1] + draws[2] - draws[1] * draws[2]))
            alpha = (1 - confidence) / 2
            for name, delta in zip(("delta_best", "delta_independence"), deltas):
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = np.quantile(delta, [alpha, 1 - alpha])
            row["status"] = "available"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns + ["n_subjects", "n_visual_subjects", "n_wind_subjects",
        "ms_probability", "visual_probability", "wind_probability", "independence_probability",
        "delta_best", "delta_best_ci_low", "delta_best_ci_high", "delta_independence", "delta_independence_ci_low", "delta_independence_ci_high", "probability_endpoint", "visual_match_status", "status"])


def run_ms_analysis(input_dir: Path | str, output_dir: Path | str, workers: int | None = None) -> dict:
    """CLI seam: one pass through raw data, then tables and comparison figures."""
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("MS output must be outside the input tree")
    trials, coverage, meta = aggregate_ms_trials(root, workers=workers)
    trials, subjects, responses = summarize_ms_trials(trials)
    comparisons = compare_ms_probabilities(subjects)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"ms_summary.csv": trials, "ms_subject_summary.csv": subjects,
              "ms_response_probabilities.csv": responses, "ms_coverage.csv": coverage,
              "ms_probability_enhancement.csv": comparisons}
    if "prestim_status" in trials:
        tables["ms_prestim_outcomes.csv"] = trials.groupby(
            ["paradigm", "subject_id", "prestim_status", "response_type", "response_group"],
            dropna=False, sort=False).size().rename("n_trials").reset_index()
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    meta["coverage"] = coverage.to_dict("records")
    meta["comparison_status"] = json.loads(comparisons[["paradigm", "type", "target_ttc_ms", "status"]].to_json(orient="records"))
    meta["unknown_type_trials"] = int(trials.modality.eq("unknown").sum())
    meta["raw_response_types"] = list(RESPONSE_TYPES)
    (output / "ms_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cercus.visualization.multisensory import plot_ms_enhancement, plot_ms_overview
    for filename, plot in (("ms_overview.svg", lambda: plot_ms_overview(subjects, coverage)),
                           ("ms_probability_enhancement.svg", lambda: plot_ms_enhancement(comparisons))):
        figure = plot()
        path = output / filename
        if path.exists():
            path.unlink()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
    log.info("MS mode: %d trials, %d animals; coverage → %s", len(trials), meta["n_subjects"], output)
    return meta

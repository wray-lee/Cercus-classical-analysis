#!/usr/bin/env python3
"""Calibrate the stimulus-angle offset of a ring-airflow cricket escape apparatus.

All trials share ONE global angular offset delta (a single rotation of the
stimulus ring): stim_true = stim_nominal + delta. Valid escapes are defined by
the cercus-cli pipeline (the same convention as main.py's trajectory plotting):
preprocess -> ternary classification -> escape-interval arena trajectory;
response_type in {Escape, PreWalk}; response angle = atan2(traj_x[-1],
traj_y[-1]) over the escape interval (angle from +y). delta is estimated from
ALL input data as the single rotation that makes the pooled left/right mean
errors as symmetric as possible about the target (-expected_error, default
-18 deg): delta = (mean_err_left + mean_err_right)/2 + expected_error. Writes
corrected copies of every events/kinematics pair under --output, mirroring the
input folder structure, plus calibration_report.json/.png. Original files are
never modified.

The correction keeps the file format identical to the originals: events are
copied byte-for-byte, and the correction is baked into the kinematics by
rotating every body-frame (dx, dy) by -delta, which rotates the arena
trajectory (escape direction) by -delta — equivalent to correcting the
stimulus angle. delta is also recorded in calibration_report.json.

Escape-direction frame is atan2(x, y) (0 = +y forward, +90 = +x right), so the
nominal defaults are left = 270 (left arena edge) / right = 90 (right edge),
matching draw_side_arrows in the trajectory plots. The two nozzles are 180 deg
apart; crickets' left/right escapes need not be — delta just centres them.

The per-group regression (response ~ nominal) is a diagnostic only: its implied
delta (intercept+1)/slope equals delta_est only when the data follows the
regression prior (slope ~ 0.9, intercept ~ -1). It is not a pass/fail gate.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Make the cercus-cli package importable when run as `python tools/calibrate_offset.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ══════════════════════════════════════════════════════════════════════════
# angle helpers
# ══════════════════════════════════════════════════════════════════════════

def wrap180(a):
    return ((np.asarray(a, dtype=float) + 180.0) % 360.0) - 180.0


def circ_mean(deg):
    d = np.asarray(deg, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(d))))))


def delta_from_trials(trials, expected_error):
    if len(trials) < 2:
        return float("nan")
    err = wrap180([t["response"] - (t["nominal"] + 180.0) for t in trials])
    return circ_mean(err) + expected_error


def global_delta(group_trials, expected_error):
    """One device offset from ALL groups: rotate the nominal stimulus angles so
    the pooled left and right mean errors sit as symmetric as possible about the
    target (-expected_error). delta = (mean_err_left + mean_err_right)/2 +
    expected_error — the least-squares centering of both sides at the target.
    Uses all input data (every group, every session)."""
    e_by_side = {"left": [], "right": []}
    for tr in group_trials:
        for t in tr:
            e_by_side[t["side"]].append(t["response"] - (t["nominal"] + 180.0))
    means = {s: circ_mean(wrap180(e_by_side[s])) for s in e_by_side if e_by_side[s]}
    if not means:
        return float("nan")
    if len(means) == 1:
        return next(iter(means.values())) + expected_error
    return (means["left"] + means["right"]) / 2.0 + expected_error


# ══════════════════════════════════════════════════════════════════════════
# scanning & nominal-angle resolution
# ══════════════════════════════════════════════════════════════════════════

SESSION_RE = re.compile(r"(.+?)_session_(\d+)_(events|kinematics)\.csv$")
ANGLE_KEYS = ("stim_angle", "wind_angle", "wind_direction",
              "stim_azimuth", "wind_azimuth", "angle")


def _side_of(det):
    for k in ("wind_dir", "screen_side"):
        v = det.get(k)
        if v is not None and str(v).strip().lower() in ("left", "right"):
            return str(v).strip().lower()
    return None


def nominal_angle(det, left_angle, right_angle):
    """(nominal_deg|nan, side|None). Prefer an explicit numeric angle key."""
    for k in ANGLE_KEYS:
        if k in det:
            v = str(det[k]).strip()
            if v.lower() in ("left", "right"):
                return (left_angle, "left") if v.lower() == "left" else (right_angle, "right")
            try:
                return float(v), None
            except ValueError:
                pass
    side = _side_of(det)
    if side is None:
        return float("nan"), None
    return (left_angle, "left") if side == "left" else (right_angle, "right")


def _sesskey(ev_path):
    m = SESSION_RE.match(os.path.basename(ev_path))
    return (m.group(1), int(m.group(2)))


def _pair_sessions(fdir):
    evs = {}
    for f in os.listdir(fdir):
        m = SESSION_RE.match(f)
        if m and m.group(3) == "events":
            evs[(m.group(1), int(m.group(2)))] = os.path.join(fdir, f)
    pairs = []
    for f in os.listdir(fdir):
        m = SESSION_RE.match(f)
        if m and m.group(3) == "kinematics":
            k = (m.group(1), int(m.group(2)))
            if k in evs:
                pairs.append((evs[k], os.path.join(fdir, f)))
    pairs.sort(key=lambda p: _sesskey(p[0]))
    return pairs


EXCLUDED_DIR_NAMES = {"garbage", "pilot", "__pycache__", "trials", "figures", "results", "output"}


def scan(input_dir, group_by="folder"):
    """Scan *input_dir* (both flat CSVs and nested subdirectories).

    Returns
    -------
    groups : dict[str, list[tuple[str, str]]]
        ``{group_key: [(events_path, kinematics_path), ...]}``
    """
    groups = {}

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIR_NAMES and not d.startswith((".", "_"))]

        evs = {}
        for f in files:
            m = SESSION_RE.match(f)
            if m and m.group(3) == "events":
                evs[(m.group(1), int(m.group(2)))] = os.path.join(root, f)

        pairs = []
        for f in files:
            m = SESSION_RE.match(f)
            if m and m.group(3) == "kinematics":
                k = (m.group(1), int(m.group(2)))
                if k in evs:
                    pairs.append((evs[k], os.path.join(root, f)))

        if not pairs:
            continue

        pairs.sort(key=lambda p: _sesskey(p[0]))
        folder_rel = os.path.relpath(root, input_dir)

        for p in pairs:
            subj = SESSION_RE.match(os.path.basename(p[0])).group(1)
            key = subj if (group_by == "subject" or folder_rel in (".", "")) else folder_rel
            groups.setdefault(key, []).append(p)

    return groups

# ══════════════════════════════════════════════════════════════════════════
# per-trial loading, estimation, stats
# ══════════════════════════════════════════════════════════════════════════

def _remap_t2g(t2g, sid, raw):
    """Map a raw global_trial_id to the concatenated index used by preprocess."""
    for cand in (int(raw), str(raw)):
        try:
            if (sid, cand) in t2g:
                return t2g[(sid, cand)]
        except (TypeError, ValueError):
            pass
    return raw


def load_group_escapes(sessions, left_angle, right_angle, min_disp):
    """Valid escapes via the cercus-cli pipeline (the convention used by main.py's
    trajectory plotting / trial_escape_angles): preprocess -> ternary
    classification -> escape-interval arena trajectory. Keeps response_type in
    {Escape, PreWalk}; response angle = atan2(traj_x[-1], traj_y[-1]) over the
    escape interval (angle from +y). A pre-existing numeric response_angle in the
    trial's details JSON is used verbatim when present."""
    from pipeline.io import load_and_concat_sessions
    from pipeline.kinematics import preprocess
    from pipeline.classifier import label_trials
    from cercus.visualization._core import compute_trajectory_masks

    meta, windows, anchors, kin, t2g = load_and_concat_sessions(sessions)
    df = preprocess(meta, windows, anchors, kin)
    df["global_trial_id"] = df["global_trial_id"].astype(str)
    df = label_trials(df)

    det_by_gid = {}
    for sess in sessions:
        sid = sess["session_id"]
        ev = pd.read_csv(sess["events"], encoding="utf-8-sig")
        for _, row in ev[ev["event_name"] == "trial_start"].iterrows():
            try:
                d = json.loads(row["details"])
            except Exception:
                d = {}
            det_by_gid[str(_remap_t2g(t2g, sid, row["global_trial_id"]))] = (
                d if isinstance(d, dict) else {}
            )

    trials = []
    for tid, grp in df.groupby("global_trial_id"):
        if grp["response_type"].iloc[0] not in ("Escape", "PreWalk"):
            continue
        det = det_by_gid.get(str(tid), {})
        nominal, side = nominal_angle(det, left_angle, right_angle)
        if math.isnan(nominal):
            continue
        resp = None
        ra = det.get("response_angle")
        if ra is not None:
            try:
                resp = float(ra)
            except (TypeError, ValueError):
                resp = None
        if resp is None:
            on, off = grp["interval_onset_ms"].iloc[0], grp["interval_offset_ms"].iloc[0]
            if pd.isna(on) or pd.isna(off):
                continue
            res = compute_trajectory_masks(grp, on, off)
            if res is None:
                continue
            tx, ty = res[0], res[1]
            if tx is None or len(tx) < 2:
                continue
            if math.hypot(tx[-1], ty[-1]) < min_disp:   # near-zero escape vector
                continue
            resp = math.degrees(math.atan2(tx[-1], ty[-1]))
        if side is None:
            side = ("left" if abs(wrap180(nominal - left_angle)) <
                    abs(wrap180(nominal - right_angle)) else "right")
        trials.append({"nominal": nominal, "response": resp, "side": side})
    return trials


def group_stats(trials, expected_error):
    st = {"n_trials": len(trials)}
    if len(trials) >= 2:
        err = wrap180([t["response"] - (t["nominal"] + 180.0) for t in trials])
        st["circ_mean_error_deg"] = circ_mean(err)
        st["delta_est_deg"] = st["circ_mean_error_deg"] + expected_error
        x = np.array([t["nominal"] for t in trials])
        y = np.array([t["response"] for t in trials])
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            st["slope"], st["intercept"] = float(slope), float(intercept)
            st["r"] = float(np.corrcoef(x, y)[0, 1])
        else:
            st["slope"] = st["intercept"] = st["r"] = float("nan")
        if not math.isnan(st["slope"]) and st["slope"] != 0.0:
            st["regression_implied_delta_deg"] = (st["intercept"] + 1.0) / st["slope"]
        else:
            st["regression_implied_delta_deg"] = float("nan")
    else:
        for k in ("circ_mean_error_deg", "delta_est_deg", "slope",
                  "intercept", "r", "regression_implied_delta_deg"):
            st[k] = float("nan")
    return st


def side_stats(trials, delta):
    out = {}
    for side in ("left", "right"):
        st = [t for t in trials if t["side"] == side]
        if not st:
            continue
        eb = wrap180([t["response"] - (t["nominal"] + 180.0) for t in st])
        mb = circ_mean(eb)
        ea = wrap180(eb - delta) if not math.isnan(delta) else eb
        ma = circ_mean(ea)
        out[side] = {"n": len(st),
                     "mean_error_before_deg": mb,
                     "mean_error_after_deg": ma,
                     "within_5deg_before": abs(mb + 18.0) <= 5.0,
                     "within_5deg_after": abs(ma + 18.0) <= 5.0}
    return out


def loo_delta_std(group_trials, expected_error):
    """Leave-one-cricket-out stability of the GLOBAL device offset -> std."""
    vals = []
    for i in range(len(group_trials)):
        # ponytail: O(g^2) pooling, fine for a handful of groups
        rest = [group_trials[j] for j in range(len(group_trials)) if j != i]
        d = global_delta(rest, expected_error)
        if not math.isnan(d):
            vals.append(d)
    return float(np.std(vals)) if len(vals) >= 2 else float("nan")

# ══════════════════════════════════════════════════════════════════════════
# correction writer
# ══════════════════════════════════════════════════════════════════════════

def write_corrected(input_dir, output_dir, groups, delta,
                    left_angle, right_angle):
    import shutil
    for key, sess in groups.items():
        for ev_path, kin_path in sess:
            rel_folder = os.path.relpath(os.path.dirname(ev_path), input_dir)
            out_dir = os.path.normpath(os.path.join(output_dir, rel_folder))
            os.makedirs(out_dir, exist_ok=True)
            # events stay byte-identical to the original (same format, no added fields).
            shutil.copy(ev_path, os.path.join(out_dir, os.path.basename(ev_path)))
            # Bake delta into the kinematics: rotate every body-frame (dx, dy) by -delta
            # so the arena trajectory (and escape direction) rotates by -delta. This makes
            # escape-relative-to-stimulus errors land at the target while keeping the file
            # format identical to the original (same columns, changed values).
            kin = pd.read_csv(kin_path, encoding="utf-8-sig")
            if not math.isnan(delta):
                # In the atan2(x, y) escape frame a CCW body rotation of +delta
                # decreases the escape angle by delta, i.e. escape' = escape - delta.
                rot = math.radians(delta)
                c, s = math.cos(rot), math.sin(rot)
                dx = kin["dx"].to_numpy(dtype=float)
                dy = kin["dy"].to_numpy(dtype=float)
                kin["dx"] = dx * c - dy * s
                kin["dy"] = dx * s + dy * c
            kin.to_csv(os.path.join(out_dir, os.path.basename(kin_path)),
                       index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════════════════════
# report & plot
# ══════════════════════════════════════════════════════════════════════════

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _safe_savefig(fig, path):
    try:
        fig.savefig(path)
    except OSError:   # Windows OSError 22 dodge (unlink-then-save)
        if os.path.exists(path):
            os.unlink(path)
        fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_plot(all_valid, delta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    before, after = [], []
    for gkey, t in all_valid:
        e = float(wrap180(t["response"] - (t["nominal"] + 180.0)))
        before.append(e)
        after.append(float(wrap180(e - delta)) if not math.isnan(delta) else np.nan)
    before = np.asarray(before)
    after = np.asarray(after)
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2)
    cb = np.zeros(0)
    ca = np.zeros(0)
    if before.size:
        cb = ax1.hist(np.mod(np.radians(before), 2 * np.pi), bins=36,
                      range=(0, 2 * np.pi), alpha=0.5,
                      color="tab:red", label="before")[0]
    if after.size:
        ca = ax1.hist(np.mod(np.radians(after), 2 * np.pi), bins=36,
                      range=(0, 2 * np.pi), alpha=0.5,
                      color="tab:blue", label="after")[0]
    rmax = max(1.0, float(cb.max()) if cb.size else 0.0,
               float(ca.max()) if ca.size else 0.0)
    ax1.set_rmax(rmax * 1.15)
    for th in (0.0, math.radians(-18.0) % (2 * np.pi)):
        ax1.plot([th, th], [0.0, rmax], "k--", lw=1)
    ax1.set_title("error rose (before vs after)")
    ax1.legend(fontsize=8, loc="upper right")

    ax2.hist(before, bins=36, range=(-180, 180), alpha=0.5,
             color="tab:red", label="before")
    ax2.hist(after, bins=36, range=(-180, 180), alpha=0.5,
             color="tab:blue", label="after")
    ax2.axvspan(-23.0, -13.0, color="gray", alpha=0.25)   # -18 ± 5 band
    ax2.axvline(-18.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("error (deg)")
    ax2.set_ylabel("count")
    ax2.set_title("error histogram (-18±5 band)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    _safe_savefig(fig, out_path)

# ══════════════════════════════════════════════════════════════════════════
# synthetic self-test (the runnable correctness check, no test framework)
# ══════════════════════════════════════════════════════════════════════════

def _synth_session(fdir, sess, trials, delta, rng):
    """Write a synthetic session pair the cercus escape pipeline can classify
    (strong wind burst -> Escape) and where response_angle is planted in every
    trial's details so the calibration recovers the planted delta exactly."""
    os.makedirs(fdir, exist_ok=True)
    ev_rows, kin_rows = [], []
    for ti, (nominal, side) in enumerate(trials):
        # 162 = 180 - expected_error(18): cancels so delta_est recovers `delta`
        resp = float(wrap180(nominal + delta + 162.0))
        det = {"type": "baseline_wind", "target_ttc_ms": None, "lv_ratio_ms": None,
               "wind_dir": side, "screen_side": side, "response_angle": resp}
        ts = ti * 1.0
        gid = str(ti)
        ev_rows.append({"event_name": "trial_start", "timestamp": ts, "session_num": sess,
                        "trial_in_session": ti + 1, "global_trial_id": gid,
                        "details": json.dumps(det)})
        ev_rows.append({"event_name": "trial_stop", "timestamp": ts + 0.5, "session_num": sess,
                        "trial_in_session": ti + 1, "global_trial_id": gid, "details": ""})
        c, s = math.cos(math.radians(resp)), math.sin(math.radians(resp))
        for k in range(5):                                # quiescent baseline
            kin_rows.append({"sys_time": ts + k * 0.005, "ard_time": 0, "dx": 0.0, "dy": 0.0,
                             "dz": 0.0, "stim_state": 0, "global_trial_id": gid})
        for k in range(40):                               # wind burst, ~400 mm/s -> Escape
            kin_rows.append({"sys_time": ts + 0.025 + k * 0.005, "ard_time": 0,
                             "dx": c * 2.0, "dy": s * 2.0, "dz": 0.0,
                             "stim_state": 1, "global_trial_id": gid})
        for k in range(10):                               # post-stimulus tail
            kin_rows.append({"sys_time": ts + 0.225 + k * 0.005, "ard_time": 0, "dx": 0.0,
                             "dy": 0.0, "dz": 0.0, "stim_state": 0, "global_trial_id": gid})
    subj = "cricket_" + os.path.basename(fdir).replace(".", "")
    pd.DataFrame(ev_rows).to_csv(os.path.join(fdir, f"{subj}_session_{sess}_events.csv"),
                                 index=False, encoding="utf-8-sig")
    pd.DataFrame(kin_rows).to_csv(os.path.join(fdir, f"{subj}_session_{sess}_kinematics.csv"),
                                  index=False, encoding="utf-8-sig")


def _dir_hashes(root):
    return {os.path.relpath(os.path.join(r, f), root):
            hashlib.sha256(open(os.path.join(r, f), "rb").read()).hexdigest()
            for r, _, fs in os.walk(root) for f in fs}


def selftest():
    print("selftest: building synthetic dataset ...")
    planted = 7.0
    rng = np.random.default_rng(0)
    trials = [(270.0, "left"), (90.0, "right")] * 6
    n_expected = len(trials) * 2 * 3   # 3 groups x 2 sessions x 12 trials
    with tempfile.TemporaryDirectory() as tmp:
        in_dir, out_dir = os.path.join(tmp, "cali"), os.path.join(tmp, "out")
        for gname in ("d1", "d2", "d3"):
            for sess in (1, 2):
                _synth_session(os.path.join(in_dir, gname), sess, trials, planted, rng)
        input_hashes = _dir_hashes(in_dir)
        groups = scan(in_dir, "folder")
        group_trials = {k: [] for k in groups}
        for key, pairs in groups.items():
            sessions = [
                {"session_id": int(SESSION_RE.match(os.path.basename(ev)).group(2)),
                 "events": Path(ev), "kinematics": Path(kin)}
                for ev, kin in pairs
            ]
            group_trials[key] += load_group_escapes(sessions, 270.0, 90.0, 1.0)
        gdelta = global_delta([group_trials[k] for k in sorted(groups)], 18.0)
        loo = loo_delta_std([group_trials[k] for k in sorted(groups)], 18.0)
        write_corrected(in_dir, out_dir, groups, gdelta, 270.0, 90.0)

        ok_delta = abs(gdelta - planted) <= 3.0
        ok_loo = math.isfinite(loo)
        ok_unchanged = _dir_hashes(in_dir) == input_hashes   # originals never modified
        n_used = sum(len(v) for v in group_trials.values())
        ok_n = n_used == n_expected                          # all classified Escape + used
        kin_files = [os.path.join(r, f) for r, _, fs in os.walk(out_dir)
                     for f in fs if f.endswith("_kinematics.csv")]
        ev_files = [os.path.join(r, f) for r, _, fs in os.walk(out_dir)
                    for f in fs if f.endswith("_events.csv")]
        ok_mirror = (len(kin_files) == len(ev_files) == 6 and
                     {os.path.basename(os.path.dirname(p)) for p in kin_files} == set(groups))
        ok_events_same = all(   # events stay byte-identical (same format, no added fields)
            hashlib.sha256(open(p, "rb").read()).hexdigest()
            == input_hashes[os.path.relpath(p, out_dir)]
            for p in ev_files
        )
        # kinematics (dx, dy) must be the original rotated by -delta
        ok_rot = True
        for kf in kin_files:
            rel = os.path.relpath(kf, out_dir)
            orig = pd.read_csv(os.path.join(in_dir, rel), encoding="utf-8-sig")
            corr = pd.read_csv(kf, encoding="utf-8-sig")
            r = math.radians(gdelta)
            c, s = math.cos(r), math.sin(r)
            for i in range(0, len(orig), max(1, len(orig) // 50)):
                dx, dy = float(orig["dx"].iloc[i]), float(orig["dy"].iloc[i])
                if math.isnan(dx) or math.isnan(dy):
                    continue
                ex, ey = dx * c - dy * s, dx * s + dy * c
                if not (math.isclose(float(corr["dx"].iloc[i]), ex, abs_tol=1e-6) and
                        math.isclose(float(corr["dy"].iloc[i]), ey, abs_tol=1e-6)):
                    ok_rot = False
                    break
            if not ok_rot:
                break
        print(f"  planted delta={planted}; recovered_global={round(gdelta, 2)}; "
              f"trials_used={n_used}/{n_expected}")
        print(f"  loo_delta_std={loo:.3f}; originals_unchanged={ok_unchanged}; "
              f"mirror_ok={ok_mirror}; events_same_format={ok_events_same}; "
              f"kin_rotated_by_delta={ok_rot}; all_classified_escape={ok_n}")
        ok = all([ok_delta, ok_loo, ok_unchanged, ok_mirror, ok_events_same, ok_rot, ok_n])
        print("PASS" if ok else "FAIL")
        return ok

# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Calibrate ring-airflow stimulus angle offset.")
    p.add_argument("--input", default="data",
                   help="input data root directory containing per-subject session folders")
    p.add_argument("--output", default=None,
                   help="directory to save report files and corrected CSVs (optional; "
                        "when omitted, delta is estimated and printed without writing files)")
    p.add_argument("--groups", type=int, default=None,
                   help="expected group count (warn on mismatch, don't crash)")
    p.add_argument("--group-by", choices=("folder", "subject"), default="folder")
    p.add_argument("--left-angle", type=float, default=270.0,
                   help="software's nominal angle for the LEFT nozzle, in the escape "
                        "frame atan2(x,y) (0=+y forward, 90=+x right); left edge = 270")
    p.add_argument("--right-angle", type=float, default=90.0,
                   help="software's nominal angle for the RIGHT nozzle (right edge = 90)")
    p.add_argument("--min-disp-mm", type=float, default=1.0)
    p.add_argument("--expected-error-deg", type=float, default=18.0)
    p.add_argument("--estimate-only", action="store_true", default=False,
                   help="only estimate delta and write report (do NOT duplicate/write corrected CSV files)")
    p.add_argument("--plot", dest="plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--selftest", action="store_true",
                   help="run synthetic correctness check and exit")
    return p.parse_args(argv)


def main(argv=None):
    import logging
    logging.basicConfig(level=logging.WARNING)   # silence pipeline INFO logs
    args = parse_args(argv)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not os.path.isdir(args.input):
        print(f"error: input dir not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    if args.output is not None and os.path.abspath(args.output) == os.path.abspath(args.input):
        print("error: --output must differ from --input", file=sys.stderr)
        sys.exit(2)

    groups = scan(args.input, args.group_by)
    if args.groups is not None and len(groups) != args.groups:
        print(f"warning: --groups {args.groups} but discovered {len(groups)} groups",
              file=sys.stderr)

    group_trials = {k: [] for k in groups}
    for key, pairs in groups.items():
        sessions = [
            {"session_id": int(SESSION_RE.match(os.path.basename(ev)).group(2)),
             "events": Path(ev), "kinematics": Path(kin)}
            for ev, kin in pairs
        ]
        group_trials[key] += load_group_escapes(sessions, args.left_angle,
                                                args.right_angle, args.min_disp_mm)

    gtrials_list = [group_trials[k] for k in sorted(groups)]
    gdelta = global_delta(gtrials_list, args.expected_error_deg)

    report_groups, all_valid = [], []
    for k in sorted(groups):
        st = group_stats(group_trials[k], args.expected_error_deg)
        st["group"] = k
        st["per_group_circ_delta_deg"] = st.pop("delta_est_deg")
        st["sides"] = side_stats(group_trials[k], gdelta)
        all_valid += [(k, t) for t in group_trials[k]]
        report_groups.append(st)

    loo_std = loo_delta_std(gtrials_list, args.expected_error_deg)

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        if not args.estimate_only:
            write_corrected(args.input, args.output, groups, gdelta,
                            args.left_angle, args.right_angle)

        pooled = [t for tr in gtrials_list for t in tr]
        report = {"n_groups": len(groups),
                  "delta_est_deg": gdelta,
                  "delta_method": ("single global device offset: (mean_error_left + "
                                   "mean_error_right)/2 + expected_error, pooled over ALL "
                                   "groups (all input data); makes left/right as symmetric "
                                   "as possible about the -expected_error target"),
                  "pooled_circ_delta_deg": delta_from_trials(pooled, args.expected_error_deg),
                  "loo_delta_std_deg": loo_std,
                  "nominal_mapping": {"left": args.left_angle, "right": args.right_angle},
                  "cross_check_note": ("per_group_circ_delta_deg is the per-group pooled "
                                       "circ-mean estimate (diagnostic; it spreads across "
                                       "groups when individual responses are noisy). "
                                       "regression_implied_delta_deg (=(intercept+1)/slope) "
                                       "equals it only under the regression prior "
                                       "(slope ~ 0.9, intercept ~ -1); diagnostic only."),
                  "parameters": vars(args), "groups": report_groups}
        with open(os.path.join(args.output, "calibration_report.json"), "w") as fh:
            json.dump(_clean(report), fh, indent=2)

        if args.plot:
            make_plot(all_valid, gdelta,
                      os.path.join(args.output, "calibration_report.png"))

    print("=" * 68)
    print("Airflow Angle Calibration Result:")
    print(f"  Analyzed {len(groups)} groups ({sum(len(v) for v in group_trials.values())} valid escape trials)")
    print(f"  Global delta estimate: {gdelta:+.1f}°  (LOO std: {loo_std:.2f}°)")
    print()
    print("To apply this offset in config.yaml, set:")
    print("  trajectory:")
    print(f"    wind_angle_offset_deg: {round(gdelta, 1):+g}")
    print("=" * 68)
    for st in report_groups:
        print(f"  {st['group']}: n={st['n_trials']:2d} "
              f"per_group_delta={st['per_group_circ_delta_deg']:+5.1f}° "
              f"circ_err={st['circ_mean_error_deg']:+5.1f}° "
              f"slope={st['slope']:.2f} r={st['r']:.2f}")
    if args.output is not None:
        print(f"Report: {os.path.join(args.output, 'calibration_report.json')}")
        if args.plot:
            print(f"Plot:   {os.path.join(args.output, 'calibration_report.png')}")


if __name__ == "__main__":
    main()

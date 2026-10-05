# Cercus Framework

Cricket escape-response behavioral analysis pipeline. Processes raw kinematics data, classifies trials into Escape / PreEscape / PreWalk / NoResponse, generates publication-grade figures (Nature/Science style), and performs Bayesian population-level inference via MCMC.

## Project Structure (Refactored)

```
Cercus-cli/
├── cercus/                          # NEW: Refactored package
│   ├── visualization/               # Publication-grade plotting
│   │   ├── __init__.py              # Re-exports all plot_* functions
│   │   ├── _core.py                 # Drawing helpers (grid, arrows, thresholds, oscilloscope)
│   │   ├── _circstats.py            # Circular statistics (Rayleigh, Watson-Williams)
│   │   ├── trajectories.py          # Trajectory overlay plots
│   │   ├── kinetics.py              # Speed & angular velocity kinetics plots
│   │   ├── heatmaps.py              # Density & trial-stacked heatmaps
│   │   ├── vmax.py                  # V_max distribution & GMM threshold plots
│   │   ├── behavior.py              # Behavior probability & habituation plots
│   │   └── polar.py                 # Polar direction histogram & angle distribution
│   ├── core/
│   │   └── kinematics/
│   │       ├── latency.py           # Escape latency backward-search & interval detection
│   │       └── trajectory_integration.py  # Dual-stage trajectory integration (Stage 1+2)
│   ├── config/
│   │   ├── __init__.py              # Unified config loader (YAML → SimpleNamespace)
│   │   ├── settings.py              # Pydantic TrajectoryConfig with cross-field validation
│   │   └── defaults/                # Default YAML configurations
│   │       ├── thresholds.yaml      # Escape/PreWalk detection thresholds
│   │       ├── geometry.yaml        # Arena geometry, timing, heatmap viz defaults
│   │       ├── colors.yaml          # Color palette (Lancet / NPG style)
│   │       ├── trajectory.yaml      # Trajectory presets (with all alternatives documented)
│   │       └── visualization.yaml   # Visualization style (bar labels, etc.)
│   ├── constants/
│   │   ├── colors.py                # Color constants (loaded from YAML)
│   │   ├── thresholds.py            # Physical thresholds (loaded from YAML)
│   │   └── geometry.py              # Geometry & timing constants (loaded from YAML)
│   └── cli/
│       └── app.py                   # Unified Typer CLI with commands: single, population, mcmc, ...
├── pipeline/                        # Legacy backward-compat wrappers
│   ├── io.py                        # CSV loading, session pairing, summary export
│   ├── kinematics.py                # Kinematics integration (ORIGINAL, stable)
│   ├── classifier.py                # Response-type classification (PreEscape switch)
│   ├── visualization.py             # ***DEPRECATED*** — re-exports from cercus.visualization
│   ├── constants.py                 # ***DEPRECATED*** — re-exports from cercus.constants + loader
│   └── mcmc.py                      # Bayesian hierachical models (PyMC / NumPyro)
├── main.py                          # Full single-subject pipeline
├── plot_trial_panels.py             # Per-trial composite panels
├── plot_all_trajectories_fixed.py   # Unified trajectory overlay
├── mcmc_analysis.py                 # MCMC psychophysics analysis
├── population_analysis.py           # Cross-subject batch: adaptive threshold + population viz
├── config.yaml                      # User overrides (merged with cercus/config/defaults/)
├── tests/
│   ├── test_visualization_regression.py  # Hash-based plot regression tests
│   └── test_classifier_golden.py         # 10 synthetic golden trials for classifier
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# NEW unified CLI (Typer-based)
python -m cercus.cli.app single --input path/to/data/ --output figures/
python -m cercus.cli.app population --input path/to/data/ --output results/
python -m cercus.cli.app mcmc --input path/to/data/ --output results/

# Equivalent old-style entry points
python main.py --input-dir path/to/data/ --output figures/
python plot_trial_panels.py --input-dir path/to/data/ --output figures/
python plot_all_trajectories_fixed.py --input-dir path/to/data/ --output trajectories.svg
python population_analysis.py --input-dir path/to/data/ --output results/
python mcmc_analysis.py --input-dir path/to/data/ --output results/
```

### CLI Commands

| Command | Description | Equivalent Legacy Script |
|---|---|---|
| `python -m cercus.cli.app single --input <dir> --output <dir>` | Single-subject analysis | `main.py` |
| `python -m cercus.cli.app population --input <dir> --output <dir> [--workers N]` | Population-level batch (parallel) | `population_analysis.py` |
| `python -m cercus.cli.app mcmc --input <dir> --output <dir>` | Bayesian MCMC analysis | `mcmc_analysis.py` |
| `python -m cercus.cli.app trial-panels --input <dir> --output <dir> [--workers N]` | Per-trial composite panels (parallel) | `plot_trial_panels.py` |
| `python -m cercus.cli.app trajectories --input <dir> --output <dir>` | Unified trajectory overlay | `plot_all_trajectories_fixed.py` |
| `python -m cercus.cli.app full --input <parent-dir> --output <dir> [--workers N]` | Cross-paradigm (full) mode: global V_max threshold + dumbbell figure | — |
| `python -m cercus.cli.app calibrate --input <dir>` | Estimate airflow stimulus angle offset | `tools/calibrate_offset.py` |

**Performance Note**: The `population` and `trial-panels` commands use multiprocessing to parallelize subject processing and visualization rendering. Use `--workers N` to control concurrency (default: all CPU cores). Single-threaded fallback: `--workers 1`.

**Full mode**: `--input` points at the *parent* directory whose subdirectories are the paradigms (e.g. `-373 30°`, `bv`). Each paradigm must contain paired `{subject}_session_{N}_{events,kinematics}.csv` files; empty/invalid directories are skipped with a warning. `Results/Test/train` (and `full/output/figures`) are excluded. The adaptive V_max threshold is computed **once on the pooled trials of all paradigms** and applied uniformly, keeping escape-probability comparisons across paradigms on a single scale. Outputs: `full_summary.csv` (trial-level, with `paradigm` + `is_valid_escape`), `full_meta.json` (threshold/method/paradigm order), and `paradigm_dumbbell.svg` — 3 panels (escape probability / stimulus-anchored RT / escape distance), where each dot is one animal's mean, the black square is the paradigm mean ± SD, and dots connect to the square by vertical lines (lab convention, cf. Frontiers fphys.2023.1153913 Fig 2). Paradigms are between-subject, so no cross-paradigm lines are drawn. Plus `paradigm_rt_dist.svg` — RT / escape distance compared **across paradigms × response class** (Escape / PreEscape / PreWalk boxes from per-subject medians, so no trial-level pseudoreplication; NoResponse excluded because RT/distance are undefined without an escape burst; Kruskal-Wallis across paradigms reported per class).

### Import Paths

| New Import (preferred) | Old Import (deprecated) |
|---|---|
| `from cercus.visualization import plot_trajectory_overlay` | `from pipeline.visualization import plot_trajectory_overlay` |
| `from cercus.visualization import plot_speed_kinetics` | `from pipeline.visualization import plot_speed_kinetics` |
| `from cercus.config.settings import TrajectoryConfig` | — |
| `from cercus.config import get_config, reload_config` | — |
| `from cercus.config import get_thresholds, get_geometry, get_colors` | — |
| `from cercus.core.kinematics.trajectory_integration import integrate_body_trajectory` | `from pipeline.visualization import _integrate_body_trajectory` |

The old `pipeline.visualization` import paths still work but emit a `DeprecationWarning`.

## Configuration System

All configurable parameters are stored as YAML files in `cercus/config/defaults/`:

| File | Contents |
|---|---|
| `thresholds.yaml` | Escape/PreWalk detection thresholds (`vmax_threshold`, `start_threshold`, `use_angular_onset_refinement`, etc.) |
| `geometry.yaml` | Arena geometry (`RADIUS_MM`), trajectory plot settings, speed window |
| `colors.yaml` | Color palette (escape, prewalk, left/right, NPG palette, etc.) |
| `trajectory.yaml` | Trajectory rendering presets (with all alternative presets documented) |
| `visualization.yaml` | Visualization style (`bar_label_style`, etc.) |
| `analysis.yaml` | Population rendering workers, adaptive thresholds, individual robustness checks |

### User Overrides

Create a `config.yaml` in the project root to override any default. Only include values you want to change:

```yaml
# Example: change escape detection threshold
escape:
  vmax_threshold: 98.0

# Example: switch trajectory preset
trajectory:
  use_escape_onset_heading: true
  dz_integration_range: "escape_interval"
```

For full trajectory preset documentation and alternatives, see `cercus/config/defaults/trajectory.yaml`.

### Python Access

```python
from cercus.config import get_config, reload_config
from cercus.constants.thresholds import ESCAPE_VMAX_THRESHOLD
from cercus.constants.geometry import RADIUS_MM
from cercus.constants.colors import COLOR_ESCAPE

# Force reload after external config changes
reload_config()
```

## Standardized Classification Criteria

Cercus classifies cricket behavioral responses into four mutually exclusive categories: **NoResponse**, **PreEscape**, **PreWalk**, and **Escape**. The pipeline uses a strict priority cascade to avoid ambiguous or overlapping labels:

$$\text{NoResponse} \longrightarrow \text{PreEscape} \longrightarrow \text{PreWalk} \longrightarrow \text{Escape}$$

```
                          ┌── [No qualifying burst] ──────────────────────────► NoResponse
                          │
[Response-window burst?]  ├── [Burst qualified ∧ onset < hardware wind trigger] ───► PreEscape
                          │
                          ├── [Burst qualified ∧ active walking before wind] ──► PreWalk
                          │
                          └── [Remaining qualified trials] ────────────────────► Escape
```

### 1. Burst Qualification Window

For any trial, let $v(t)$ denote walking speed, $V_b$ the burst threshold, and $t_w$ the airflow stimulus onset.
- **Airflow-containing trials**: The response interval is evaluated within $W = [t_w, t_w + 250\text{ ms}]$. A burst qualifies if $\max_{t \in W} v(t) > V_b$.
- **Visual-only baseline trials (`baseline_visual`)**: Evaluated from trial onset up to theoretical collision ($t \le \text{TTC}$).
- **Default thresholds**: The package default is $V_b = 98\text{ mm/s}$ (configurable via YAML; research configurations typically use $50\text{ mm/s}$).
- **Absolute veto**: If no speed sample in the window exceeds $V_b$, the trial is classified as **NoResponse**. Prior locomotion cannot override this veto.

### 2. PreEscape (Pre-Wind Escape)

When enabled (`classification.use_preescape: true`), wind trials with a qualifying burst are evaluated for early movement onset:
- **Burst anchor**: Find the first sample exceeding the burst threshold in $[t_w, t_w + 250\text{ ms}]$.
- **Onset search**: On the acquisition clock, search backward within the contiguous observed motion segment to the last causal 20-ms averaged speed $\le 10\text{ mm/s}$; the immediately following sample marks $t_e$. A low-speed observation separates a preceding burst; a missing observation or a record starting in motion cannot establish an onset. If a candidate has no observable transition, check later candidates in the same hardware-anchored burst window. Classification, wind escape RT and independent RTm share this onset. Acquisition tables leave unresolved timing missing; legacy tables without acquisition columns retain their descriptive timing basis.
- **Criterion**: If the observed onset precedes the hardware wind trigger ($t_e < t_w - \text{buffer}$, with buffer default $0\text{ ms}$), the trial is classified as **PreEscape**. Arrival calibration does not shift this comparison.
- **Significance**: In multimodal paradigms, these represent genuine escapes triggered early by the looming visual stimulus rather than the wind. In pure-wind trials, they reflect spontaneous acceleration.

### 3. PreWalk (Pre-Stimulus Locomotion)

For airflow trials, PreWalk identifies crickets already moving when the stimulus arrives:
- **1-Second History**: Evaluates the 1000 ms immediately preceding airflow arrival on the hardware acquisition clock (`ard_time`).
- **Locomotor Eligibility (`pause_moving_eligible`)**:
  1. The 1-second record must be complete and uncorrupted.
  2. Over this window, the duration-weighted occupancy of causal 20-ms displacement-averaged speeds above $10\text{ mm/s}$ must meet `prewalk.min_moving_fraction` (default 15%).
  3. The causal averaged speed at the stimulus reference itself must exceed $10\text{ mm/s}$.
- Trials meeting these conditions (and not flagged as PreEscape) are classified as **PreWalk**.

*Visual-only baseline note*: Evaluates the legacy window from 1000 ms to 50 ms prior to escape onset ($>15\%$ active frames).

### 4. Reaction Time (RT) Measurements

Cercus cleanly decouples **escape movement onset** from **stimulus-induced stopping**:

- **Escape Latency (`escape_reaction_time_ms`)**:
  Wind trials export the observed acquisition-clock `escape_onset_ms` and $t_e - t_a$, where $t_a$ is the calibrated airflow reference. Classification, escape RT, `short_rt` and the stopping cap share this source measurement; unresolved transitions stay missing.
  - The burst window and PreEscape comparison remain hardware-anchored. Arrival calibration shifts the motion-history and RT reference only.
  - PreEscape retains its measured lead time. These filtered endpoints are not validated physiological latencies.
  - `latency_ms` and `interval_onset_ms` remain descriptive centered-speed measurements for kinetics markers, distance/trajectory analysis and the existing MCMC survival view. Nonwind and legacy timing retains this descriptive basis.
- **Causal Stopping Latency T1 (`pause_stopping_time_ms`)**:
  The latency from airflow arrival to when walking speed first drops below the $10\text{ mm/s}$ stillness threshold. Includes optical-count resolution margins (requiring $>13.33\text{ mm/s}$ at onset and confirming a drop below $<6.67\text{ mm/s}$) to guard against sensor quantization noise.
- **Pause-to-Escape Interval T2 (`pause_to_escape_time_ms`)**:
  The duration from the confirmed pause endpoint to the subsequent escape burst.
- **Causal Motor RTm (`pause_reaction_time_ms`)**:
  With `thresholds.reaction_time.independent_rtm: true` (default), RTm is the observed acquisition-clock escape onset minus the calibrated wind reference; a stopping endpoint is optional. Pre-wind onsets supply PreEscape lead time, never negative RTm. Setting the switch to `false` retains the stop-dependent T1/T2 measurement: RTm requires both endpoints and equals $\text{T1} + \text{T2}$. The switch does not change classifier membership. Missing causal endpoints remain `NaN`, with no centered or synthetic imputation.

### 5. Population Summaries & Cohort Consistency

- **Single Authority**: Population and cross-paradigm summaries strictly follow the final classifier labels (`response_type`).
- **Trial Integrity**: Missing reaction-time endpoints do not alter cohort sizes; trials with unresolved RT remain in their respective response class, and counts are reported alongside coverage statistics.
- **Distance Integration**: Escape distance (`distance_mm`) retains its descriptive centered-speed interval (`interval_onset_ms` to `interval_offset_ms`). It is independent of causal RT availability; these interval timestamps may differ from the acquisition-clock onset used for classification and RTm.

## Trajectory Configuration (`config.yaml`)

All trajectory drawing parameters are centralised in `config.yaml` under the `trajectory` key:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_z_degree_to_draw` | bool | `true` | Enable per-frame heading integration from dz. When `false`, dx/dy are used raw. |
| `use_escape_onset_heading` | bool | `true` | Heading initialisation for non-rigid mode. `true` = rotate by cumulative dz from trial start to escape onset; `false` = reset angle to zero at onset. Only effective when `use_z_degree_to_draw=true`. |
| `use_escape_onset_only_xy` | bool | `true` | Slice dx/dy to the escape interval only. Applies to both Escape and PreWalk trials. |
| `dz_integration_range` | str | `"escape_interval"` | Range of dz used for Stage 2 rigid rotation angle. `"full_trial"`, `"escape_interval"`, `"trial_to_onset"`, `"escape_angular_peak"`, `"escape_onset_heading"`, or `"peak_bracket"`. Stage 1 curvature always uses the full escape interval dz. |
| `use_rigid_rotation` | bool | `false` | Rigid-body rotation mode: accumulate dx/dy linearly, then apply total yaw as a single rotation. Takes precedence over per-frame heading when `true`. |
| `use_angular_velocity_offset` | bool | `false` | Refine escape interval offset to the first angular-velocity zero-crossing after its peak. |

### Validation (Pydantic TrajectoryConfig)

When using `TrajectoryConfig` from `cercus.config.settings`:
- If `use_rigid_rotation=False`, `dz_integration_range` is ignored with a warning
- If `use_escape_onset_heading=True` and `use_z_degree_to_draw=False`, raises `ValueError`

### Two Rotation Modes

**Non-rigid** (`use_rigid_rotation: false`):
Per-frame dynamic heading integration. Each frame's dx/dy is rotated by the accumulated dz up to that point. Preserves local curvature and S-turns.

**Rigid** (`use_rigid_rotation: true`):
Two-stage decoupled design:
- **Stage 1 (curvature)**: per-frame heading integration using the full escape interval dz. Preserves S-turns and local trajectory shape.
- **Stage 2 (rigid rotation)**: applies a single macro rotation whose angle is determined by `dz_integration_range`:
  - `"escape_interval"` — `sum(dz)` over the escape interval
  - `"trial_to_onset"` — `sum(dz)` from trial start to escape onset
  - `"escape_angular_peak"` — `sum(dz)` from onset to the first angular-velocity zero-crossing after the peak (filters out air-ball rebound)
  - `"escape_onset_heading"` — cumulative heading at escape onset (`cumsum(dz)[onset] / RADIUS`)
  - `"peak_bracket"` — integrate dz from the pre-peak sign change to the post-peak zero-crossing, isolating the dominant rotational impulse. Recommended for cross-paradigm comparison (visual vs wind vs bimodal). See [`cercus/config/defaults/comparison.yaml`](cercus/config/defaults/comparison.yaml).

## Key Modules

### `cercus.core.kinematics.latency.py`
Pure physics functions for escape latency detection via backward-search threshold crossing and escape interval computation. Supports **angular-onset refinement** (opt-in via `escape.use_angular_onset_refinement`): pulls the latency onset earlier to the angular-velocity zero-crossing before the rotational peak, while preserving the coarse 10 mm/s anchor for PreWalk interval detection (`latency_coarse_ms`). No matplotlib dependency.

### `cercus.core.kinematics.trajectory_integration.py`
Dual-stage trajectory integration algorithm:
- **Stage 1**: per-frame heading integration preserving local curvature and S-turns
- **Stage 2**: curvature-thresholded rigid macro rotation for correct left/right fan dispersion

### `pipeline/classifier.py`
State classifier (Escape / PreEscape / PreWalk / NoResponse; PreEscape behind `classification.use_preescape`). It first measures a valid burst in the configured wind-relative detection window, then applies the exclusion checks in priority order: (1) `NoResponse` if no valid burst, (2) `PreEscape` if the measured onset is before wind onset, (3) `PreWalk` if the wind moving-eligibility gate is met (legacy nonwind activity rule otherwise), (4) residual `Escape`. Adds `response_type`, `v_max`, `latency_ms`, `interval_onset_ms`, `interval_offset_ms`, `reaction_time_ms`, `distance_mm`, `distance_500ms_mm` columns.

### `cercus/visualization/`
Publication-grade plotting (Nature/Science/Cell style), split into focused modules:

| Module | Contents |
|---|---|
| `trajectories.py` | Trajectory overlay and unified global overlay |
| `kinetics.py` | Speed kinetics, population spaghetti, single-trial kinetics |
| `heatmaps.py` | Onset-aligned density heatmap, trial-stacked waterfall heatmap |
| `vmax.py` | V_max histogram + KDE, GMM threshold distribution plots |
| `behavior.py` | Behavior probability bars, habituation curves, PreWalk stillness |
| `polar.py` | Escape angle distribution, polar direction rose with Rayleigh test |

### Key Figure Highlights

- `plot_population_speed_kinetics()` — Mean ± SEM speed curves by response type (Escape/PreWalk/NoResponse), aligned to TTC
- `plot_population_spaghetti_kinetics()` — Individual trial spaghetti + mean overlay by response type
- `plot_spaghetti_kinetics_heatmap()` — **Onset-aligned heatmap** (re-aligns time to `interval_onset_ms=0`, uses trial density instead of alpha lines to avoid overplotting)
- `plot_population_polar_histogram()` — 360° polar rose with circular mean μ, resultant R, and **Rayleigh test p-value** for non-uniformity

#### Circular-Stat Tests on Polar Figures (`cercus/visualization/_circstats.py`)

Angles are circular data — linear tests don't apply. The three tests answer three orthogonal questions and are the backbone of the first-level (population) directional analysis:

| Test | Question | Null hypothesis | Where |
|---|---|---|---|
| **Rayleigh** (`rayleigh_p`) | Is there a preferred direction at all? | Directions uniformly distributed on the circle (R too small for n) | per-panel annotation in `polar.py`; per-subject in `individual.py` |
| **Watson-Williams** (`watson_williams_test`) | Do two groups point the *same way*? | Equal mean angles across groups (circular ANOVA; requires grouping R̄ ≥ 0.45, else a correction is needed) | Escape vs PreWalk comparison in `polar.py` |
| **Wallraff** (`wallraff_test`) | Do two groups *scatter* the same much? | Equal angular dispersion around (common) reference — consistency, not direction | Escape vs PreWalk in `polar.py`; cross-condition consistency in `individual.py` |

Mnemonic: **Rayleigh = "is there a direction", Watson-Williams = "same direction?", Wallraff = "equally tight?"**

### `population_analysis.py`

Cross-subject batch processor. Scans an input directory, runs the full preprocess → classify pipeline on every subject in parallel, and outputs unified summary CSV plus population-level visualizations.

**Performance**: Uses `multiprocessing.Pool` for parallel execution. Each subject's data loading and classification runs independently, followed by parallel figure rendering. Control concurrency with `--workers N` (default: all CPU cores). Expected speedup: near-linear with core count for the data processing phase; ~15x for visualization phase on 16+ core machines.

#### Adaptive V_max Threshold

Computes three candidate thresholds independently on **all trials** (not filtered by classification), then selects one for tagging via a priority cascade.

| Priority | Method | Domain | Key Idea |
|---|---|---|---|
| 1 | **KDE Valley** | Physical | Fits a KDE curve (`bw_method=0.3`), detects peaks via `argrelmax`, locates the deepest trough between the first two peaks via `argrelmin`. |
| 2 | **Log-GMM (3-component)** | Log-space | Applies `np.log()` to compress the right tail, fits a **3-component** `GaussianMixture` on **all trials**, produces `start_threshold` and `escape_threshold`. |
| 3 | **IQR-GMM** | Physical (adaptive truncation) | Computes `upper_bound = Q3 + 1.5 * IQR` as a data-driven ceiling, truncates outliers, fits 2-component GMM. |
| 4 | Hardcoded fallback | — | Falls back to 120 mm/s if all methods fail. |

#### Population Visualizations

| Figure | Purpose |
|---|---|
| `habituation.svg` | Fatigue inspection: per-subject V_max traces + mean ± SEM |
| `vmax_gmm.svg` | Threshold determination: all trials + GMM markers |
| `vmax_moving_gmm.svg` | 2-component moving GMM: windowed threshold evolution |
| `vmax_response.svg` | Effective response inspection: Escape+PreWalk only |
| `behavior_prob.svg` | Response proportions bar chart |
| `reaction_distance_panel.svg` | Class-selected timing, distance, and paired final Wind PreWalk T1/T2; pure wind/multimodal split, classifier N and endpoint coverage |
| `prewalk_analysis_panel.svg` | PreWalk polar distribution + stillness scatter |
| `prewalk_stillness.svg` | PreWalk stillness analysis |
| `speed_kinetics.svg` | Population speed kinetics by response type |
| `spaghetti_kinetics.svg` | Population spaghetti plots |
| `spaghetti_kinetics_heatmap.svg` | Onset-aligned density heatmap |
| `spaghetti_kinetics_heatmap_fullres.svg` | Full-trial high-resolution heatmap (no onset alignment) |
| `escape_angle_distribution.svg` | Escape direction histogram + KDE |
| `polar_direction_histogram.svg` | Polar rose with Rayleigh p-value |

### `pipeline/mcmc.py`
Bayesian psychophysics via PyMC/NumPyro. Fits psychometric sigmoid functions to escape probability vs. TTC, tests multisensory integration hypotheses (ROPE-based posterior probability), computes Bayesian optimal integration (variance reduction), and performs survival analysis (Kaplan-Meier, Race Model Inequality).

**Data Requirements**:
- **Minimum**: ≥2 TTC conditions (e.g. `-308`, `-373`) for valid sigmoid fitting
- **Recommended**: ≥4 TTC conditions for smooth Time-Window of Integration (TWoI) curves (2-3 conditions result in linear/quadratic interpolation, which is mathematically correct but visually appears as a straight/segmented line)
- Each condition should have ≥5 trials for stable posterior estimation

#### `--binary-mode`

| Mode | Escape (= 1) | Non-escape (= 0) |
|---|---|---|
| `escape_only` (default) | Escape | PreEscape, PreWalk, NoResponse |
| `escape_prewalk` | Escape, PreEscape, PreWalk (any burst response) | NoResponse |

`escape_only` deliberately **excludes PreEscape** from the sigmoid fit: a pre-wind, pure-vision escape is not waiting on the looming → including it would bias the TTC–P(Escape) curve. The Kaplan–Meier survival / race-model analysis, in contrast, counts both Escape and PreEscape as escape events (a burst is a burst, regardless of when it fires relative to the wind).

## Input Data Format

The input directory must contain paired CSV files following the naming convention:

- `{subject_name}_session_{N}_events.csv` — columns: `event_name`, `timestamp`, `global_trial_id`, `details` (JSON)
- `{subject_name}_session_{N}_kinematics.csv` — columns: `sys_time`, `dx`, `dy`, `dz`, `stim_state`, `global_trial_id`

Sessions are auto-discovered and paired by subject name and session ID.

## Output Structure

### Single-subject pipeline (`main.py` or `python -m cercus.cli.app single`)

```
figures/
└── <subject>/
    ├── response/              trajectory_overlay.svg, speed_kinetics.svg, spaghetti_kinetics.svg
    ├── prewalk/               trial_<N>_prewalk.svg
    ├── no_response/           trial_<N>_noresponse.svg
    ├── behavior_probability_distribution.svg
    ├── habituation_curve.svg
    ├── vmax_distribution_diagnostic.svg
    ├── escape_angle_distribution.svg
    └── individual_escapes/    trial_<N>_escape.svg
```

### Population batch (`population_analysis.py`)

Subject processing uses up to `--workers` processes (all CPUs by default), capped
by the number of subjects. Rendering serializes the population DataFrame once to
a temporary snapshot; each worker reads it once and figure jobs carry only plot
options. The snapshot is removed when the rendering pool exits. Its directory is
set by `analysis.parallel.cache_dir` (default `/mnt/d/Data/Results/tmp`). The pool is
limited by `analysis.parallel.render_workers` (default `4`), `--workers`, and the
number of figure jobs. Override the YAML values to tune another dataset:

```yaml
analysis:
  parallel:
    render_workers: 4
```

Logs report subject processing, rendering, individual checks, and total pipeline
time. The local `population.ps1` batch runner uses the required WSL `torch`
environment and accepts `-Workers`, `-Paradigms`, and WSL-path `-InputRoot` /
`-OutputRoot` arguments.

```
<output-dir>/
├── population_summary.csv             # Per-trial metrics with is_valid_escape tagging
├── subject_escape_rates.csv           # Per-subject escape rate summary
├── habituation.svg                    # Fatigue curve
├── vmax_gmm.svg                       # All trials V_max + GMM thresholds
├── vmax_moving_gmm.svg                # 2-component moving GMM threshold evolution
├── vmax_response.svg                  # Escape+PreWalk V_max only
├── behavior_prob.svg                  # Response proportion bar chart
├── prewalk_analysis_panel.svg         # PreWalk polar + stillness scatter
├── prewalk_stillness.svg              # PreWalk stillness analysis
├── speed_kinetics.svg                 # Population speed kinetics
├── spaghetti_kinetics.svg             # Population spaghetti plots
├── spaghetti_kinetics_heatmap.svg     # Onset-aligned density heatmap
├── spaghetti_kinetics_heatmap_fullres.svg  # Full-trial high-res heatmap
├── escape_angle_distribution.svg      # Histogram of escape angles
└── polar_direction_histogram.svg      # Polar rose with Rayleigh p-value
```

## Regression Testing

```bash
# Run classifier golden tests (10 synthetic trials, bitwise comparison)
pytest tests/test_classifier_golden.py -v

# Run visualization regression tests (hash-based pixel comparison)
pytest tests/test_visualization_regression.py -v

# Update visualization baselines
rm tests/baselines/*.sha256 && pytest tests/test_visualization_regression.py -v
```

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Array computation, DataFrame manipulation |
| `matplotlib`, `scipy` | Figure generation, Savitzky-Golay smoothing |
| `scikit-learn` | Gaussian Mixture Model for adaptive V_max threshold |
| `pyyaml` | `config.yaml` parsing |
| `pymc`, `arviz` | Bayesian MCMC model specification and diagnostics |
| `numpyro`, `jax`, `jaxlib` | JAX-based NUTS sampler (10–100× faster, optional) |
| `lifelines` | Kaplan-Meier survival analysis |
| `pydantic` | Configuration model validation (TrajectoryConfig) |
| `typer` | Unified CLI application |
| `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `pyyaml` | Core scientific stack |

## License

[Add license here]

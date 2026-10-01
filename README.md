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

For each trial, let `v(t)` denote walking speed, `V_b` the burst threshold, `t_w` the wind onset, and `t_e` the measured escape onset. For every standard wind-containing trial, define the post-stimulus response interval

```text
W = [t_w, t_w + 250 ms]
```

where `t_w = target_ttc_ms`. For other non-`baseline_visual` trials, `t_w` is replaced by the trial's stimulus reference (`t_rel = 0`). A valid burst exists iff

```text
max(v(t) : t ∈ W) > V_b
```

with `V_b = 98 mm/s` by package default and `V_b = 50 mm/s` under the current root `config.yaml`. If no valid burst exists, the response is `NoResponse`; no pre-stimulus condition can override this veto.

For the pre-movement check, define

```text
P(a) = mean( v(t) > 10 mm/s | a − 1000 ms ≤ t < a − 50 ms )
```

using valid speed **frames** only. For wind trials, `a = t_w`; for other standard trials, `a = t_e`. `PreWalk` requires a valid burst and `P(a) > 0.15`; this is a cumulative frame proportion and does not require the above-threshold frames to be consecutive.

For eligible multimodal (`looming+wind`) trials only, `PreEscape` requires a valid burst and

```text
t_e < t_w − preescape_buffer_ms
```

The current buffer is `0 ms`. `PreEscape` is disabled when `thresholds.classification.use_preescape` is false. `Escape` is the residual class:

```text
valid_burst ∧ ¬PreEscape ∧ ¬PreWalk
```

The implemented priority is therefore:

```text
NoResponse → PreEscape → PreWalk → Escape
```

`baseline_visual` is the only exception. Its response domain is `t ≤ TTC` rather than the standard 250-ms post-stimulus interval. A valid burst is defined by `max(v(t) : t ≤ TTC) > V_b`; its pre-movement check uses `a = t_e` in `P(a)`. `PreEscape` is undefined for this paradigm, so the effective routing is `NoResponse → PreWalk → Escape`.

On the standard path, escape interval onset is obtained by locating the first sample above `V_b` in the response window and searching backward for the last sample below `10 mm/s`; the following sample is `t_e`. The interval offset is the first subsequent sample below `10 mm/s`. For `baseline_visual`, the interval instead brackets the maximum-speed peak up to TTC using the nearest below-threshold samples before and after that peak. Thus, a separate early movement that ends below `10 mm/s` before a later post-wind burst may be visible in a full-trial heatmap without determining `t_e` or satisfying the implemented `PreEscape` condition.

The 250-ms duration and speed thresholds are configurable; the expressions above use the current timing setting. Missing response-window observations are routed to `NoResponse` by the implementation, but are not evidence of immobility. If the pre-movement window has no valid samples, `P(a)` is undefined and that check alone does not assign `PreWalk`.

For pure-wind trials, the backward search cannot reuse a pre-wind walking onset. Ongoing movement without an observable fresh post-wind onset retains its burst-based response class but has missing onset/RT metrics. Multimodal trials still allow pre-wind onsets for `PreEscape`.

### Wind reaction-time measurements

Let `t_a = t_w + thresholds.reaction_time.wind_arrival_delay_ms` denote calibrated airflow arrival. The default delay is `0 ms` (hardware-trigger reference, **not** independently measured airflow arrival).

```text
escape_reaction_time_ms   = escape movement onset − t_a
stillness_reaction_time_ms = stopping threshold crossing − t_a  (Wind PreWalk only; T1-like)
stop_to_escape_interval_ms = acquisition-clock escape onset − stopping crossing  (T2-like)
```

The paper's T2 (87.16 ± 65.31 ms) is **stopping-to-response time**, not stopping latency. `stop_to_escape_interval_ms` requires both endpoints on `t_acquisition_rel` and finite, increasing source timestamps between them; otherwise it is `NaN`. It measures the endpoint interval, not verified uninterrupted immobility. Do not compute it by subtracting the host-clock `escape_reaction_time_ms` from source-clock stopping latency.

`stillness_presence` reports an observed causal averaged low-speed state separately: `low_speed`, `no_low_speed`, or `unobserved` for Wind PreWalk; other classes are `not_applicable`. Its observation window uses wholly post-reference speed support and ends before known escape onset or at the 250-ms limit. A valid below-threshold observation establishes presence; absence requires valid window coverage. An animal already quiet at wind can have `low_speed` without a stopping RT. `prewalk_stillness.svg` displays these three categories and annotates RT coverage separately; `NaN` RT is **not** absence of stillness.

The paper's [Figure 1D](https://pmc.ncbi.nlm.nih.gov/articles/PMC10405261/#fig1) gives **88.75% pause incidence at 1.00 m/s** among trials stimulated in its moving state (>10 mm/s for >1 s), including pauses without subsequent escape. This denominator is not our broad, escape-selected PreWalk class. Mixed multisensory trials, locally moving trials, and finite stopping-RT counts must not be pooled as that paper probability.

Wind PreWalk's compatibility `reaction_time_ms` is **stopping latency**, not escape latency; other wind responses use escape latency. The stopping threshold (`thresholds.baseline.quiet_mm_s`, default `10 mm/s`) and 250-ms earlier period follow [Motor state changes escape behavior of crickets (2023)](https://doi.org/10.1016/j.isci.2023.107345). A broad PreWalk label alone does **not** establish that the animal was moving at stimulus onset. The local kinematic measurement requires above-threshold causal speed at the reference, separated from the threshold by the count-resolution margin below. A full, valid preceding `thresholds.prewalk.window_ms` (default `1000 ms`) is still required. `stillness_baseline_status` records `continuous_moving` when every averaged observation in that history is ≥ threshold, otherwise `intermittent_moving`; missing history is `unobserved`, and other response classes are `not_applicable`. Earlier interruptions no longer erase an otherwise measurable post-reference stop. These labels describe the observed averaged trace, not verified uninterrupted locomotion: count quantization alone may break a genuinely continuous near-threshold walk. The paper does not specify how its “more than 1 s” rule handled averaging or intermittent counts; neither label establishes membership in its moving cohort. Local endpoints from `intermittent_moving` trials must not be pooled as an exact paper T1 replication.

`speed_raw` uses displacement divided by **firmware acquisition-time differences** (`ard_time`, milliseconds), not `sys_time` serial-receive differences. `t_acquisition_rel` places the first active stimulus row at the same wind coordinate as `t_rel` while retaining source-clock intervals. Host chunking can assign one timestamp to several sensor rows; host intervals are not sensor integration durations. Missing acquisition timestamps are not silently replaced with the centered `speed` signal.

Stopping detection uses a **causal displacement average** over `thresholds.stillness.speed_window_ms` (default `20 ms`). The configured primary-axis displacement quantum is about `0.047 mm`, observed in the audited calibrated recordings and consistent with the sibling recorder's local `calibration_cfg.json` coefficient (`0.04712389037013054 mm/count`). It is rig-specific host calibration, not sensor-native CPI or a datasheet constant; calibration provenance and the exact recorder/firmware versions used for all recordings remain unverified. Other rigs must supply their own `displacement_quantum_mm`; taking the smallest positive increment is not a safe calibration because the recorder matrix can mix rotational counts into tiny planar increments. With this scale, near `10 mm/s` a 5-ms interval contains roughly one count, whereas 20 ms contains roughly four. A single zero-count row therefore cannot establish instantaneous stillness. The first averaged-speed sample below threshold is the stopping endpoint only if the same below-threshold excursion reaches below `threshold − sqrt(2) × displacement_quantum_mm × 1000 / speed_window_ms` before rebound, escape onset, or the search-window end. This local diagonal-count margin is a resolution safeguard, **not a validated speed error bound**. At the defaults the margin is approximately `3.33 mm/s`: reference speed must be `>13.33 mm/s`, and the same excursion must reach `<6.67 mm/s` to confirm its first `<10 mm/s` crossing. These eligibility gates are stricter than the paper's threshold alone. Reference speeds within the margin on either side of the threshold are `threshold_unresolved`; `not_moving_at_wind` requires speed below `threshold − margin`. Post-reference dips that do not resolve are also `threshold_unresolved` and cannot be replaced by a later crossing. There is no separate 30-ms dwell or 50-ms reaction-time floor. Its averaging interval must be wholly post-reference, so the earliest eligible endpoint is one full averaging window after the reference; absence of shorter endpoints is not evidence of a biological minimum. A first stop whose window straddles wind is unresolved, not replaced by a later crossing. Finite, ordered acquisition data with gaps no larger than `thresholds.stillness.max_frame_gap_factor × median source interval` are required throughout the measurement history. Missing data after an already observed stop do not erase it.

Stopping can be measured with unresolved escape onset. When escape onset is observed, its row is mapped onto the acquisition clock; stopping must **strictly precede** it. Missing or ineligible measurements yield `NaN` and a `stillness_status` reason without changing response classes. `observed` means the defined event was measurable, **not** that airflow caused it. The measurement status is distinct from `stillness_baseline_status`, which records whether the preceding valid 1-s speed history was continuously above threshold; `intermittent_moving` trials may have a local observed endpoint but have not established eligibility under the paper's moving-state definition. Other measurement statuses include `not_moving_at_wind`, `threshold_unresolved`, `insufficient_baseline`, `wind_boundary`, `invalid_data`, `missing_acquisition_clock`, `invalid_acquisition_clock`, `escape_first`, `no_preescape_stop`, and `not_applicable`.


The stopping endpoint is the timestamp of an **averaged-speed threshold crossing**, not an exact physiological reaction instant. Causal averaging adds speed-dependent detection lag; the 20-ms support is neither a confidence interval nor a validated bound on the true reaction time. No fixed filter-delay subtraction is applied. With arrival delay `0`, the value is stimulus-flag-relative, not independently calibrated airflow-arrival-relative. Causality and a hard biological minimum cannot be recovered from these kinematics alone.

Movement onset, distance integration, and visual-only timing remain unchanged. `distance_mm` and `distance_500ms_mm` still start at escape **movement** onset. `escape_reaction_time_ms` still uses the existing centered trajectory-speed measurement; its short values require a separate onset-method and hardware-timing audit. `short_rt` flags only nonnegative escape latency below `thresholds.reaction_time.short_escape_ms` (default `50 ms`), without clamping or discarding it. PreEscape negatives are pre-wind leads, not negative reactions.

This is **not an exact reproduction** of the paper's preprocessing or complete T1/T2/RTm analysis. The causal averaging is a local count-resolution safeguard. Increased or decreased valid sample count is not evidence of improved biological reaction time.

#### Separate pause-response measurement

All airflow-containing trials additionally receive `pause_stopping_time_ms` (T1-like), `pause_to_escape_time_ms` (T2-like), and `pause_reaction_time_ms` (RTm-like = T1 + T2). Both endpoints use the **same causal displacement-average speed on the acquisition clock**. The first pause must occur within 250 ms of the airflow reference and before a qualifying escape; the follow-up burst is sought within 250 ms **after the stopping endpoint**, not only after stimulus onset. The burst and onset thresholds reuse the configured escape criteria (currently 50 and 10 mm/s). No new filter is attributed to the paper, whose retrieved methods do not specify its velocity filtering.

`pause_status` reports stopping eligibility, `pause_escape_status` reports `observed`, `no_escape`, `invalid_data`, or `not_applicable`, and `pause_baseline_status` records the independent moving-history proxy. In this separate pipeline, `continuous_moving` requires every valid causal-speed observation over the preceding configured 1-s history, including the reference, to be **strictly >10 mm/s** (roundoff-safe); equality and earlier low-speed interruptions are `intermittent_moving`. The legacy `stillness_baseline_status` definition is unchanged. A local observed transition remains exported in an intermittent trial but is not promoted to a paper-cohort result. A stop without a resolved subsequent escape retains T1 but has missing T2/RTm; incomplete follow-up is `invalid_data`, not absence of escape. These fields run before the legacy no-burst/classification gates, so a stop-relative later escape may be observed even when the original class is `NoResponse`. **They do not change response classes, distance integration, or any legacy RT field.** Both stopping pipelines use raw acquisition intervals, but their escape caps differ: legacy `stillness_*` ends before the mapped centered-speed escape onset, whereas `pause_*` ends before the causal onset or earlier burst. Their stopping statuses can therefore differ; `no_preescape_stop` means no accepted stop before that pipeline's cap, not no low-speed state anywhere in the trial. They are neither a continuous-immobility duration nor an exact physiological RT. No fixed 20-ms correction, biological floor, or literature-mean fitting is applied.

`population/reaction_distance_panel.svg` has three panels: stimulus-to-escape latency, escape distance, and **paired strict-moving T1/T2**. It never presents the compatibility stopping latency as total escape RT. For an observed PreWalk pause it prefers the new causal RTm-like endpoint (including `NaN` when unresolved); other trials retain `escape_reaction_time_ms`. Missing causal endpoints are not replaced with centered ones, and RT coverage is annotated. When causal fields are available, the T1/T2 panel defaults to `pause_baseline_status == continuous_moving` across **all wind response classes**, not broad PreWalk; missing history cannot establish eligibility. `plot_reaction_distance_panel(..., pause_cohort="all_local")` provides a separately labeled local-transition diagnostic view, including intermittent movers. Legacy tables without causal fields retain an explicitly labeled PreWalk fallback. Pure wind and multimodal pairs stay separate; boxes/dots show one median per subject from complete trial pairs, while faint lines with coincident endpoint markers join each actual trial pair. Coverage/legend text stays outside the data rectangle. Paired coverage is not pause incidence; low-speed levels in already-stopped animals are not stopping transitions. The population job remains inside the existing worker-controlled `multiprocessing.Pool`; no separate serial rendering pipeline was added. Figure size is YAML-backed (`visualization.reaction_distance_figsize`).

In trial-stacked heatmaps, the hardware wind reference is shown using `colors.yaml: wind_mark`: a full-height line in TTC-aligned panels and one tick per row in onset-aligned panels.

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
State classifier (Escape / PreEscape / PreWalk / NoResponse; PreEscape behind `classification.use_preescape`). It first measures a valid burst in the configured detection window, then applies the exclusion checks in priority order: (1) `NoResponse` if no valid burst, (2) `PreEscape` if the measured burst onset is before wind onset (multimodal only), (3) `PreWalk` if pre-wind activity exceeds threshold, (4) residual `Escape`. Adds `response_type`, `v_max`, `latency_ms`, `interval_onset_ms`, `interval_offset_ms`, `reaction_time_ms`, `distance_mm`, `distance_500ms_mm` columns.

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
| `reaction_distance_panel.svg` | Stimulus-to-escape RT (not PreWalk stopping RT), distance, and paired Wind PreWalk T1/T2; pure wind/multimodal split, subject medians and endpoint coverage |
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

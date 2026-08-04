# Cercus Framework

Cricket escape-response behavioral analysis pipeline. Processes raw kinematics data, classifies trials into Escape / PreWalk / NoResponse, generates publication-grade figures (Nature/Science style), and performs Bayesian population-level inference via MCMC.

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
│   ├── classifier.py                # Ternary classification
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
| `python -m cercus.cli.app population --input <dir> --output <dir>` | Population-level batch | `population_analysis.py` |
| `python -m cercus.cli.app mcmc --input <dir> --output <dir>` | Bayesian MCMC analysis | `mcmc_analysis.py` |
| `python -m cercus.cli.app trial-panels --input <dir> --output <dir>` | Per-trial composite panels | `plot_trial_panels.py` |
| `python -m cercus.cli.app trajectories --input <dir> --output <dir>` | Unified trajectory overlay | `plot_all_trajectories_fixed.py` |

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
| `thresholds.yaml` | Escape/PreWalk detection thresholds (`vmax_threshold`, `start_threshold`, etc.) |
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

| Type | Criteria |
|---|---|
| **Escape** | Pre-stimulus speed < 10 mm/s; post-stimulus V_max > (gmm threshold) mm/s within 250 ms |
| **PreWalk** | Pre-stimulus speed > 10 mm/s in the 1-s window before onset; post-stimulus V_max > (gmm threshold) mm/s within 250 ms |
| **NoResponse** | Post-stimulus V_max ≤ (gmm threshold) mm/s within 250 ms |

Escape latency is defined as the first time speed exceeds 10 mm/s just before reaching 50 mm/s. The escape interval spans from latency onset to the point where speed drops back below 10 mm/s, with optional angular-velocity zero-crossing refinement.

Classification priority order: **NoResponse > PreWalk > Escape**. A trial is routed to the first matching category — e.g. if both PreWalk and Escape conditions are met, the trial is classified as PreWalk. When baseline speed ≥ 10 mm/s and no pre-walk activity is detected, the trial falls back to NoResponse even if a valid burst exists.

### `baseline_visual` Special Handling

For `baseline_visual` (visual-only looming) trials the stimulus onset precedes TTC (`t_rel = 0`), so the standard fixed-window burst detection does not apply. The classifier adapts as follows:

| Aspect | Standard (wind / bimodal) | `baseline_visual` |
|---|---|---|
| Burst detection window | `[onset, onset + 250 ms]` | Entire stimulus period (`t_rel ≤ 0`) |
| PreWalk check anchor | Stimulus onset (`t_rel = 0` or wind onset) | `latency_ms` (escape onset) |
| Escape baseline check | Speed at stimulus onset | Speed at the frame immediately before `latency_ms` |

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
Pure physics functions for escape latency detection via backward-search threshold crossing and escape interval computation. No matplotlib dependency.

### `cercus.core.kinematics.trajectory_integration.py`
Dual-stage trajectory integration algorithm:
- **Stage 1**: per-frame heading integration preserving local curvature and S-turns
- **Stage 2**: curvature-thresholded rigid macro rotation for correct left/right fan dispersion

### `pipeline/classifier.py`
Ternary state classifier. Priority order: (1) NoResponse if no valid burst, (2) PreWalk if pre-stimulus activity exceeds threshold, (3) Escape if baseline is quiescent. Adds `response_type`, `v_max`, `latency_ms`, `interval_onset_ms`, `interval_offset_ms` columns.

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

### `population_analysis.py`

Cross-subject batch processor. Scans an input directory, runs the full preprocess → classify pipeline on every subject, and outputs unified summary CSV plus population-level visualizations.

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
| `vmax_response.svg` | Effective response inspection: Escape+PreWalk only |
| `behavior_prob.svg` | Response proportions bar chart |
| `prewalk_stillness.svg` | PreWalk stillness analysis |
| `speed_kinetics.svg` | Population speed kinetics by response type |
| `spaghetti_kinetics.svg` | Population spaghetti plots |
| `spaghetti_kinetics_heatmap.svg` | Onset-aligned density heatmap |
| `escape_angle_distribution.svg` | Escape direction histogram + KDE |
| `polar_direction_histogram.svg` | Polar rose with Rayleigh p-value |

### `pipeline/mcmc.py`
Bayesian psychophysics via PyMC/NumPyro. Fits psychometric sigmoid functions to escape probability vs. TTC, tests multisensory integration hypotheses (ROPE-based posterior probability), computes Bayesian optimal integration (variance reduction), and performs survival analysis (Kaplan-Meier, Race Model Inequality).

#### `--binary-mode`

| Mode | Escape (= 1) | Non-escape (= 0) |
|---|---|---|
| `escape_only` (default) | Escape | PreWalk, NoResponse |
| `escape_prewalk` | Escape, PreWalk | NoResponse |

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
├── vmax_response.svg                  # Escape+PreWalk V_max only
├── behavior_prob.svg                  # Response proportion bar chart
├── prewalk_stillness.svg              # PreWalk stillness analysis
├── speed_kinetics.svg                 # Population speed kinetics
├── spaghetti_kinetics.svg             # Population spaghetti plots
├── spaghetti_kinetics_heatmap.svg     # Onset-aligned density heatmap
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

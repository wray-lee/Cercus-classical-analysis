# Cercus Framework

Cricket escape-response behavioral analysis pipeline. Processes raw kinematics data, classifies trials into Escape / PreWalk / NoResponse, generates publication-grade figures (Nature/Science style), and performs Bayesian population-level inference via MCMC.

## Standardized Classification Criteria

| Type | Criteria |
|---|---|
| **Escape** | Pre-stimulus speed < 10 mm/s; post-stimulus V_max > 50 mm/s within 250 ms |
| **PreWalk** | Pre-stimulus speed > 10 mm/s in the 1-s window before onset; post-stimulus V_max > 50 mm/s within 250 ms |
| **NoResponse** | Post-stimulus V_max ≤ 50 mm/s within 250 ms |

Escape latency is defined as the first time speed exceeds 10 mm/s just before reaching 50 mm/s. The escape interval spans from latency onset to the point where speed drops back below 10 mm/s, with optional angular-velocity zero-crossing refinement.

Classification priority order: **NoResponse > PreWalk > Escape**. A trial is routed to the first matching category — e.g. if both PreWalk and Escape conditions are met, the trial is classified as PreWalk. When baseline speed ≥ 10 mm/s and no pre-walk activity is detected, the trial falls back to NoResponse even if a valid burst exists.

### `baseline_visual` Special Handling

For `baseline_visual` (visual-only looming) trials the stimulus onset precedes TTC (`t_rel = 0`), so the standard fixed-window burst detection does not apply. The classifier adapts as follows:

| Aspect | Standard (wind / bimodal) | `baseline_visual` |
|---|---|---|
| Burst detection window | `[onset, onset + 250 ms]` | Entire stimulus period (`t_rel ≤ 0`) |
| PreWalk check anchor | Stimulus onset (`t_rel = 0` or wind onset) | `latency_ms` (escape onset) |
| Escape baseline check | Speed at stimulus onset | Speed at the frame immediately before `latency_ms` |

## Project Structure

```
Cercus-cli/
├── main.py                      # Full pipeline: preprocess → classify → visualize
├── plot_trial_panels.py         # Per-trial composite panels (speed, angvel, stimulus, trajectory)
├── plot_all_trajectories_fixed.py  # Unified trajectory overlay across all subjects
├── mcmc_analysis.py             # Bayesian MCMC psychophysics analysis
├── population_analysis.py       # Cross-subject batch: adaptive threshold + population visualization
├── config.yaml                  # Trajectory drawing configuration
├── requirements.txt
├── pipeline/
│   ├── io.py                    # CSV loading, session pairing, summary export
│   ├── kinematics.py            # Speed, angular velocity, escape latency & interval computation
│   ├── classifier.py            # Ternary classification: Escape / PreWalk / NoResponse
│   ├── visualization.py         # Publication-grade plotting functions
│   ├── constants.py             # Thresholds, colors, geometry, YAML config loader
│   └── mcmc.py                  # Bayesian hierarchical models (PyMC / NumPyro)
└── test/
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline: classify + visualize all trials
python main.py --input-dir path/to/data/ --save figures/

# Per-trial composite panels
python plot_trial_panels.py --input-dir path/to/data/ --save figures/

# Unified trajectory overlay
python plot_all_trajectories_fixed.py --input-dir path/to/data/ --save trajectories.svg

# Population batch: adaptive threshold + summary CSV + population figures
python population_analysis.py --input-dir path/to/data/ --output-dir results/

# Bayesian MCMC analysis
python mcmc_analysis.py --input-dir path/to/data/ --output-dir results/
```

## Input Data Format

The input directory must contain paired CSV files following the naming convention:

- `{subject_name}_session_{N}_events.csv` — columns: `event_name`, `timestamp`, `global_trial_id`, `details` (JSON)
- `{subject_name}_session_{N}_kinematics.csv` — columns: `sys_time`, `dx`, `dy`, `dz`, `stim_state`, `global_trial_id`

Sessions are auto-discovered and paired by subject name and session ID. Legacy merged CSV data can be converted via `test/converterOld.py`.

## Trajectory Configuration (`config.yaml`)

All trajectory drawing parameters are centralised in `config.yaml` under the `trajectory` key:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_z_degree_to_draw` | bool | `true` | Enable per-frame heading integration from dz. When `false`, dx/dy are used raw. |
| `use_escape_onset_heading` | bool | `true` | Heading initialisation for non-rigid mode. `true` = rotate by cumulative dz from trial start to escape onset; `false` = reset angle to zero at onset. Only effective when `use_z_degree_to_draw=true`. |
| `use_escape_onset_only_xy` | bool | `true` | Slice dx/dy to the escape interval only. Applies to both Escape and PreWalk trials. |
| `dz_integration_range` | str | `"escape_interval"` | Range of dz used for Stage 2 rigid rotation angle. `"full_trial"`, `"escape_interval"`, `"trial_to_onset"`, `"escape_angular_peak"`, or `"escape_onset_heading"`. Stage 1 curvature always uses the full escape interval dz. |
| `use_rigid_rotation` | bool | `false` | Rigid-body rotation mode: accumulate dx/dy linearly, then apply total yaw as a single rotation. Takes precedence over per-frame heading when `true`. |
| `use_angular_velocity_offset` | bool | `false` | Refine escape interval offset to the first angular-velocity zero-crossing after its peak. |

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

The two stages are independent: changing `dz_integration_range` only affects the rotation angle, not the trajectory curvature.

## Key Modules

### `pipeline/kinematics.py`
Pure physics layer. Integrates body-frame dx/dy/dz into trajectories with Savitzky-Golay smoothing, computes speed (mm/s) and angular velocity (rad/s), extracts escape latency and interval via backward-search threshold crossing.

### `pipeline/classifier.py`
Ternary state classifier. Priority order: (1) NoResponse if no valid burst, (2) PreWalk if pre-stimulus activity exceeds threshold, (3) Escape if baseline is quiescent. Adds `response_type`, `v_max`, `latency_ms`, `interval_onset_ms`, `interval_offset_ms` columns.

### `pipeline/visualization.py`
Publication-grade plotting (Nature/Science/Cell style). Includes trajectory overlay, speed kinetics, spaghetti plots, behavior probability, habituation curves, V_max distribution, and the two-stage trajectory integration algorithm (local heading + curvature-thresholded rigid rotation).

### `population_analysis.py`

Cross-subject batch processor. Scans an input directory, runs the full preprocess → classify pipeline on every subject, and outputs unified summary CSV plus population-level visualizations.

#### Adaptive V_max Threshold

Computes three candidate thresholds independently on **all trials** (not filtered by classification), then selects one for tagging via a priority cascade. Using all trials decouples the adaptive threshold from the fixed `ESCAPE_VMAX_THRESHOLD`, avoiding circular dependency.

| Priority | Method | Domain | Key Idea |
|---|---|---|---|
| 1 | **KDE Valley** | Physical | Fits a KDE curve (`bw_method=0.3`), detects peaks via `argrelmax`, locates the deepest trough between the first two peaks via `argrelmin`. Returns `None` if fewer than 2 peaks are found. |
| 2 | **Log-GMM (3-component)** | Log-space | Applies `np.log()` to compress the right tail, fits a **3-component** `GaussianMixture` on **all trials** (no classification dependency), producing two intersection points: `start_threshold` (no-response vs movement) and `escape_threshold` (movement vs burst). Back-transforms with `np.exp()`. |
| 3 | **IQR-GMM** | Physical (adaptive truncation) | Computes `upper_bound = Q3 + 1.5 * IQR` as a data-driven ceiling, truncates outliers above it, then fits a 2-component GMM in the original physical domain. Disabled by default (`_ENABLE_IQR_GMM = False`). |
| 4 | Hardcoded fallback | — | Falls back to 120 mm/s if all methods fail. |

The Log-GMM produces **two** thresholds from a single fit:

| Threshold | Meaning | Typical range |
|---|---|---|
| `start_threshold` | No-response vs any movement | ~20 mm/s |
| `escape_threshold` | Weak movement vs escape burst | ~160 mm/s |

`escape_threshold` is used for `is_valid_escape` tagging and per-subject escape rates. The V_max distribution plot displays both thresholds (orange = start, red = escape) plus the IQR-GMM threshold (blue) if enabled.

#### Population Visualizations

Three SVG figures are generated (headless rendering via `matplotlib.use("Agg")`):

| Figure | Purpose | Content |
|---|---|---|
| `habituation.svg` | Fatigue inspection | Per-subject V_max traces + mean ± SEM ribbon across trial sequence |
| `vmax_gmm.svg` | **Threshold determination** | All trials (Escape + PreWalk + NoResponse) histogram + KDE, with GMM start (orange `#E69F00`), GMM escape (red `#DC0000`), optional IQR-GMM (blue `#3C5488`), and fixed reference lines (50 / 10 mm/s) |
| `vmax_response.svg` | **Effective response inspection** | Escape + PreWalk only histogram + KDE, with the active tagging threshold (red) — excludes NoResponse noise |
| `behavior_prob.svg` | Response proportions | Escape / PreWalk / NoResponse bar chart across all subjects |

### `pipeline/mcmc.py`
Bayesian psychophysics via PyMC/NumPyro. Fits psychometric sigmoid functions to escape probability vs. TTC, tests multisensory integration hypotheses (ROPE-based posterior probability), computes Bayesian optimal integration (variance reduction), and performs survival analysis (Kaplan-Meier, Race Model Inequality).

#### `--binary-mode`

Controls how the ternary classification is collapsed into a binary escape/non-escape response for the Bernoulli likelihood:

| Mode | Escape (= 1) | Non-escape (= 0) |
|---|---|---|
| `escape_only` (default) | Escape | PreWalk, NoResponse |
| `escape_prewalk` | Escape, PreWalk | NoResponse |

Example:

```bash
python mcmc_analysis.py --input-dir data/ --output-dir results/ --binary-mode escape_prewalk
```

## Output Structure

### Single-subject pipeline (`main.py`)

```
figures/
└── <subject>/
    ├── response/       trial_<N>_escape.svg
    ├── prewalk/        trial_<N>_prewalk.svg
    └── no_response/    trial_<N>_noresponse.svg
```

### Population batch (`population_analysis.py`)

```
<output-dir>/
├── population_summary.csv      # Per-trial metrics with is_valid_escape tagging
├── subject_escape_rates.csv    # Per-subject escape rate summary
├── habituation.svg             # Fatigue curve: per-subject lines + mean±SEM ribbon
├── vmax_gmm.svg                # All trials V_max + GMM threshold lines (threshold determination)
├── vmax_response.svg           # Escape+PreWalk V_max only (effective response inspection)
└── behavior_prob.svg           # Escape / PreWalk / NoResponse proportion bar chart
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

## License

[Add license here]

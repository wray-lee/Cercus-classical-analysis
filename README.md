# Cercus Framework

Cricket escape-response behavioral analysis pipeline. Processes raw kinematics data, classifies trials into Escape / PreWalk / NoResponse, generates publication-grade figures (Nature/Science style), and performs Bayesian population-level inference via MCMC.

## Standardized Classification Criteria

| Type | Criteria |
|---|---|
| **Escape** | Pre-stimulus speed < 10 mm/s; post-stimulus V_max > 50 mm/s within 250 ms |
| **PreWalk** | Pre-stimulus speed > 10 mm/s in the 1-s window before onset; post-stimulus V_max > 50 mm/s within 250 ms |
| **NoResponse** | Post-stimulus V_max ≤ 50 mm/s within 250 ms |

Escape latency is defined as the first time speed exceeds 10 mm/s just before reaching 50 mm/s. The escape interval spans from latency onset to the point where speed drops back below 10 mm/s, with optional angular-velocity zero-crossing refinement.

## Project Structure

```
Cercus-cli/
├── main.py                      # Full pipeline: preprocess → classify → visualize
├── plot_trial_panels.py         # Per-trial composite panels (speed, angvel, stimulus, trajectory)
├── plot_all_trajectories_fixed.py  # Unified trajectory overlay across all subjects
├── mcmc_analysis.py             # Bayesian MCMC psychophysics analysis
├── population_analysis.py       # Cross-subject batch summary (zero-rendering)
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

# Population summary (no rendering)
python population_analysis.py --input-dir path/to/data/ --output-csv population_summary.csv

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
| `dz_integration_range` | str | `"escape_interval"` | Range of dz used for heading integration and total yaw (rigid mode). `"full_trial"`, `"escape_interval"`, or `"trial_to_onset"`. |
| `use_rigid_rotation` | bool | `false` | Rigid-body rotation mode: accumulate dx/dy linearly, then apply total yaw as a single rotation. Takes precedence over per-frame heading when `true`. |
| `use_angular_velocity_offset` | bool | `false` | Refine escape interval offset to the first angular-velocity zero-crossing after its peak. |

### Two Rotation Modes

**Non-rigid** (`use_rigid_rotation: false`):
Per-frame dynamic heading integration. Each frame's dx/dy is rotated by the accumulated dz up to that point. Preserves local curvature and S-turns.

**Rigid** (`use_rigid_rotation: true`):
Straight-line accumulation of dx/dy, then a single rigid-body rotation by `sum(dz) / RADIUS_MM`. Produces fan-shaped dispersion without per-frame curvature.

## Key Modules

### `pipeline/kinematics.py`
Pure physics layer. Integrates body-frame dx/dy/dz into trajectories with Savitzky-Golay smoothing, computes speed (mm/s) and angular velocity (rad/s), extracts escape latency and interval via backward-search threshold crossing.

### `pipeline/classifier.py`
Ternary state classifier. Priority order: (1) NoResponse if no valid burst, (2) PreWalk if pre-stimulus activity exceeds threshold, (3) Escape if baseline is quiescent. Adds `response_type`, `v_max`, `latency_ms`, `interval_onset_ms`, `interval_offset_ms` columns.

### `pipeline/visualization.py`
Publication-grade plotting (Nature/Science/Cell style). Includes trajectory overlay, speed kinetics, spaghetti plots, behavior probability, habituation curves, V_max distribution, and the two-stage trajectory integration algorithm (local heading + curvature-thresholded rigid rotation).

### `pipeline/mcmc.py`
Bayesian psychophysics via PyMC/NumPyro. Fits psychometric sigmoid functions to escape probability vs. TTC, tests multisensory integration hypotheses (ROPE-based posterior probability), computes Bayesian optimal integration (variance reduction), and performs survival analysis (Kaplan-Meier, Race Model Inequality).

## Output Structure

```
figures/
└── <subject>/
    ├── response/       trial_<N>_escape.svg
    ├── prewalk/        trial_<N>_prewalk.svg
    └── no_response/    trial_<N>_noresponse.svg
```

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Array computation, DataFrame manipulation |
| `matplotlib`, `scipy` | Figure generation, Savitzky-Golay smoothing |
| `pyyaml` | `config.yaml` parsing |
| `pymc`, `arviz` | Bayesian MCMC model specification and diagnostics |
| `numpyro`, `jax`, `jaxlib` | JAX-based NUTS sampler (10–100× faster, optional) |
| `lifelines` | Kaplan-Meier survival analysis |

## License

[Add license here]

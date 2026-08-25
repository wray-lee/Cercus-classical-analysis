# CLAUDE.md — Cercus CLI

## What this repo is
Cricket escape-response behavioral analysis pipeline (`README.md`). Raw kinematics CSVs → ternary classification (Escape / PreWalk / NoResponse) → publication-grade SVG figures + population CSV + Bayesian MCMC psychophysics (PyMC/NumPyro). Entry is `python -m cercus.cli.app <command>` (Typer) or the legacy root scripts.

## ENV_CONSTRAINT (MUST follow; no native-python, no default terminal)
Any command that runs code / tests / training MUST run inside WSL via this atomic chain:
```
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <python command>"
```
- `-i` is required or `openconda` alias won't load.
- Never split the chain; keep it atomic.
- All paths must be WSL-compatible: `D:\x` → `/mnt/d/x`.
- Source: `.claude/skills/orchestrator/SKILL.md`; `batsh.ps1` (native `python .\main.py ...`) is a legacy Windows-only runner and conflicts with this rule.

## Git Identity & Commit Boundary (CRITICAL)
- **Author / Committer**: MUST be `wray-lee <i@wray7.top>` (verified GitHub primary email).
- **Rule**: Never override local `.git/config` with secondary or unlinked emails (e.g. `wray.lee@outlook.com`).
- **Verification**: Always ensure `git config user.email` returns `i@wray7.top` before committing.

## Entry points
| Typer command | Legacy script |
|---|---|
| `cercus.cli.app single` | `main.py` |
| `cercus.cli.app population` | `population_analysis.py` |
| `cercus.cli.app mcmc` | `mcmc_analysis.py` |
| `cercus.cli.app trial-panels` | `plot_trial_panels.py` |
| `cercus.cli.app trajectories` | `plot_all_trajectories_fixed.py` |

Example:
```
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && python -m cercus.cli.app single --input /mnt/d/data --output /mnt/d/figures"
```
CLI flags live in `cercus/cli/app.py`; each legacy script has its own argparse (`--input-dir`, `--output`, `--binary-mode` (choices `escape_only|escape_prewalk`), `--individual-checks`, …). `main.py`/`cercus.cli.app single` also accept `--control-type` / `--stim-type`.

## Input data format
Directory of paired CSVs auto-discovered by `pipeline/io.py::scan_and_pair_sessions` (regex `{subject}_session_{N}_events.csv` / `{subject}_session_{N}_kinematics.csv`). Events CSV needs `event_name, timestamp, global_trial_id, details`; kinematics CSV needs `sys_time, dx, dy, dz, stim_state, global_trial_id`. Sessions are paired by `(subject, session_id)`, concatenated, and remapped to a continuous `global_trial_index`. Legacy merged CSVs can be converted with `test/old/converterOld.py`.

## Pipeline flow
`pipeline/io.py` (load+pair) → `pipeline/kinematics.py::preprocess` (per-trial TTC alignment + Savitzky-Golay smoothing + speed/angular-velocity) → `pipeline/classifier.py::label_trials` (adds `response_type`, `v_max`, `latency_ms`, `escape_interval_ms`, `interval_onset_ms`, `interval_offset_ms`) → plotting/export.

**Performance**: `population_analysis.py` uses multiprocessing for parallel subject processing and visualization rendering. Default: all CPU cores; override with `--workers N`. Each subject's load→preprocess→classify runs independently; visualization jobs (15 plots) render in parallel after data aggregation.

Key time-axis logic in `pipeline/kinematics.py::preprocess`:
- multimodal (`looming_wind*`) → `t_rel=0` at TTC (`wind_onset - target_ttc_ms/1000`)
- visual-only/looming → TTC anchor or theoretical `lv/(1−sinθ)`
- wind-only → `stim_state` onset; else trial midpoint.

Classification (see `Standardized Criteria.md` and `pipeline/classifier.py::classify_trial`):
- Priority: **NoResponse > PreWalk > Escape**. No burst (window `v_max` ≤ `ESCAPE_VMAX_THRESHOLD`) ⇒ NoResponse absolute veto. The threshold comes from YAML (`thresholds.yaml` / `config.yaml`, default **98 mm/s**) — the `50 mm/s` appearing in old docstrings is stale.
- Escape latency = backward search: locate the first frame exceeding `ESCAPE_VMAX_THRESHOLD` (98 mm/s), then scan backwards for the last frame below `ESCAPE_START_THRESHOLD` (10 mm/s).
- `baseline_visual` trials use the whole stimulus period (`t_rel ≤ 0`), not `[onset, +250ms]`; PreWalk anchor differs per paradigm (wind → wind onset; pure looming → escape onset).

## Package layout & refactor status
- `cercus/` — refactored target. Subpackages: `analysis` (individual-level circular-stat robustness), `classification` (empty, future strategy-based classifier), `cli` (Typer app), `config` (YAML → attribute-access config proxy + Pydantic validation), `constants` (YAML-backed constants), `core` (domain models + pure-physics kinematics), `visualization` (publication plots).
- `pipeline/` — legacy wrappers, still functional. `pipeline/visualization.py` and `pipeline/constants.py` are **DEPRECATED** (emit `DeprecationWarning` / marked in docstring) and re-export from `cercus.*`. Prefer new imports: `from cercus.visualization import plot_*`, `from cercus.core.domain import Trial, ResponseType`, `from cercus.config import get_config`.
- `cercus/core/kinematics/{latency,velocity,smoothing,trajectory_integration}.py` are pure physics — **no matplotlib imports allowed** (enforced by convention; `latency.py` / `trajectory_integration.py` docstrings explicitly declare "No matplotlib", the others are pure functions with no matplotlib import).
- `cercus/core/kinematics/trajectory_integration.py` holds the dual-stage trajectory integration: Stage 1 per-frame dz heading (preserves S-turns), Stage 2 rigid macro rotation (fan dispersion). `_core.py::compute_trajectory_masks` is the shared per-trial mask/builder used by all trajectory and angle plots.
- `cercus/visualization/` modules: `trajectories`, `kinetics`, `heatmaps`, `vmax`, `behavior`, `polar`, `individual`, plus `_core` (drawing helpers) and `_circstats` (Rayleigh, Watson-Williams, Wallraff).

## Config system — YAML, not hardcode (hard rule)
All tunable values live in YAML and load through `cercus/config/__init__.py::ConfigManager` (singleton, auto-reload via file watcher, attribute access `config.thresholds.ESCAPE_VMAX_THRESHOLD`):
- Defaults: `cercus/config/defaults/*.yaml` (`thresholds`, `geometry`, `colors`, `trajectory`, `visualization`, `analysis`).
- User overrides: root `config.yaml`, merged **by section** over the defaults (a whole sub-section replaces its default; don't override only part of a sub-key set and expect siblings to survive).
- Access: `get_config()`, `reload_config()`, `get_thresholds()`, `get_geometry()`, `get_colors()`, `get_analysis()`; `cercus.constants.*` constants are themselves YAML-backed.
- `cercus/config/settings.py` provides Pydantic `TrajectoryConfig` with cross-field validation (rigid-rotation/dz-range interplay, escape-onset-heading requires z-degree) and `BarLabelStyle` enum.

⚠ Known issue: root `config.yaml` line 30 has `use_escape_onset_heading: fales` (typo). YAML parses it as the string `"fales"`, and `bool("fales")` is `True` (see `pipeline/constants.py` / `settings.py::from_dict`), so it silently resolves to `true`. Fix the spelling if `false` is intended.

## Testing
```
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && pytest tests/test_classifier_golden.py tests/test_visualization_regression.py -v"
```
- `tests/test_classifier_golden.py` — 10 synthetic trials, bitwise classification expectations (Escape/PreWalk/NoResponse incl. `baseline_visual` and early-wind cases).
- `tests/test_visualization_regression.py` — hash-based pixel regression vs `tests/baselines/*.sha256`. Missing baseline = auto-written on first run.
- Updating baselines: `rm tests/baselines/*.sha256` then re-run pytest (the error message's `--update-baselines` flag does **not** exist). Windows note: `population_analysis.py`/`cercus/analysis/individual.py` use `_safe_savefig` (unlink-then-save) to dodge OSError 22.

## Code conventions
- Section separators use box-drawing `═`/`─`; comments are bilingual (Chinese technical notes common); log via `logging`, INFO level.
- New code goes in `cercus/`; keep `pipeline/` imports only for legacy-compat at call sites.
- Config values must come from YAML — never hardcode thresholds/colors/geometry in Python.
- Circular stats (`cercus/visualization/_circstats.py`) only: means via `circ_mean_rad`, never linear means on angles.

## Common workflows
1. Single subject: `python -m cercus.cli.app single --input <dir> --output <dir>` → per-subject folder with `response/`, `prewalk/`, `no_response/`, `individual_escapes*/`, behavior/habituation/vmax SVGs + `{subject}_summary_metrics.csv`.
2. Population batch: `python -m cercus.cli.app population --input <dir> --output <dir> [--individual-checks]` → `population_summary.csv`, `subject_escape_rates.csv`, adaptive V_max threshold (KDE valley → log-GMM → fallback 120), population SVGs under `population/` (+`heatmap/`). `--individual-checks` runs per-animal circular robustness (`cercus/analysis/individual.py`) → `suppl_*.png` + `individual_summary.csv`.
3. MCMC: `python -m cercus.cli.app mcmc --input <dir> --output results --binary-mode escape_only|escape_prewalk` → psychometric sigmoid fitting, TTC50/k posteriors, ROPE tests, variance reduction, KM survival, race-model bound; writes `mcmc_summary.json` + SVGs.
4. Refactor/feature work: use the repo-registered `orchestrator` skill (`.claude/skills/orchestrator/SKILL.md`) — NSMoR dev→double-blind review→test→commit closed loop, enforces the WSL chain. MCMC-specific task template: `.claude/prompt/mcmc.md`. Global skills (ponytail family, nsmor-* loops) are available but live outside the repo.

## To-fill / gaps
- `README.md` License section: `[Add license here]` — still empty.
- `cercus/classification/` is a stub (`__init__.py` only) — classifier lives in `pipeline/classifier.py`.

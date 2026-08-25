# Project Boundaries & Constraints

## 1. Git Identity & Commit Boundary
- **Git Author & Committer**:
  - `user.name`: `wray-lee`
  - `user.email`: `i@wray7.top` (Must match verified GitHub primary address to guarantee avatar and contribution graph association)
- **Constraint**:
  - Do NOT use or configure `wray.lee@outlook.com` or other secondary unverified emails in local `.git/config`.
  - Check before committing: `git config user.email` must resolve to `i@wray7.top`.

## 2. Environment & Execution Boundary
- **OS / Subsystem**: Code execution, tests, and pipeline scripts MUST run in WSL environment:
  ```bash
  wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <command>"
  ```
- **Path Mapping**: Always map Windows paths (`D:\...`) to WSL mount paths (`/mnt/d/...`).

## 3. Configuration & Code Boundary
- **YAML-driven Configuration**: All tunable thresholds, geometry parameters, trajectory parameters, and plot aesthetics must reside in `config.yaml` or `cercus/config/defaults/*.yaml`. Never hardcode numerical parameters in analysis/plotting code.
- **Pure Physics Kinematics**: `cercus/core/kinematics/` modules are pure computation without `matplotlib` or UI dependencies.
- **Repository Scope**: This repository (`Cercus-classical-analysis` / `D:\Projects\Cercus-cli`) is dedicated to data analysis and visualization. Acquisition/experiment hardware code belongs in `Desktop/Cercus`.

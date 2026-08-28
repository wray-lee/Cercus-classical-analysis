# Cercus Framework — Product Requirements Document (PRD)

> **Version**: 1.0  
> **Date**: 2025-07  
> **Owner**: wray-lee  
> **Status**: Living Document  

---

## 1. Executive Summary

Cercus 是一个面向蟋蟀逃跑反应行为研究的完整数据分析框架。从原始运动学 CSV 数据出发，经过自动化管线处理，产出出版级（Nature/Science/Cell 风格）可视化图表与贝叶斯统计推断结果。系统面向神经行为学研究者，消除手动分析中的主观偏差，以可复现的自动化工作流取代逐试次人工判定。

---

## 2. Problem Statement

蟋蟀尾须（cercus）感觉系统的逃跑反应研究涉及以下难题：

1. **人工分类主观性**: 需由算法统一判定逃跑（Escape）/ 预行走（PreWalk）/ 无响应（NoResponse）。
2. **阈值自适应**: 不同个体间的速度分布差异巨大，需要动态计算而非人为固定阈值。
3. **高通量批处理**: 多被试 × 多 Session × 多 Trial 的数据规模需并行处理。
4. **出版级图表**: 需符合 Nature/Science 投稿标准的一致化图表样式。
5. **贝叶斯推断**: 多感觉整合假设（视觉 vs 风 vs 双模态）需通过 MCMC 后验分析验证。

---

## 3. Target Users

| Persona | Description | Primary Use |
|---------|-------------|-------------|
| **神经行为学 PI** | 实验室负责人，需快速查看群体统计结果 | Population/MCMC 报告 |
| **研究生/博后** | 执行日常分析、调参、出图 | CLI 全链路调用 |
| **合作者/审稿人** | 验证方法可复现性 | Config YAML + Git 版本 |

---

## 4. Functional Requirements

### 4.1 数据摄入 (Data Ingestion)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | 自动发现输入目录下所有 `{subject}_session_{N}_kinematics.csv` 和 `{subject}_session_{N}_events.csv` 配对文件 | P0 |
| FR-1.2 | 支持多被试多 Session 自动聚合 | P0 |
| FR-1.3 | Events CSV 的 `details` 列中 JSON 解析刺激参数（TTC、角度等） | P0 |
| FR-1.4 | 对缺失列、空文件等异常数据给出明确错误信息 | P1 |

### 4.2 运动学计算 (Kinematics Processing)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | 从 `dx/dy/dz` 采样值计算瞬时速度标量序列 | P0 |
| FR-2.2 | Savitzky-Golay 平滑滤波（可配置窗口与阶数） | P0 |
| FR-2.3 | 角速度计算（`dz / dt`，用于轨迹积分与精细化潜伏期检测） | P0 |
| FR-2.4 | 双阶段轨迹积分（Stage 1: 逐帧航向积分；Stage 2: 刚体旋转） | P0 |
| FR-2.5 | 后向搜索逃跑潜伏期检测（从 50 mm/s 锚点回溯至 10 mm/s 穿越） | P0 |
| FR-2.6 | 可选角速度零点精细化（`use_angular_onset_refinement`） | P1 |

### 4.3 三元分类器 (Ternary Classifier)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | 分类优先级: NoResponse > PreWalk > Escape | P0 |
| FR-3.2 | 爆发检测窗口: 标准刺激 [onset, onset+250ms]; `baseline_visual` 全刺激期 | P0 |
| FR-3.3 | 预行走检测: 刺激前 1s 窗口内速度 > 10 mm/s | P0 |
| FR-3.4 | 输出列: `response_type`, `v_max`, `latency_ms`, `interval_onset_ms`, `interval_offset_ms` | P0 |
| FR-3.5 | 10 组合成黄金试次的位级回归测试通过 | P0 |

### 4.4 自适应阈值系统 (Adaptive Threshold)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | 四级优先级级联: KDE Valley → Log-GMM(3) → IQR-GMM(2) → Hardcoded(120) | P0 |
| FR-4.2 | 所有候选阈值方法均在全量试次上计算（不按分类预过滤） | P0 |
| FR-4.3 | 可视化诊断图（GMM 拟合线 + 阈值标记）随结果一同输出 | P0 |

### 4.5 可视化输出 (Visualization)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | 出版级 SVG 输出: Lancet/NPG 色彩方案、Nature-style 无衬线标签 | P0 |
| FR-5.2 | 轨迹叠加图: 按分类着色、扇形展布 | P0 |
| FR-5.3 | 速度动力学曲线: Mean ± SEM by response type, aligned to TTC | P0 |
| FR-5.4 | 起始对齐热图 (Onset-aligned Heatmap): 试次密度可视化 | P0 |
| FR-5.5 | 极坐标玫瑰图: 逃跑方向分布 + Rayleigh 检验 p 值标注 | P0 |
| FR-5.6 | V_max 分布直方图 + KDE + GMM 阈值线 | P0 |
| FR-5.7 | 习惯化曲线: 逐试次 V_max 时间演变 | P1 |
| FR-5.8 | SVG 像素哈希回归测试防漂移 | P0 |

### 4.6 贝叶斯 MCMC 推断 (Bayesian Inference)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | 心理物理 sigmoid 拟合 (Escape probability vs TTC) | P0 |
| FR-6.2 | 多感觉整合假设检验 (ROPE-based posterior) | P0 |
| FR-6.3 | 贝叶斯最优整合 (方差缩减) 计算 | P0 |
| FR-6.4 | Kaplan-Meier 生存分析 + Race Model Inequality | P1 |
| FR-6.5 | 支持 PyMC/NumPyro 双后端 (NumPyro 10-100× 加速) | P1 |
| FR-6.6 | `--binary-mode`: `escape_only` (default) / `escape_prewalk` | P1 |

### 4.7 CLI 与批处理 (CLI & Batch)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-7.1 | Typer 统一命令行入口: `single`, `population`, `mcmc`, `trial-panels`, `trajectories`, `calibrate` | P0 |
| FR-7.2 | `--workers N` 多进程并行（默认全核心，单线程回退 `--workers 1`） | P0 |
| FR-7.3 | Legacy 脚本入口（`main.py`, `population_analysis.py` 等）保持兼容 | P1 |

---

## 5. Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | 所有 Python 执行必须在 WSL + Conda `torch` 环境中运行 | Environment |
| NFR-2 | 数值参数禁止硬编码，必须通过 YAML 配置驱动 | Maintainability |
| NFR-3 | `cercus/core/kinematics/` 零 matplotlib 依赖（纯计算模块） | Architecture |
| NFR-4 | Population 批处理在 16 核机器上实现近线性加速 | Performance |
| NFR-5 | 分类器黄金测试与图表哈希回归测试在每次提交前必须通过 | Quality |
| NFR-6 | Git 提交使用 Conventional Commits 格式 (`feat(scope):` / `fix(scope):`) | Process |
| NFR-7 | 旧 import path (`pipeline.visualization.*`) 保持可用但发出 `DeprecationWarning` | Compatibility |

---

## 6. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               INPUT                                             │
│  {subject}_session_{N}_kinematics.csv  +  {subject}_session_{N}_events.csv      │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  pipeline/io.py: load_session_data() → DataFrame merge (kinematics + events)     │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  pipeline/kinematics.py: compute speed, angular velocity, Savitzky-Golay smooth  │
│  cercus/core/kinematics/latency.py: escape latency backward-search               │
│  cercus/core/kinematics/trajectory_integration.py: dual-stage integration        │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  population_analysis.py: adaptive_threshold_cascade() → V_max threshold          │
│  [KDE Valley → Log-GMM(3) → IQR-GMM(2) → Fallback(120)]                         │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  pipeline/classifier.py: classify_trial() → Escape / PreWalk / NoResponse        │
└──────────────┬───────────────────────────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────────────────────────────────────────────────────┐
│  CSV Export │  │  cercus/visualization/*: SVG publication figures               │
│  (summary)  │  │  - trajectories, kinetics, heatmaps, polar, vmax, behavior    │
└─────────────┘  └──────────────────────────────────────────────────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  pipeline/mcmc.py │ (Optional)
               │  Bayesian MCMC    │
               └───────────────────┘
```

---

## 7. Configuration Schema

### 7.1 参数治理分层

| Layer | Location | Scope |
|-------|----------|-------|
| Framework Defaults | `cercus/config/defaults/*.yaml` | 全局基线值 |
| User Override | `config.yaml` (project root) | 覆盖默认值的局部实验参数 |
| Runtime Validation | `cercus.config.settings.TrajectoryConfig` (Pydantic) | 强类型跨字段校验 |

### 7.2 关键可配置参数

| Domain | Parameters | File |
|--------|-----------|------|
| 逃跑检测阈值 | `vmax_threshold`, `start_threshold`, `prewalk_speed_threshold`, `use_angular_onset_refinement` | `thresholds.yaml` |
| 领地几何 | `RADIUS_MM`, `SAMPLE_RATE`, speed window | `geometry.yaml` |
| 轨迹渲染 | `use_z_degree_to_draw`, `use_rigid_rotation`, `dz_integration_range`, `use_escape_onset_heading` | `trajectory.yaml` |
| 色彩方案 | Escape/PreWalk/NoResponse 颜色, Left/Right, NPG palette | `colors.yaml` |
| 图表样式 | `bar_label_style`, axis formatting | `visualization.yaml` |
| 跨范式比较 | `peak_bracket` 预设、跨条件对比参数 | `comparison.yaml` |
| 分析参数 | Smoothing window, Savitzky-Golay 配置 | `analysis.yaml` |

---

## 8. Module Dependency Map

```
cercus.cli.app (Typer)
    ├── main.py → pipeline/io → pipeline/kinematics → pipeline/classifier
    │                                                    → cercus/visualization/*
    ├── population_analysis.py → [same] + multiprocessing.Pool
    ├── mcmc_analysis.py → pipeline/mcmc (PyMC / NumPyro)
    ├── plot_trial_panels.py → cercus/visualization/* + multiprocessing
    └── plot_all_trajectories_fixed.py → cercus/visualization/trajectories

cercus/config/__init__.py
    └── loads cercus/config/defaults/*.yaml + root config.yaml
    └── exposes: get_config(), reload_config(), get_thresholds(), get_geometry(), get_colors()

cercus/core/kinematics/ (ZERO matplotlib dependency)
    ├── latency.py: find_escape_latency(), compute_escape_interval()
    └── trajectory_integration.py: integrate_body_trajectory()

cercus/visualization/ (publication-grade SVG output)
    ├── _core.py: shared drawing helpers (grid, arrows, oscilloscope)
    ├── _circstats.py: Rayleigh test, Watson-Williams, circular mean
    ├── trajectories.py: overlay & unified global trajectory
    ├── kinetics.py: speed curves, spaghetti, population mean±SEM
    ├── heatmaps.py: onset-aligned density, trial-stacked waterfall
    ├── vmax.py: V_max histogram + KDE + GMM threshold
    ├── behavior.py: probability bars, habituation curves
    ├── polar.py: direction rose with Rayleigh p-value
    └── individual.py: per-trial composite panel rendering

cercus/constants/ (thin loaders from YAML)
    ├── thresholds.py: ESCAPE_VMAX_THRESHOLD, START_THRESHOLD, ...
    ├── geometry.py: RADIUS_MM, SAMPLE_RATE, ...
    └── colors.py: COLOR_ESCAPE, COLOR_PREWALK, ...

cercus/analysis/ (analysis utilities)
    └── __init__.py
```

---

## 9. Quality Assurance Strategy

| Layer | Method | Automation |
|-------|--------|------------|
| Unit/Golden | `test_classifier_golden.py` — 10 synthetic boundary trials | `pytest` |
| Visual Regression | `test_visualization_regression.py` — SHA-256 pixel hash | `pytest` |
| Integration | Full pipeline smoke test via CLI (`cercus.cli.app single`) | Manual / CI |
| Code Review | NSMoR Orchestrator dual-blind review (A + B must both accept) | Agent-automated |
| Environment | WSL + Conda `torch` sandbox enforced for all executions | BOUNDARY.md |

---

## 10. Roadmap & Future Considerations

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Core pipeline (IO → Classify → Visualize) | ✅ Complete |
| Phase 2 | Unified CLI (Typer) + Multiprocessing | ✅ Complete |
| Phase 3 | MCMC Bayesian inference (PyMC + NumPyro) | ✅ Complete |
| Phase 4 | Refactored package (`cercus/`) + YAML config | ✅ Complete |
| Phase 5 | CI/CD pipeline (GitHub Actions, automated regression) | 🔲 Planned |
| Phase 6 | Real-time acquisition integration (bridge to `Desktop/Cercus`) | 🔲 Future |

---

## 11. Constraints & Boundaries

1. **Repository scope**: 纯数据分析与可视化。硬件采集代码在 `Desktop/Cercus` 仓库中。
2. **Git identity**: `user.email = i@wray7.top`（已验证 GitHub 主邮箱）。
3. **No hardcoded numerics**: 所有物理参数必须从 YAML 加载。
4. **Pure kinematics**: `cercus/core/kinematics/` 不允许导入任何可视化库。
5. **Backward compatibility**: `pipeline.*` 旧 import 保持可用（发出 DeprecationWarning）。

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| Classifier accuracy vs expert annotation | ≥ 95% agreement |
| Population batch speedup (16 cores) | ≥ 10× vs single-threaded |
| Visual regression false-positive rate | 0% (hash-stable across runs) |
| MCMC convergence (R-hat) | < 1.01 for all parameters |
| Config change → figure update turnaround | < 60s for single subject |

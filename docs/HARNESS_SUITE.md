# Cercus-cli Full 5-Layer Harness Suite 工程文档

本文档定义 Cercus 蟋蟀逃跑行为分析系统的 **Full 5-Layer Harness Suite** 架构标准与测试/工程治理规范。

---

## 架构概览 (System Architecture)

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 5: CLI & Subagent Orchestration (Typer CLI + NSMoR Agent)       │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Quality & Verification Harness (Golden + SVG Hash Tests)     │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Cascading Config & Governance (YAML Defaults + Pydantic)     │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Classification & Analysis Pipeline (Ternary Classifier)      │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Domain Models & Biophysical Schema (Kinematics & Geometry)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: 领域模型与生物物理约束层 (Domain Models & Biophysical Schema)

### 1.1 核心数据实体 (Entities & Value Objects)
- **Kinematics Series (`dx`, `dy`, `dz`, `sys_time`)**: 采样运动学元数据。
- **Trial & Session**: 包含 `global_trial_id`、刺激类型（`wind` / `visual` / `bimodal`）及相依事件。
- **ResponseCategory (三元分类状态)**:
  - `Escape` (逃逸): 静止基线后的爆发性逃跑。
  - `PreWalk` (预行走): 刺激前 1s 内已有运动活动 ($V_{pre} > 10\text{ mm/s}$)。
  - `NoResponse` (无响应): 刺激后窗口内最大速度未达阈值 ($V_{max} \le GMM_{thresh}$)。

### 1.2 物理不变量与算式 (Physical Invariants)
- **速度标量**: $V(t) = \frac{\sqrt{\Delta x_t^2 + \Delta y_t^2}}{\Delta t}$
- **潜伏期起点 (Escape Latency)**: 速度首次突破 $10\text{ mm/s}$（前向回溯自 $50\text{ mm/s}$ 锚点），可由角速度峰值前的零过点进一步精细修正 (`use_angular_onset_refinement`)。
- **双阶段轨迹积分算法 (Dual-Stage Trajectory Integration)**:
  - **Stage 1 (局部曲率)**: 逐帧 dz 积分，保留 S 型弯曲与局部微形变。
  - **Stage 2 (刚体旋转)**: 根据主Impulse（如 `peak_bracket`）计算全局刚体旋转偏转角，维持扇形逃逸角散度。

---

## Layer 2: 分析管线与分类状态机层 (Execution Pipeline)

### 2.1 状态机判定优先级 (Classifier Cascade Rule)
分类严格遵循优先路由逻辑：
$$\text{NoResponse} \succ \text{PreWalk} \succ \text{Escape}$$

1. **爆破检测窗口 (Burst Window)**:
   - 标准刺激（Wind / Bimodal）: $[t_{onset}, t_{onset} + 250\text{ ms}]$。
   - `baseline_visual` 刺激: 刺激整个持续期间 ($t_{rel} \le 0$)。
2. **基线静止判定 (Pre-stimulus Baseline Check)**: 刺激前 1s 内速度不高于 $10\text{ mm/s}$。

### 2.2 四级自适应 $V_{max}$ 动态阈值级联 (Adaptive Threshold Cascade)
1. **KDE Valley (核密度估计凹谷法)**: `bw_method=0.3`，在第 1 与第 2 峰值间寻找极小值点。
2. **Log-GMM (3-Component Log 高斯混合模型)**: 对全量试次速度取 $\log$，拟合 3 分量 GMM。
3. **IQR-GMM (截断 IQR 2-Component GMM)**: 过滤上限 $Q3 + 1.5 \times \text{IQR}$ 异常值后拟合 2 分量 GMM。
4. **Hardcoded Fallback**: 若前置拟合均失败，使用默认回退值 $120.0\text{ mm/s}$。

---

## Layer 3: 配置与参数治理层 (Cascading Config & Governance)

### 3.1 配置分层模型 (Configuration Hierarchy)
```
cercus/config/defaults/
├── thresholds.yaml       # 物理阈值 (Escape / PreWalk)
├── geometry.yaml         # 领地几何参数 (RADIUS_MM)
├── colors.yaml           # 色彩规范 (Lancet / NPG)
├── trajectory.yaml       # 轨迹渲染预设
├── visualization.yaml    # 图表样式与视觉参数
├── comparison.yaml       # 跨范式比较预设 (peak_bracket 等)
└── analysis.yaml         # 分析参数 (smoothing, window)
       ↓ Override
根目录 config.yaml         # 用户局部覆盖
       ↓ Validation
cercus.config.settings    # Pydantic TrajectoryConfig 运行时强类型校验
```

### 3.2 强校验约束规则 (Validation Rules)
- `use_escape_onset_heading = True` 必须依赖 `use_z_degree_to_draw = True`，否则引发 `ValueError`。
- `use_rigid_rotation = False` 时忽略 `dz_integration_range` 并发出警告。

---

## Layer 4: 质量校验与测试套件层 (Quality Verification Harness)

### 4.1 黄金测试套件 (Golden Classifier Suite)
- 文件: `tests/test_classifier_golden.py`
- 包含 10 组合成边界试次（边界极限速度、极短潜伏期、PreWalk 边缘数据等），进行位级准确性断言。

### 4.2 视觉回归验证 Harness (Visual Hash Regression Suite)
- 文件: `tests/test_visualization_regression.py`
- 对生成的渲染图表提取 SHA-256 图像像素哈希，防止非预期的图表渲染漂移。

### 4.3 特性回归测试 (Feature Regression Tests)
- `tests/test_peak_bracket.py` — `peak_bracket` dz 积分范围正确性验证。
- `tests/test_angular_onset_refinement.py` — 角速度零点精细化潜伏期修正验证。

### 4.4 物理环境执行强约束 (WSL Sandbox Rule)
所有 Python 代码执行与测试运行必须约束在指定的 Linux/WSL Conda 环境下：
```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <command>"
```

---

## Layer 5: 主控编排与 CLI 调度层 (CLI & Subagent Orchestrator)

### 5.1 Typer CLI 交互架构 (`cercus.cli.app`)
提供统一命令行入口，支持 `multiprocessing` 多进程并行计算：
```bash
python -m cercus.cli.app single --input <data> --output <figures>
python -m cercus.cli.app population --input <data> --output <results> --workers N
python -m cercus.cli.app mcmc --input <data> --output <results>
python -m cercus.cli.app trial-panels --input <data> --output <figures> --workers N
python -m cercus.cli.app trajectories --input <data> --output <out.svg>
python -m cercus.cli.app calibrate --input <data>
```

### 5.2 可视化模块内部架构 (Visualization Internals)
`cercus/visualization/` 内部分为公开 API 模块和私有辅助模块：
- **`_core.py`** (private): 共享绘图原语 — 网格、箭头、阈值线、示波器风格。
- **`_circstats.py`** (private): 纯数学圆统计 — Rayleigh test、Watson-Williams、circular mean/resultant。
- **`individual.py`** (public): 单试次复合面板渲染（速度曲线 + 轨迹 + 角速度，用于 `trial-panels` 命令）。

### 5.3 Agent 闭环重构主控 (NSMoR Orchestrator State Machine)
在 `.claude/skills/orchestrator/SKILL.md` 中定义的 Multi-Agent 状态机：
1. **`@nsmor_developer`**: 负责核心代码重构与修改。
2. **`@nsmor_reviewer_A` / `@nsmor_reviewer_B`**: 独立并行双盲审查，必须同时获得 `[is_accepted: TRUE]`。
3. **`@nsmor_tester`**: 触发 WSL 真实测试套件并自动提交 Conventional Commit。
4. **Watchdog**: 包含 5 分钟无响应断线重连与 10 分钟长耗时脚本心跳监测机制。

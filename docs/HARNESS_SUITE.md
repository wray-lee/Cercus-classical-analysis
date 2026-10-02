# Cercus 框架分层架构与工程治理规范 (Harness Suite)

本文档规范了 Cercus 分析系统的五层架构体系、核心生物物理模型与工程质量治理标准。

---

## 架构总览

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 5: CLI 与任务调度层 (Typer CLI + 多进程并行)                    │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 4: 质量保障与回归验证层 (Golden Tests + 像素哈希回归)           │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 3: 级联配置与参数治理层 (YAML Defaults + Pydantic 校验)         │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 2: 行为分类状态机与动力学分析层 (四元状态机 + 反应时间测量)     │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 1: 领域实体与生物物理计算层 (运动学平滑 + 轨迹积分)             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: 领域实体与生物物理计算层 (Domain & Kinematics)

### 1. 核心实体模型
- **Kinematics Series (`dx`, `dy`, `dz`, `sys_time`, `ard_time`)**：来自球形跑步机编码器的多通道高频运动采样序列。
- **Trial & Session**：承载实验元数据、刺激类型（纯风 `wind`、纯视觉 `visual`、多模态 `bimodal`）及对齐事件。
- **ResponseCategory (行为状态)**：
  - `Escape`（逃逸）：由静止基线在刺激诱发下启动的高速逃跑。
  - `PreEscape`（提前起跑）：起跑时刻早于气流刺激到达（主要见于视觉提前触发）。
  - `PreWalk`（刺激前行走）：受刺激瞬时机体处于行走状态。
  - `NoResponse`（无响应）：刺激后窗口内未达到爆发速度阈值。

### 2. 物理量解算与轨迹积分
- **瞬时速度标量**：
  $$v(t) = \frac{\sqrt{\Delta x_t^2 + \Delta y_t^2}}{\Delta t}$$
  结合自适应 Savitzky-Golay 滤波器滤除机械振动噪声。
- **潜伏期起点检测 (Escape Latency)**：
  自合格爆发峰值向前回溯，寻找速度降至 10 mm/s 以下的最后一个样本，其后一帧定为起跑点（可结合角速度零过点进一步精细修正）。
- **双阶段轨迹积分 (Dual-Stage Integration)**：
  - **Stage 1 (局部弯曲细节)**：逐帧根据角速度 dz 进行方向积分，保留逃逸过程中的避障弯折与局部 S 弯。
  - **Stage 2 (刚体旋转展开)**：提取主要转向冲量（如 `peak_bracket`）进行宏观旋转，呈现面向不同刺激角度时的扇形发散分布。
- **架构硬约束**：本层（`cercus/core/kinematics/`）为纯数学与物理计算，禁止引入任何 `matplotlib` 绘图依赖。

---

## Layer 2: 行为状态机与动力学分析层 (Classification & Dynamics)

### 1. 状态机判定级联
分类严格遵循单向优先级路由：
$$\text{NoResponse} \succ \text{PreEscape} \succ \text{PreWalk} \succ \text{Escape}$$

1. **响应爆发窗口 (Burst Window)**：
   - 气流相关刺激：$[t_{wind}, t_{wind} + 250\text{ ms}]$。
   - 纯视觉基线刺激：刺激持续全程直至理论碰撞时刻（$t \le \text{TTC}$）。
   - 若窗口内未达到爆发阈值，直接判定为 NoResponse（绝对否决）。
2. **提前逃逸判定 (PreEscape)**：起跑时刻 $t_e < t_{wind}$，有效分离视觉诱导的超前逃逸。
3. **前置行走判定 (PreWalk)**：刺激前 1 秒因果均速达标，且刺激到达瞬时保持运动。

### 2. 四级自适应爆发阈值级联 (Adaptive Threshold Cascade)
用于跨个体与群体统计时自适应获取分割爆发与慢速运动的最佳速度门槛：
1. **KDE Valley (核密度凹谷法)**：在核密度曲线的前两个峰值之间寻找主低谷极小值。
2. **Log-GMM (三组分对数高斯混合模型)**：压缩速度长尾分布后拟合三组分 GMM。
3. **IQR-GMM (两组分截断高斯混合模型)**：剔除四分位异常值后拟合双峰。
4. **Hardcoded Fallback**：前置拟合均未收敛时，使用稳健回退值 120 mm/s。

---

## Layer 3: 级联配置与参数治理层 (Config & Governance)

### 1. 分层配置体系
```text
cercus/config/defaults/
├── thresholds.yaml       # 物理速度与判定阈值
├── geometry.yaml         # 实验舱几何尺寸与时间窗口
├── colors.yaml           # Nature/Lancet 风格期刊配色
├── trajectory.yaml       # 轨迹渲染预设
├── visualization.yaml    # 视觉排版与标签样式
└── analysis.yaml         # 滤波窗口与平滑参数
       ↓ 覆盖合并
根目录 config.yaml         # 用户个性化参数覆盖
       ↓ 模式校验
cercus.config.settings    # Pydantic 强类型与交叉约束校验
```

### 2. 运行时校验规则
- 若设置 `use_escape_onset_heading = True`，必须同时满足 `use_z_degree_to_draw = True`，否则抛出 `ValueError`。
- 若 `use_rigid_rotation = False`，刚体积分区间 `dz_integration_range` 将自动失效并记录警告日志。

---

## Layer 4: 质量保障与回归验证层 (Quality & Verification)

### 1. 黄金测试套件 (Golden Classifier Suite)
- 位置：`tests/test_classifier_golden.py`
- 覆盖典型边界合成试次（极短潜伏期、风前加速、多模态超前起跑、边缘慢速走动等），确保状态机逻辑变更时不发生静默越界。

### 2. 图像哈希回归测试 (Visual Hash Regression)
- 位置：`tests/test_visualization_regression.py`
- 对图表输出提取图像像素级 SHA-256 哈希比对，防止样式改动导致视觉排版非预期漂移。

### 3. 环境执行约束
为保证计算与随机种子的跨平台一致性，代码运行与测试必须通过统一的 Linux/WSL 环境原子化执行：
```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <command>"
```

---

## Layer 5: CLI 与任务调度层 (CLI Orchestration)

基于 Typer 构建现代化命令行界面，并提供内置多进程加速：
```bash
# 单个体分析
python -m cercus.cli.app single --input <dir> --output <dir>

# 群体并行批处理
python -m cercus.cli.app population --input <dir> --output <dir> [--workers N]

# 跨范式全景对比
python -m cercus.cli.app full --input <parent_dir> --output <dir>
```
各子命令既可独立调用，亦可在自动化工作流中通过脚本无缝集成。

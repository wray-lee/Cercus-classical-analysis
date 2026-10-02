# Cercus 分析框架中文文档

> 英文主文档请参阅 [README.md](README.md) · 标准行为判据说明请参阅 [Standardized Criteria-cn.md](Standardized%20Criteria-cn.md)

Cercus 是面向蟋蟀尾须感觉系统与逃逸反应（Escape Response）的行为学分析框架。系统能够从原始高频运动学 CSV 数据出发，自动完成时间对齐、运动学积分、行为状态分类（Escape / PreEscape / PreWalk / NoResponse）、绘制出版级矢量图表（Nature/Science 风格），并通过分层贝叶斯模型（MCMC）进行群体心理物理学统计推断。

---

## 1. 快速上手

### 环境安装与依赖
```bash
pip install -r requirements.txt
```
*(注：深度学习与 MCMC 加速环境需在 WSL 内通过指定 Conda 环境运行。)*

### 统一 CLI 命令（基于 Typer）
```bash
# 单被试（个体）分析
python -m cercus.cli.app single --input path/to/data/ --output figures/

# 群体批处理分析（多被试并行汇总与自适应阈值）
python -m cercus.cli.app population --input path/to/data/ --output results/

# 贝叶斯 MCMC 心理物理学拟合
python -m cercus.cli.app mcmc --input path/to/data/ --output results/

# 绘制逐试次多通道面板图
python -m cercus.cli.app trial-panels --input path/to/data/ --output figures/

# 绘制全局聚合轨迹叠加图
python -m cercus.cli.app trajectories --input path/to/data/ --output trajectories.svg

# 跨范式全景模式（全局自适应阈值 + 哑铃对比图）
python -m cercus.cli.app full --input parent_data_dir/ --output results/
```

- **并行性能**：`population` 与 `trial-panels` 支持通过 `--workers N` 设置并发核心数（默认启用全部 CPU 核心；单线程排查可指定 `--workers 1`）。

---

## 2. 项目核心架构

```text
Cercus-cli/
├── cercus/                          # 核心业务模块
│   ├── visualization/               # 出版级图表绘制（轨迹、动力学、极坐标玫瑰图、热图等）
│   ├── core/kinematics/             # 纯物理计算模块（潜伏期回溯、双阶段航向与刚体旋转积分）
│   ├── config/                      # YAML 统一配置管理与 Pydantic 校验
│   ├── constants/                   # 物理阈值、几何尺寸与配色常量
│   └── cli/                         # Typer 统一命令行入口
├── pipeline/                        # 历史兼容与数据流管线
│   ├── io.py                        # 数据扫描与配对加载
│   ├── kinematics.py                # 滤波与时空对齐预处理
│   ├── classifier.py                # 四元行为状态分类器
│   └── mcmc.py                      # PyMC / NumPyro 分层贝叶斯拟合
├── config.yaml                      # 用户本地配置覆盖文件
└── tests/                           # 黄金测试与渲染哈希回归测试
```

---

## 3. 标准行为分类判据

分类器遵循单向绝对优先级级联，避免分类重叠与歧义：

$$\text{NoResponse} \longrightarrow \text{PreEscape} \longrightarrow \text{PreWalk} \longrightarrow \text{Escape}$$

```
                          ┌── [无合格爆发] ──────────────────────────► NoResponse
                          │
[刺激窗口内检测爆发] ──────┼── [爆发合格 ∧ 起跑时刻早于刺激到达] ─────► PreEscape
                          │
                          ├── [爆发合格 ∧ 刺激到达前 1 秒处于行走状态] ─► PreWalk
                          │
                          └── [其余合格爆发试次] ────────────────────► Escape
```

1. **响应爆发窗口（Burst Window）**：
   - 含风试次：在气流刺激到达后的 **250 ms** 窗口内，最大速度超过爆发阈值（默认 98 mm/s，常规分析配置为 50 mm/s）。
   - 纯视觉基线：在小球理论碰撞前（$t \le \text{TTC}$）最大速度达标。
   - 若窗口内未达阈值，直接判定为 **NoResponse**（具有绝对否决权）。

2. **提前起跑（PreEscape）**：
   - 爆发峰值仍在风后 250 ms 内，但从该峰值向前回溯找到的起跑点（速度降至 10 mm/s 以下的后一帧）位于风刺激到达前。
   - 在多模态实验中，这代表由视觉刺激提前诱发的真逃逸；在纯风实验中反映自发运动。

3. **刺激前行走（PreWalk）**：
   - 气流到达前 1000 ms 内，因果均速大于 10 mm/s 的持续时间占比达到 15% 以上，且到达瞬时速度亦大于 10 mm/s。
   - 刻画动物在刺激到来时正处于运动状态的试次。

4. **标准逃逸（Escape）**：
   - 刺激到达前处于静止基线，刺激后启动的典型逃逸反应。

---

## 4. 反应时间与运动学指标

- **逃逸起跑时间（`escape_reaction_time_ms`）**：机体开始加速起跑的时刻距离刺激到达的时间差。Escape 为正数，PreEscape 为负数（如实反映提前量）。
- **因果制动延迟 T1（`pause_stopping_time_ms`）**：PreWalk 试次中，刺激到达后速度首次跌破静止阈值的潜伏期（具备传感器量化噪声保护保护）。
- **停顿至再逃逸间隔 T2（`pause_to_escape_time_ms`）**：从停步到后续再次爆发逃逸的时间差。
- **综合因果反应时间 RTm（`pause_reaction_time_ms`）**：$\text{RTm} = \text{T1} + \text{T2}$。端点无法明确解析时如实记录为 NaN，绝不人为插值填充。
- **逃逸距离（`distance_mm`）**：统一从机体逃逸起跑点（Onset）开始对速度积分。

---

## 5. 配置与参数管理

所有阈值、几何尺寸与配色均维护在 `cercus/config/defaults/*.yaml` 中。
如需自定义参数，只需在根目录创建 `config.yaml` 覆盖特定字段，例如：

```yaml
escape:
  vmax_threshold: 50.0  # 修改爆发判定阈值

classification:
  use_preescape: true   # 启用 PreEscape 分类分支
```

---

## 6. 测试与质量保证

```bash
# 运行分类器 10 组黄金试次位级校验
pytest tests/test_classifier_golden.py -v

# 运行出版级图表像素哈希防漂移测试
pytest tests/test_visualization_regression.py -v
```

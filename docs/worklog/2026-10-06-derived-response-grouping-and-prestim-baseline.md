# 派生响应分组 + 独立刺激前基线

记录日期：2026-10-06。

## 派生分组：raw `response_type` 永不改写

分类器仍输出四个原始类别（`Escape` / `PreEscape` / `PreWalk` / `NoResponse`）。
新增**派生**列 `response_group`：`analysis.response_grouping.merge_prewalk=true`
（默认）时把 raw `PreWalk` 并入 `Escape`，`PreEscape` / `NoResponse` 保持独立；
`false` 时与 raw 标签一一对应。未知/缺失标签原样保留，绝不回退成 `NoResponse`。

- 开关必须是 YAML 布尔值；字符串（如 `"false"`）会被拒绝而非 `bool()` 强转。
- 派生逻辑集中在 `cercus/analysis/response_groups.py`，由 `label_trials` 末尾统一附加，
  覆盖 single / population / full / 导出。
- raw `response_type`、`RESPONSE_TYPES` / `BURST_CLASSES` / `ESCAPE_CLASSES` 不变。

### 陷阱：raw vs derived

主聚合视图（行为概率、群体行为概率、群体速度/spaghetti 动力学、密度热图、
**极坐标默认主面板**、**群体 V_max 响应注释**、范式 dumbbell 的响应率面板、
范式 RT/distance 主分组图）使用**派生**分组。以下仍是 **raw 诊断**，
不得并入派生分组：

- 逐 trial 目录路由、PreWalk stillness / movement、habituation 计数与**配色**；
- **逐类 V_max 分布**（`plot_vmax_distribution` 只取 raw `Escape`，不是主视图）；
- 显式传入类型的极坐标面板（`response_types=` 走 raw `response_type`）；
- MCMC 二值化与 KM 成员（`escape_only` / `escape_prewalk`）——语义不变；
- `individual` 队列与极坐标 Escape-vs-PreWalk 的 Watson-Williams / Wallraff；
- `select_escape_latency` 的 RT 成员判定：**先**在 raw class 上选取 endpoint，
  **再**派生显示分组（范式 by-class 与 RT/distance 主图都遵守此顺序）。
- 群体 `escape_rate` 仍基于 `is_valid_escape`；新增独立的 `escape_group_rate`。

## 独立刺激前基线（`prestim_*`）

在 `preprocess` 里对**原始全 session** 运动学测量「首个刺激前 1 s」的运动，附加
`prestim_reference_kind` / `prestim_reference_offset_ms` /
`prestim_reference_sample_offset_ms` / `prestim_status` / `prestim_moving_fraction` /
`prestim_reference_speed_mm_s`，覆盖所有类别（含 `NoResponse`）。绝不在
`label_trials` 的裁剪帧上计算。

参考点（`prestim_reference_kind`）：

- `visual_event_sample`：视觉 / 多感觉试次，用文档化的 `Looming` phase-transition
  事件（**首个**刺激）映射到该事件之前最近的可观测采样；缺事件 = `unobserved`，
  绝不回退到之后的 wind onset / trial start / TTC / escape。
- `calibrated_wind`：纯风试次，用观测到的首个二值阀帧 + `wind_arrival_delay_ms`。
- `unobserved`：无视觉且无风刺激，或历史不可测。

**注意**：BV（`baseline_visual`）与 MS（`looming_wind`）走 `visual_event_sample`；
BW（`baseline_wind`）走 `calibrated_wind`。MS 既有的 `pause_*` 诊断仍是**风锚定**，
与 `prestim_*` 相互独立，不改变。BV 的 legacy PreWalk 仍是 escape-onset 锚定的活动规则。

### 源时钟与采样参考的边界

- 宿主 `sys_time` / 事件时间戳是 HOST 秒，仅用于**定位**事件采样；`ard_time` 是采集
  毫秒，速度分母只用采集间隔（`hypot(dx,dy)/Δard_time`），不用宿主计时、不用裁剪帧的
  边界重置位移。
- 参考点用**映射后的行号**定位（不是对可能损坏的时钟做全局 `searchsorted`）；视觉参考
  取事件前最近可观测样本，映射超过一个采集 gap 视为远端 → `unobserved`。
- 历史限制在参考点周围的**局部连续采集段**：回退遇无效 / 非递增 / 超 gap 即停；可进入
  上一个 trial id，但绝不跨 session、绝不跨无效时间戳、绝不 bridge。
- 完整历史需要覆盖 1 s 窗口**加上**因果位移平均的前置 margin；不足即 `unobserved`
  （fraction = NaN，绝不当作 stationary）。
- 该基线与 `label_trials` 的裁剪无关，远端风与 `NoResponse` 同样获得真实基线。

## 保留（未改动）的 legacy 行为

- BV legacy PreWalk：escape-onset 锚定，非论文风群体。
- 显式 legacy MCMC 模式（`escape_only` / `escape_prewalk`）成员不变。
- 既有 raw 诊断列（`pause_*` / `stillness_*`）不变。

## 证据边界（当前实测，勿夸大）

- **完整 2520 trial**：33 个原有列（含分类 / timing / distance）相对**冻结前**结果
  逐 trial **精确一致**；`prestim_*` 是**新增**列，不在该对比范围内。
- **合并/拆分守恒**：`merge_prewalk` 开关下**仅 `response_group` 列变化**（trial 集合、
  raw 列、含基线在内的其余列全部一致）。
- **性能**：裁剪后的基线计算 18.4 s，对照修复前 16.2 s（同量级）。
- **独立重建审计**：最终实现抽样 216 trial（每范式一只动物），从原始 session
  重建完整连续采集段与参考点，复用既有底层因果平均与历史测量函数。全部状态与
  数值比对通过：fraction 绝对容差 1e-11；speed 绝对容差 1e-8 mm/s、相对容差
  1e-10。这验证了裁剪与参考映射，不是对共享底层算法的独立生理验证。早期完整段
  与裁剪段计算的 reference speed 最大差 6.48e-10 mm/s，来自累积和起点的浮点舍入。
- **全量 CLI**：6 范式 / 2520 trial 全成功，两张 SVG 均产出；`full_summary.csv`
  全精度与最终 replay 一致。
- **BV**：720 trial 的 CSV / rates / prestim summary 对照裁剪后 replay 校验通过；
  **workers=4 因 `Errno12` 内存分配失败**；workers=1 完成全 720 导出，
  **渲染器完成 7 张 SVG 后在后台 600 s 上限超时**——无内存回溯，**不**能声称
  workers=1 OOM。
- **BW**：3 张 preview 渲染成功。
- **不规则采样**：carry-forward 裁剪已修复，含 33 项聚焦检查（MS visual missing /
  drift / delay history / session boundaries）。
- **图像回归**：原始 raw Escape 的 V_max 诊断恢复原哈希；群体 V_max 的合并注释
  在合并/拆分成员检查通过后更新对应哈希。其余更新限于派生分组改变的主图。
- **真实数据预览**：BV 与六范式 RT/distance 图均渲染合并/拆分版本，图例和队列
  覆盖守恒；跨范式 footer 原有覆盖率/统计文字重叠已修复，布局参数放在 YAML。
  两种视图的 burst classifier N=1994、RT observed=1986、missing=8。
- **最终验证状态**：WSL 已恢复；完整测试套件 **356 passed**（22 条既有
  Matplotlib warning，18.16 s），402 个受保护文件未变。本 worklog 记录的是当前证据，
  不是论文级生理验证结论。

# SPEC: PreEscape 行为类 + RT/Distance 分析 + 热图风标识重构

状态: ready-for-agent · 分支: cli · 依据: CONTEXT.md (Phase -1 锁定)

## Problem Statement

多模态 (looming_wind) 数据中约 20–41% 的 PreWalk/边界 trial 实为**风前纯视觉触发的真逃逸**（起跑早于给风）。现三元分类把它们混进 Escape/PreWalk，污染了多感官整合的机制解释（教授组会意见），且 MCMC `escape_only` 拟合被不等风的 trial 拉偏。同时：刺激锚定 reaction time 与逃逸距离两项基础分析缺失；onset 热图上的金框/白框注释（13e1210）在 PreEscape 成类后失去存在意义。

## Solution

1. **PreEscape 第四行为类**：开关 `classification.use_preescape`（YAML 默认 true，false=一键回退今日三元行为）。规则：多模态 trial ∧ burst 成立 ∧ `interval_onset_ms < target_ttc_ms − 50` ⇒ PreEscape。优先级 NoResponse > PreEscape > PreWalk > Escape。
2. **RT/Distance 补齐**：`reaction_time_ms` = onset − 刺激锚点（wind: target_ttc；纯视觉: 0），PreEscape 为负=提前量；`distance_mm` = speed 在 [onset, offset] 梯形积分；`distance_500ms_mm`。全部进 summary CSV + 分布图。
3. **热图重构**：回滚金框/白框/★/图例（13e1210 的注释部分）；wind onset 标识——TTC 面板整根竖线、onset 面板每行青色刻度；PreEscape 自动获得面板。

## User Stories

- 作为研究者，我切换 `use_preescape: false` 即可复现旧三元结果，保证论文返修可比。
- 作为研究者，population summary CSV 每 trial 有 RT 与逃逸行程，可直接做组间统计。
- 作为研究者，热图上我能一眼指出风到达的时刻，判断逃逸窗内亮带先于还是后于风。
- 作为分析者，MCMC escape_only 自动只含风触发逃逸，无需手工过滤。

## Implementation Decisions

- 判定在 `classify_trial` 内（纯函数、可单测），不新增后处理脚本；PreWalk 检测之前插入，burst veto 之后。
- buffer 常量进 `thresholds.yaml::classification`（`use_preescape: true`, `preescape_buffer_ms: 50.0`），经 `cercus.constants` 导出，禁硬编码。
- RT/distance 计算放 `cercus/core/kinematics`（纯物理无 matplotlib，trapz 用 numpy），`label_trials` 汇出列；`export_summary_metrics` 增三列。
- `colors.yaml` 增 `preescape`（候选 `#E69F00` npg orange）；下游 viz 遍历 `response_types` 列表处由 3 元组改为读配置常量。
- MCMC：`escape_only` 集合自然收窄（response_type=="Escape"），`escape_prewalk` 保持；不改先验。
- 热图回滚=删除式改动，PreWalk 排序键退化为 `(mean_in)` 单键（prewind 标志随 PreEscape 出类而失去意义 → 排序按风窗均速即可）；wind onset 用 `target_ttc_ms`（TTC 面板）与逐行 `target_ttc − onset`（onset 面板）。
- NoResponse trial 无 onset ⇒ RT/distance = NaN（沿用现有 NaN 约定）。

## Testing Decisions

- 顶层 seam 不变：`label_trials` golden 测试——prewind-escape 用例开关双态各断言（true⇒PreEscape / false⇒今日标签）。
- `classify_trial` 边界单测：`onset == wind − 50`（不含）、`onset == wind − 49`（不算）、非 wind trial 永不 PreEscape。
- RT/distance 纯函数：合成阶梯速度曲线，梯形积分手算对照。
- 视觉：`tests/baselines/*.sha256` 全量刷新（默认 true 改变面板构成，不可避免），hash 回归流程不变。
- 全部执行走 WSL 链；`pytest tests/ -q` 为 VALIDATION_CMD。

## Out of Scope

- PreEscape 的 MCMC 专门建模（先只做类别与导出）。
- 朝向/净位移投影、到刺激源距离。
- `use_angular_onset_refinement` 交互细化（粗 onset 仍是锚，沿用现状）。
- README 三检验小节（已交付 c6290fb）。

## Further Notes

- 风险：下游对 `["Escape","PreWalk","NoResponse"]` 字面量的硬编码散布 ~9 文件；T4 统一收敛到单一常量源。
- 数据集基线事实：-373 30° PreWalk=65/20% prewind；-308 36° 112/41%——开关翻转后这些 trial 迁移，人口图 n 标注随动。

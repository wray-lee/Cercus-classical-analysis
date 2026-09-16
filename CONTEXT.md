# CONTEXT.md — Cercus CLI 领域词汇与事实

> Phase -1 (grill-with-docs) 产出。构建期间 builder 不得直接修改本文件（走 TERM_PROPOSAL）。

## 领域

蟋蟀（Gryllus） escape-response 行为分析：视觉逼近（looming, TTC）+ 风（wind puff）多感官刺激 → 三/四类行为判定 → 运动学统计 + SVG 图 + MCMC 心理物理。

## 时间轴（事实，勿混）

| 坐标系 | 原点 | 风到达 | 用途 |
|---|---|---|---|
| `t_rel`（ms） | 多模态：TTC=0（wind onset = `target_ttc_ms`，如 −373/−308，恒定 per dataset） | `target_ttc_ms` | 存储轴，PreWalk 窗、escape latency 都在这上面算 |
| onset 对齐（s） | `interval_onset_ms`（每 trial 自己的起跑点） | 逐行漂移：`(target_ttc_ms − interval_onset_ms)/1000` | 热图/动力学叠加显示 |

## 行为类（classifier 输出 `response_type`）

- **NoResponse**（绝对否决）：burst 窗内 v_max ≤ ESCAPE_VMAX_THRESHOLD(98)。
- **PreEscape**（新增，仅多模态 wind trial）：burst 成立且 `interval_onset_ms < target_ttc_ms − preescape_buffer_ms(50)` —— 风前纯视觉触发的真逃逸。优先级压在 PreWalk 前。
- **PreWalk**：风窗 `[wind−1000, wind−50]` 内 >10mm/s 帧占比 >15%（风前已在散步 → 风反应不可评估）。
- **Escape**：burst 成立且以上皆非（= 风后多感官触发逃逸）。
- 优先级链：**NoResponse > PreEscape > PreWalk > Escape**。
- 总开关 `classification.use_preescape`（YAML，默认 **true**）；false = 完全回退今日三元行为。

## 关键量（trial 级，进 summary CSV）

| 列 | 定义 | 状态 |
|---|---|---|
| `latency_ms` | t_rel 轴上起跑时刻（含角速度精修） | 已有 |
| `interval_onset_ms` / `interval_offset_ms` | 10mm/s 起止 | 已有 |
| `reaction_time_ms` | **刺激锚定 RT**：wind trial = onset − target_ttc；纯视觉 = onset − TTC(0)。PreEscape 为负=提前量 | 新增 |
| `distance_mm` | speed 在 [onset, offset] 梯形积分（一次逃逸行程） | 新增 |
| `distance_500ms_mm` | onset 后 500ms 行程 | 新增 |

## 热图标注（决策后形态）

- 回滚 13e1210 的金色框 / 白色刻度 / ★ / 图例（PreEscape 成类后不再需要逐行分类窗解释）。
- 保留 PreWalk 排序改进但改用 `response_type`（PreEscape 自己成面板）。
- **wind onset 标识**：TTC 对齐面板 = 整根竖线（同面板内 `target_ttc_ms` 恒定）；onset 对齐面板 = 每行青色小刻度。

## 硬规矩（事实）

- 一切可调值进 YAML（`cercus/config/defaults/*.yaml` + 根 `config.yaml` 按节合并），禁止 Python 硬编码阈值/颜色。
- 所有执行走 WSL 原子链：`wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <cmd>"`；路径 `/mnt/d/...`。
- 提交作者 `wray-lee <i@wray7.top>`；Conventional Commits。
- 圆统计只用 `_circstats`（Rayleigh/Watson-Williams/Wallraff），禁止线性均值于角度。
- 视觉回归基线：`tests/baselines/*.sha256`，删后重跑自动重写。

## 测试缝隙（seams）

1. 顶层 seam：`label_trials(df) → df`（golden 测试已覆盖）。开关双态各测。
2. `classify_trial` 纯函数——PreEscape/buffer/优先级单测面。
3. RT/distance 计算：kinematics 层纯函数（无 matplotlib），新 golden 用例面。
4. 渲染 seam：`plot_trial_stacked_heatmap` 像素哈希。

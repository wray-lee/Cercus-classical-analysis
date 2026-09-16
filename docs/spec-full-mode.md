# Spec: `full` 模式 — 跨范式个体-组均值连线图（laboratory dumbbell convention）

## Problem Statement
实验室传统要求呈现"个体均值点 → 组均值黑方块 + 连线"图（cf. Frontiers fphys.2023.1153913 Fig 2）。
本实验有 9 个范式（5 个有数据：bv/bw/-373 30°/-308 36°/-261 42°；+200/-119 80°/-225 48°/0 180° 暂空），
范式间为 between-subject（动物不跨范式复用，已验证 subject id 无重叠）→ 跨范式折线不存在，
只能画"范式内个体点连范式均值"。现有 `population` 命令输入=单范式目录，无范式维度，承载不了此图。

## Solution
新 Typer 子命令 `cercus.cli.app full --input <父目录> --output <目录>`：
- 每个非空子目录 = 一个范式组（组名=目录名；排除 `_EXCLUDED_DIR_NAMES` + 无 kinematics CSV 的目录并 log warning 跳过）。
- 范式级多进程（Pool，与 population 同套路，不新建线程池）跑 io→preprocess→label_trials，
  拼表加 `paradigm` 列 → `full_summary.csv` + `full_meta.json`（含全局阈值与方法名）。
- **全局 V_max 自适应阈值**：所有范式 v_max 池化后跑一次 KDE valley → log-GMM → fallback 链
  （复用 population 的三个 `_compute_*` 函数，提取为可导入 helper），仅用于 `is_valid_escape` 标签；
  分类器本身走 config 固定阈值（98 mm/s），天然跨范式一致。
- 新图 `plot_paradigm_dumbbell(df, figsize)` → `paradigm_dumbbell.svg`，3 panel：
  1. **响应概率**：每动物 = (Escape+PreEscape)/全部 trial（trial 级，config 阈值口径）
  2. **RT**：每动物 = reaction_time_ms 均值（仅逃逸 trial）
  3. **distance**：每动物 = distance_mm 均值（仅逃逸 trial）
  画法规式（复刻 Fig 2）：x=范式（按 target_ttc_ms 数值升序：bv(视觉-only 最左)→bw→-373→-308→-261→…→+200），
  点=单动物均值（类别色不区分——每范式一色即可，用 NPG 前向色阶按范式序），
  竖线 point→black square（范式均值±SD 误差棒），n=动物数标在刻度下。
- 空/无数据范式：warning 跳过，图只画有数据的。

## User Stories
- 作为作者，我跑 `full --input /mnt/d/data` 得到 9(→当前5)范式并排的三 panel 连线图 + 全量 CSV，
  直接满足实验室呈现传统与论文主图需求。
- 作为审稿人，我看到检验/均值单位是动物（点=动物），无伪重复。

## Implementation Decisions
- `cercus/full/` 不建——放 `cercus/cli/app.py` 新 command + 新模块 `cercus/analysis/full.py`（数据侧）
  + `cercus/visualization/paradigm.py`（绘图侧）。垂直切片：io/分类/绘图三层复用现有 seam。
- population 的三个 adaptive-threshold 函数提取到 `cercus/analysis/vmax_threshold.py`，
  population_analysis.py 改为导入（单一事实源，避免复制粘贴分叉）。
- 范式排序：从目录名解析首整数（`-373 30°`→-373；`bv`/`bw`→无 wind→排最左，bv<bw 或按 n 大小，用目录名字典序兜底）。
  用 `re.search(r'[-+]?\d+', name)`。
- dumbbell 点线样式复用现有常量池（RESPONSE_COLORS 不适用——范式不是行为类；用 NPG_PALETTE 循环）。

## Testing
- 单测 `tests/test_full_analysis.py`：synthetic 双范式 df → `aggregate_paradigm_table` 输出行数/列/n 正确；
  空范式被跳过并 warning。
- 绘图 smoke：dumbbell 对 synthetic 出图不抛异常、3 axes、每范式点数=动物数。
- 真实数据端到端：full --input '/mnt/d/data' 跑通（当前 5 个有数据范式）。

## Out of Scope
- 方向/极坐标跨范式图（已有 per-paradigm polar）；统计检验 p 值标注（先出图，检验按审稿意见加）；
  MCMC 跨范式；per-paradigm 那 15 张图（那是 population 的活）。

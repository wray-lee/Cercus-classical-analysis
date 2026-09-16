# TICKET_DAG — PreEscape + RT/Distance + 热图重构

spec: docs/spec-preescape.md · base: cli @ c6290fb · 冲突检测: 同层无文件交集
并行票各自 worktree 分支 `orchestrator/T{n}`，validator 通过后 merge 回 cli。

```
T1 ──┬── T2 ──┐
     └── T3 ──┴── T4 ── T5
```

## T1 · feat(classify): PreEscape 行为类（开关式）  [complexity 4 → scout]
blocked-by: —
affected_files: `cercus/config/defaults/thresholds.yaml`, `cercus/constants/thresholds.py`, `pipeline/constants.py`, `pipeline/classifier.py`, `tests/test_classifier_golden.py`
- `classification.use_preescape: true` + `preescape_buffer_ms: 50.0` 进 YAML，经 constants 导出。
- `classify_trial`：burst 成立后、PreWalk 前插入分支——wind trial ∧ `interval_onset_ms < target_ttc_ms − buffer` ⇒ `PreEscape`。
- golden 测试：prewind 用例开关双态断言；边界 −50/−49；非 wind 永不 PreEscape。
- 验证: `pytest tests/test_classifier_golden.py -q`

## T2 · feat(kinematics): reaction_time_ms + distance_mm 导出  [3 → scout]
blocked-by: T1（classifier.py 文件重叠，强制串行）
affected_files: `cercus/core/kinematics/velocity.py`(或新 `distance.py`), `pipeline/classifier.py`, `pipeline/io.py`, `tests/test_rt_distance.py`
- 纯函数：`reaction_time_ms = onset − anchor`（wind: target_ttc；looming-only: 0；wind-only: stim onset 已=TTC 锚，按现轴语义）；`distance_mm` = speed 在 [onset,offset] numpy.trapezoid；`distance_500ms_mm`。
- `label_trials` 出三列；NoResponse ⇒ NaN。`export_summary_metrics` 增列。
- 合成阶梯曲线手算对照 trapz。
- 验证: `pytest tests/test_rt_distance.py tests/test_classifier_golden.py -q`

## T3 · feat(viz): 热图 wind onset 标识 + 回滚金框/白框  [3 → scout]
blocked-by: T1（PreEscape 面板需真实类别；colors.yaml 新增 preescape 色）
affected_files: `cercus/visualization/heatmaps.py`, `cercus/config/defaults/colors.yaml`, `tests/baselines/trial_stacked_heatmap*.sha256`
- 删除式回滚 13e1210 注释（`_flag_trial`、Rectangle/★/白刻度/图例）；PreWalk 排序退化为风窗均速单键。
- wind onset：**TTC 面板 = 整根竖线**（面板内 `target_ttc_ms` 唯一值各画一根，青色虚线+标注）；**onset 面板 = 每行青色 1px 刻度**（`target_ttc_ms − interval_onset_ms`）。
- conditions 面板含 PreEscape（n= 标注随动）。
- 验证: `pytest tests/test_visualization_regression.py -q`（热图基线重写后二次跑绿）

## T4 · feat(population): PreEscape 下游贯通（单一类别源）  [4 → scout]
blocked-by: T2, T3
affected_files: `cercus/constants/__init__.py`(新增 `RESPONSE_TYPES` 常量), `cercus/visualization/{behavior,polar,vmax,kinetics,individual}.py`, `cercus/analysis/individual.py`, `population_analysis.py`, `pipeline/mcmc.py`, `tests/baselines/{behavior,polar,vmax}*.sha256`
- 收敛 9 处 `["Escape","PreWalk","NoResponse"]` 字面量 → `RESPONSE_TYPES` 常量；颜色/中文名从 config 读。
- RT/distance 分布图：新图 `reaction_distance_panel.svg`（RT 小提琴×类型 + distance 箱线×类型，onset 后 500ms 行程）。PreEscape vs Escape 的 RT/distance 对比即教授要的机制分离证据图。
- mcmc `escape_only` 自动收窄（不改逻辑，验证集合语义）；polar/Wallraff 组对比纳入 PreEscape。
- 验证: `pytest tests/ -q`

## T5 · test+docs: 双态回归 + 基线定稿 + 文档  [2]
blocked-by: T4
affected_files: `tests/baselines/*`, `README.md`, `CLAUDE.md`, `CONTEXT.md`
- `use_preescape:false` 跑全链冒烟（population 单 subject，产物比对旧语义类别构成）；默认 true 全量基线定稿。
- README: 第四类 + 判定表 + RT/distance 图说明；CLAUDE.md 分类段更新优先级链。
- 验证: `pytest tests/ -q` 全绿 + 冒烟脚本输出含 PreEscape 计数

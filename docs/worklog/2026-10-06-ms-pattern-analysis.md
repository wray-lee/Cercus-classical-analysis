# MS pattern 输入与比较口径

记录日期：2026-10-06。

## 父目录中的 all 不能当作一个新范式

现象：`D:\Data\train` 同时包含标准 paired source pattern 目录和 `all`。
`all` 中包含旧 schema（events `session_id,trial_id,time_ms,event_type,event_value`；
kinematics `time_ms,x_pos,y_pos,heading,velocity,...`）与标准文件。

根因：文件名能配对不代表列/物理量相同。代表旧文件的导出 `velocity` 峰值
约 41 mm/s，verdict metadata 的 `peak_speed` 约 239 mm/s；两者不能直接等同于
当前 classifier 的 acquisition-clock displacement / centered-speed 输出。

避免：MS mode 输入标准 pattern 父目录；`analysis.multisensory.exclude_directories`
默认排除 `all`，在 meta 明示排除原因。未知 schema 按 subject 报 unsupported；
空目录与缺目录分别保留为 empty / missing。重复 session 的规范化键不得由 legacy
scanner 的 last-write 行为静默决定（例如 session_1 与 session_01 是同一编号）。
碰撞检测使用 scanner 同一次遍历和相同目录排除规则，garbage/pilot 中的废弃副本
不会让有效动物误报 unsupported。现有 raw data 不转换、不改写。

## 主概率与增强比较使用不同的、显式命名的 endpoint

主图用派生 Escape：可按 YAML 合并 raw PreWalk，PreEscape 仍独立。
增强比较用三种 modality 都一致的 any-burst（raw `BURST_CLASSES`，包含 PreEscape）。
否则 MS 的视觉触发 PreEscape 会被剔除，BV 的视觉触发 burst 却计入，对比的
outcome 不一致。增强 endpoint 不受 merge_prewalk 切换影响。

动物在范式内独立等权；同一动物多 condition 行先按 trial 数合并其成功比例，
bootstrap 在每个独立范式动物集内重采样，不按 frames/trials 构造群体 CI。
`P(V)+P(W)-P(V)P(W)` 是描述性概率独立参考，不是 RT race-model bound 或机制证明。

## 当前匹配与时间端点的证据边界

标准 train BV/MS 均记录 `lv_ratio_ms=120`，未记录 `init_half_angle_deg`。
匹配双方共同记录的 lv_ratio，并要求 initial-angle 缺失状态一致；CSV 明示
`lv_ratio_only_angle_unrecorded`。若双方有角度记录则精确匹配；不同已记录角度不混用。
这个现有 metadata 无法证明所有视觉刺激参数完全相同。

Overview 按完整原始 condition 值（type、TTC、l/v、initial angle）分组；格式化仅用于
显示，不能用 `:g` 生成组键，六位有效数字会把邻近但不同的刺激设置静默合并。
同一范式内不同设置必须保留独立动物点和可区分标签。

RT 先根据 raw 最终分类选择，再计算派生显示分组。BV 使用 TTC-relative timing；
wind/MS 使用校准风参考。PreEscape 是负 lead time；RT 缺失不剔除 classifier N，
不回填 centered/stopping endpoint。距离继续使用描述性 escape interval。
比较不能把视觉 TTC-relative timing 当作风刺激后的生理 RT。

## 实测

- 当前标准源：6 个有数据范式、70 个范式内动物、2520 trial；BV 720、BW 360，
  四个 MS pattern 各 360。
- `-119 80°`、`0 180°`、`+200` 保持 empty，未填充为零响应。
- RT classifier cohort 1994；observed 1986；missing 8。Trial 全局键无重复。
- 完整 CLI 产出 6 CSV、meta JSON、2 SVG；临时结果在
  `D:\Data\Results\tmp\ms_mode_20261006\cli`。
- 原始 train 下 541 个 CSV 的运行前后 SHA256 完全一致。

所有空/失败时仍导出 coverage 和空状态图；meta status 为 no_data。默认 worker=1，
worker 回传 trial 表而非全量 frames，避免跨范式累计庞大 frame DataFrame。

## 验证状态

独立审查发现并修复了规范化 session 键碰撞、废弃目录副本误拒绝，以及 condition
组键格式化丢失精度；新增了对应回归场景，并将 MS 图形样式参数移至 YAML。
WSL 曾在 Python 启动前返回 `Wsl/Service/E_UNEXPECTED`，恢复后完成最终运行验证：

- MS 测试：17 passed（3.81 s）。
- 全套测试：373 passed、22 个已有 Matplotlib layout warnings（11.62 s）。
- 从已验证 trial 导出重新生成最终 2 SVG 与 PNG 预览，更新有效配置 meta；检查了
  footer 宽度及图形布局，空条件与 RT observed/missing 说明可见。
- 最终 scanner 发现 70 个动物任务、140 个 session，无碰撞/unsupported；3 个空条件保留。
- 再次读取 train 下全部 541 个 CSV，最终 SHA256 与运行前快照完全一致。

最终结果、测试日志和验证报告位于 `D:\Data\Results\tmp\ms_mode_20261006`。

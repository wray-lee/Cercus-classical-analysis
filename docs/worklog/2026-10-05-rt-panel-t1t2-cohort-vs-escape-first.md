# Wind escape onset 口径不一致导致 PreEscape 误判和 T1/T2 面板缺失

记录日期：2026-10-05；修复与复核：2026-10-06。

## 现象

`-225 48°` 的默认 Wind PreWalk T1/T2 子图曾显示没有完整配对。源数据实际有两个可观测的 T1/T2 配对，但旧分类将这两个 trial 都判为 PreEscape，因此它们不在 PreWalk 子图中。

排查期间曾把问题归因于“continuous_moving 与 T1/T2 互斥”，这个解释错误：continuous_moving 检查气流参考点**之前**的运动历史，T1/T2 检查参考点**之后**的停顿和再启动。动物可以风前持续运动，风后停下，再逃逸。

## 根因：复用了描述性 onset，随后又引入了另一套因果 onset

PreEscape 在 `aa0747b` 引入时直接复用 `interval_onset_ms` 与 wind onset 比较。对非 baseline_visual，interval onset 使用未做角速度细化的 `latency_coarse_ms`。原始实现这样可以复用现有线速度阈值起点，避免角速度细化改变区间；提交记录没有证明额外的生理测量设计意图。

这里没有“取中值”或 median 操作。“居中”指上游 `pipeline/kinematics.py` 的 Savitzky–Golay 位置平滑，再差分得到速度。随后向后搜索的是这条描述性速度曲线的最后一次低速观测。另一条源时钟路径使用 `ard_time` 间隔与 causal 20-ms 位移平均速度。这两种滤波和时钟可能产生不同的低速穿越点。

100-ms 居中平滑不能单独解释约 770-ms 的时间差；关键是两条速度曲线是否保留了 burst 前的低速段。如果短暂停顿在描述性曲线上消失，回溯就会找到更早一段运动的起点。仅凭这些指标，不能断言较早运动一定是普通行走而非逃逸，也不能把因果滤波端点叫作生理真值。

到 `9fcce0d`，PreEscape 分类仍采用描述性 interval onset，T1/T2 已采用源时钟因果 onset。同一 trial 因而可以同时得到“风前起点”和“风后 T1/T2”。这是共享测量口径不一致的实现错误。

## 已核实的两个 trial

trial 编号仅在 subject 内唯一，所有对比必须使用 `(subject_id, global_trial_id)`。

| subject_id | trial | wind onset | 描述性 interval onset | 源时钟 causal onset | T1 | T2 | causal RTm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.732cricket_20260928_143240` | 18 | -225 ms | -339.071 ms | -95 ms | 110 ms | 20 ms | 130 ms |
| `0.881cricket_20260925_202334` | 33 | -225 ms | -934.667 ms | -164 ms | 21 ms | 40 ms | 61 ms |

两个因果起点都晚于 wind onset。旧 `9fcce0d` 根据描述性 interval onset 把两者判为 PreEscape；`36cc0d8` 统一 wind 分类与因果起点后两者进入 PreWalk。

默认 `prewalk` cohort 的完整配对实际是 **0 → 2**，不是“trial 18 原本在 PreWalk、只新增 trial 33”。那个说法来自遗漏 subject 的 trial 编号比较，已废弃。

## 严格历史视图是另一个集合

`pause_cohort="strict_moving"` 选择风前完整 1 s 的 continuous_moving 子集。这两个配对 trial 的运动占比分别为 0.615、0.685，是 intermittent_moving，所以不会进入该严格视图。

这与 PreEscape 判定是两个条件：PreEscape 比较起点与硬件风触发；风前历史决定通过 burst/PreEscape 检查之后的 PreWalk 资格。intermittent 本身并不等于 PreEscape，也不能说运动历史完全不参与分类。

`-225 48°` 的三个 continuous_moving trial 是 `0.766…/5`、`0.881…/10`、`0.881…/34`。它们的观测起点确实早于 wind，停点状态为 escape_first。因此该严格子集没有配对。不能据此推导 continuous_moving 必然没有 T1/T2。

若共享起点已在风前，`escape_first` 关闭“风后停点先于这次逃逸”的搜索是合理缺失；若停点搜索采用错误的描述性起点，则同一短路也可能误关搜索。必须先核对使用的是哪个起点，不能仅看到状态名就断言正确。

## 隔离重放的历史结果

同一批原始 CSV，每个 commit 独立进程，保留完整复合 trial 键：

| commit | Escape | PreEscape | PreWalk | NoResponse | 全 wind T1/T2 配对 | 默认面板 cohort |
|---|---:|---:|---:|---:|---:|---|
| `054e5f1` | 271 | 63 | 18 | 8 | 尚无 pause_* | — |
| `cb26f2b` | 271 | 63 | 18 | 8 | 2 | strict_moving |
| `952764c` | 289 | 63 | 0 | 8 | 2 | strict_moving |
| `0401736` / `9fcce0d` | 282 | 63 | 7 | 8 | 2 | prewalk |
| `36cc0d8` | 286 | 59 | 7 | 8 | 2 | prewalk |

`0401736` 改默认 cohort 后，这两个配对仍被判为 PreEscape，没有立即出现在默认 PreWalk 面板。到 `36cc0d8` 修正分类才出现。PreWalk 总数恰好 7 → 7，不代表成员未变：4 个原 PreWalk 变为 PreEscape，8 个原 PreEscape 变为4个 Escape 和4个 PreWalk。

证据保存在 `D:\Data\Results\tmp\legacy_gate\{cb26,952764c,0401736,9fcce0d,cur}.json`。

## 2026-10-06：修复残留的 Escape 时间路径

`36cc0d8` 修正了 PreEscape 分类，但残余 Wind Escape 的 escape_reaction_time_ms 和 short_rt 仍可能沿用居中 interval onset；兼容停点搜索也仍以那个起点作截止。此次继续统一：

- acquisition wind trial 导出 escape_onset_ms；wind 分类、escape_reaction_time_ms、short_rt 共用该因果观测。有采集列但起点无法观测时保持 NaN，不回填居中时间。
- 爆发搜索仍锚定硬件风触发，历史、停止和 RT 参考点使用气流到达校准。arrival delay 不得移动合格 burst 窗口或 PreEscape 比较点。
- 从完整硬件 burst 窗口查找首个有有效低速过渡的 candidate。向后只走连续有效观测，到最后低速观测停止；不能越过缺失观测或假设记录开头就是起点。
- 前一 burst 与后一 burst 之间若有可观测低速段，后一 onset 不得回到前一段。若没有观测到低速过渡，不凭任意时间截断创造新的 onset；真正持续的风前逃逸仍保留风前 lead time。
- 两个停止入口共用 first-burst 守卫后的 stopping cap，防止前一个 burst 的减速被误当作后一个 burst 之前的风诱发停顿。
- independent_rtm 仅控制 RTm 是否要求 T1/T2 配对；不改变分类和可观测的 Escape RT。true 使用硬件窗内的共享起点；false 保留 stop-relative T1/T2 RT，后者窗口可能延伸到硬件窗之外。

描述性 latency_ms、interval_onset_ms/offset_ms 和 distance 不重新定义：它们仍用于行程、轨迹/角度、动力学标记、热图和已有 MCMC survival 视图。这些描述性消费者不因为 RT 修复而自动变为生理反应时测量。

真实重放结果（360 trials、40 raw CSV）：分类仍为 Escape 286 / PreEscape 59 / PreWalk 7 / NoResponse 8；描述性区间和行程保持一致；293 个 trial 的 escape_reaction_time_ms 更新为源时钟值。两个配对 trial 的兼容停点也变为 observed，与 T1/T2 一致。原始 CSV 和已有正式输出的 SHA-256 前后相同。

验证报告：`D:\Data\Results\tmp\causal_escape_20261006\report.json`。重放图和导出仅写到该 tmp 目录。

## 避免复发

1. 区分历史资格、最终分类、观测端点和绘图 cohort。先检查 trial 所属类别及使用的时间口径，再解释“没有配对”。
2. 同时导出描述性 interval 与源时钟 onset，端点不可观测时保留缺失；用短暂停顿、多 burst、数据缺失和非零 arrival delay 回归测试检查各消费者是否复用同一测量。
3. 跨 commit 重放必须验证实际 import 路径。Windows `git worktree add /tmp/...` 会创建 Windows Temp 目录；WSL `/tmp/...` 是另一目录。不存在的 sys.path 项会静默回退到当前代码。本次早期“旧版也是7个PreWalk”重放因此无效，不作为证据。
4. 每个 commit 使用独立进程，避免模块缓存；验证前后 raw hashes。所有暂存产物放 Results/tmp，worklog 正式纳入 Git 追踪。

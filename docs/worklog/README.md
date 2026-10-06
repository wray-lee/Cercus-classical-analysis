# Worklog

工程踩坑与实测记录。与 `docs/spec-*.md` / `docs/decision-*.md` 的分工：

| | 记什么 |
|---|---|
| `docs/spec-*.md` | 功能规格 — 一个特性应该怎么工作 |
| `docs/decision-*.md` | 架构决策 — 难以逆转、需要解释「为什么这么设计」 |
| `docs/worklog/` | 工程踩坑 — 静默出错的陷阱、违反直觉的实测、语义陷阱 |

## 写入原则

只记**下次会再踩**的东西：

- ✅ 静默出错的陷阱（指标看起来对，含义其实错了）
- ✅ 同名不同义的字段 / 指标口径混淆
- ✅ 违反直觉的实测结果
- ❌ 已经被代码或 git history 记录的（改了什么、修了哪个 bug）
- ❌ 一次性的操作细节

每条包含：**现象 → 根因 → 怎么避免**。命名 `YYYY-MM-DD-slug.md`。

## 索引

- [2026-10-06 — MS pattern 输入与比较口径](2026-10-06-ms-pattern-analysis.md) — legacy all 与标准 source schema、any-burst 增强端点、动物 bootstrap，以及空范式/缺 RT 的报告边界

- [2026-10-05/06 — Wind onset 口径不一致与 T1/T2 面板缺失](2026-10-05-rt-panel-t1t2-cohort-vs-escape-first.md) — 描述性与因果起点混用、错误历史对比的纠正，以及共享采集时钟端点的修复和实测
- [2026-10-06 — 派生响应分组 + 独立刺激前基线](2026-10-06-derived-response-grouping-and-prestim-baseline.md) — raw `response_type` 与派生 `response_group` 的口径陷阱（哪些视图用派生、哪些必须留 raw），以及 `prestim_*` 在原始全 session 上的测量边界与证据边界

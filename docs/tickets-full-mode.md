# 工单与执行流 (Ticket DAG)：`full` 跨范式全景对比与哑铃图

> **对应规范**: [docs/spec-full-mode.md](spec-full-mode.md) · **状态**: 已完成并在主分支锁定

本方案将跨范式对比功能分解为 4 个结构清晰的迭代工单，涵盖阈值算法抽取、多范式并行处理、定制矢量图表与命令行集成。

```text
T1 (自适应阈值函数抽取) ──► T2 (跨范式数据聚合) ──┐
                                                  ├────► T4 (CLI 集成与全量测试)
                         T3 (组均值哑铃图绘制) ──┘
```

---

## T1 · refactor(analysis): 抽取通用的 V_max 自适应阈值模块
- **目标**：消除代码重复，将原本内嵌于 `population_analysis.py` 的自适应爆发阈值计算逻辑抽取为可独立导入的通用模块。
- **改动范围**：新建 `cercus/analysis/vmax_threshold.py`，改造 `population_analysis.py`。
- **核心要点**：
  - 抽取 KDE 凹谷法、对数三组分 GMM、IQR 截断双组分 GMM 及默认回退机制。
  - 提供统一的级联评估函数 `select_vmax_threshold(all_vmax, cfg) -> tuple[float, str]`。
  - `population_analysis.py` 改为直接调用该公共函数，保持既有对外接口兼容。

---

## T2 · feat(analysis): 跨范式数据聚合与多进程处理
- **目标**：实现对多范式实验目录的递归扫描、时序排序与并行处理。
- **改动范围**：新建 `cercus/analysis/full.py`，新增单元测试 `tests/test_full_analysis.py`。
- **核心要点**：
  - 范式自动发现与排序：依据时序与刺激时间差（`target_ttc_ms`）升序排序（纯视觉 `bv` 排在最前）。
  - 多进程并行流水线：使用进程池对各范式子目录并发执行预处理与分类，汇总为全量 DataFrame 并打上 `paradigm` 标识列。
  - 全局自适应阈值：在全量池化样本上运行 T1 的自适应阈值算法，生成跨组统一的 `is_valid_escape` 标签。

---

## T3 · feat(viz): 出版级跨范式哑铃图绘制
- **目标**：绘制符合期刊规范的“个体均值点 - 组均值黑方块”三面板哑铃图。
- **改动范围**：新建 `cercus/visualization/paradigm.py`，补充绘图单测。
- **核心要点**：
  - 绘制 3 个并排子面板：响应概率、逃逸反应时间与逃逸位移。
  - 视觉规范：散点代表单只动物均值，按范式赋予典雅色系；黑色方块表示范式整体均值 $\pm\text{SD}$ 误差棒；细垂直线连接个体点与组均值。
  - 横坐标各刻度下方标注该范式的独立动物数（$n$），明确样本统计单位。

---

## T4 · feat(cli): `full` 命令行集成、端到端验证与文档
- **目标**：在统一 CLI 中暴露 `full` 子命令，完成全链路验证。
- **改动范围**：`cercus/cli/app.py`、`README.md`、`CLAUDE.md`。
- **核心要点**：
  - 命令行注册：`python -m cercus.cli.app full --input <dir> --output <dir> [--workers N]`。
  - 自动化导出：生成 `full_summary.csv`、`full_meta.json` 与 `paradigm_dumbbell.svg`。
  - 执行全量真实数据验证，确保多范式并发执行稳定流畅。

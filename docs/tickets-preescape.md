# 工单与执行流 (Ticket DAG)：PreEscape 类别引入、反应时间与位移量化

> **对应规范**: [docs/spec-preescape.md](spec-preescape.md) · **状态**: 已完成并在主分支锁定

本方案将 PreEscape 拓展与指标计算划分为 5 个串并联工单，确保测试覆盖与下游可视化无缝衔接。

```text
T1 (分类器核心改造) ──┬── T2 (反应时间与位移物理计算) ──┐
                     │                                  │
                     └── T3 (热图参考线与面板视觉重构) ─┴── T4 (群体分析贯通) ──► T5 (全量回归)
```

---

## T1 · feat(classify): PreEscape 状态机分支实现
- **目标**：在分类器中引入可由配置控制的第四类 `PreEscape`。
- **改动范围**：`thresholds.yaml`、`pipeline/classifier.py`、`tests/test_classifier_golden.py`。
- **核心要点**：
  - 增加 `classification.use_preescape` 与 `preescape_buffer_ms`（默认 0 ms）配置。
  - 在合格爆发（Burst）之后、PreWalk 检验之前插入判定：风刺激试次中，若回溯起跑点早于气流到达点，则归入 PreEscape。
  - 补充边界回归测试，确保非风试次及未提前起跑的试次不发生误判。

---

## T2 · feat(kinematics): 反应时间与逃逸位移指标导出
- **目标**：为每个试次计算物理意义明确的反应时间与逃逸位移。
- **改动范围**：`cercus/core/kinematics/`、`pipeline/classifier.py`、`pipeline/io.py`。
- **核心要点**：
  - 纯函数计算：$\text{RT} = t_{onset} - t_{anchor}$（Escape 为正，PreEscape 为负提前量；NoResponse 保持 NaN）。
  - 位移计算：对速度曲线进行数值梯形积分，输出区间完整位移 `distance_mm` 与起跑后 500 ms 固定位移 `distance_500ms_mm`。
  - 将上述指标如实汇入单被试及群体 summary CSV。

---

## T3 · feat(viz): 热图气流到达标记与面板拆分
- **目标**：在瀑布流热图中清晰标识气流刺激到达时刻，并为 PreEscape 新增专属面板。
- **改动范围**：`cercus/visualization/heatmaps.py`、`colors.yaml`。
- **核心要点**：
  - 在 TTC 对齐面板中增加完整竖向风刺激参考线；在 Onset 对齐面板中每行准确标出相对风到达时刻刻度。
  - 自动渲染 PreEscape 独立面板，直观展现提前逃逸的高速轨迹分布。

---

## T4 · feat(population): 群体统计与下游可视化适配
- **目标**：下游各图表和统计模块全面支持四元分类体系。
- **改动范围**：`cercus/constants/response_types.py`、`population_analysis.py`、各出图模块。
- **核心要点**：
  - 收敛散落的类别字面量，统一通过常量源引用；调色板引入专用的 PreEscape 色彩。
  - 新增反应时间与位移复合图（小提琴图 + 箱线图），直观展示 Escape 与 PreEscape 的动力学分离。
  - 确认 MCMC `escape_only` 集合语义自动排除 PreEscape，防止视觉逃逸污染气流反应拟合。

---

## T5 · test+docs: 全量冒烟测试、图表哈希定稿与文档同步
- **目标**：验证开关双态回退能力，锁定测试基线，更新项目说明。
- **核心要点**：
  - 验证 `use_preescape: false` 能无缝复现经典三元行为。
  - 重新确认并锁定所有图表的 SHA-256 像素哈希基线。
  - 更新中英文文档中的分类判据与图表解读说明。

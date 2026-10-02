# 规范说明：PreEscape 行为分类、反应时间与逃逸位移分析

> **状态**: 已实施交付 · **分支**: cli · **上下文关联**: [Standardized Criteria.md](../Standardized%20Criteria.md)

---

## 1. 背景与问题定义

在多模态（视觉逼近 + 气流 `looming_wind`）实验中，约有 20%–41% 的试次实际为**风刺激到达前由视觉单独诱发的真逃逸**（即起跑时刻早于给风）。在过去的三元分类逻辑中，这部分试次容易被机械归入 PreWalk 或 Escape：
1. **机理混淆**：把视觉引发的逃逸当成风诱导逃逸，干扰了多感觉整合机制的解析。
2. **拟合偏差**：在 MCMC `escape_only` 模式下，未等风来就起跑的试次会严重拉偏气流逃逸概率曲线。
3. **指标缺失**：缺乏以刺激到达为基准的严格反应时间（Reaction Time）与逃逸位移（Distance）量化。

---

## 2. 解决方案设计

### 2.1 引入 PreEscape 第四行为分类
- **可配置开关**：通过 YAML 配置 `classification.use_preescape` 控制（默认 `true`；设为 `false` 可一键回退到经典三元分类模式，确保既往对比数据完全兼容）。
- **判定规则**：
  - 气流试次中，在风后 250 ms 内检测到合格爆发（$V_{max} > V_{burst}$）。
  - 从爆发峰值向前回溯，起跑时刻早于气流到达点（即提前量超过 buffer 阈值，当前默认 buffer 设为 0 ms）。
- **优先级级联**：
  $$\text{NoResponse} \succ \text{PreEscape} \succ \text{PreWalk} \succ \text{Escape}$$

### 2.2 补齐反应时间与逃逸位移物理指标
- **逃逸反应时间 (`escape_reaction_time_ms`)**：
  $$\text{RT} = t_{onset} - t_{anchor}$$
  - 对 Escape 试次，RT 为正数，表示生理潜伏期。
  - 对 PreEscape 试次，RT 为负数，直接反映机体提前起跑的“提前量”（Lead Time）。
- **逃逸位移量化**：
  - `distance_mm`：在 $[t_{onset}, t_{offset}]$ 完整逃逸区间内对速度进行数值梯形积分。
  - `distance_500ms_mm`：自起跑点向后截取固定 500 ms 窗口积分，消除不同试次记录时长差异，便于横向比较。

### 2.3 热图视觉规范重构
- **明确风到达标记**：在 TTC 对齐面板中增加醒目的气流到达参考线；在 Onset 对齐面板中每行清晰绘制风到达相对位置刻度。
- **独立子面板展示**：PreEscape 试次拥有专属瀑布流面板，直观呈现其在风前便已形成的明亮逃逸速度带。

---

## 3. 实现与架构考量

- **核心算法纯粹性**：判定与积分函数置于 `cercus/core/kinematics/` 与 `pipeline/classifier.py`，保持纯数值计算，不与绘图逻辑耦合。
- **统一类别源**：弃用散落在各模块的类别字符串硬编码，统一自 `cercus.constants.response_types` 导出。
- **透明度优先**：NoResponse 或起跑点无法解析的试次，RT 与位移指标如实保持 `NaN`，不在底层进行主观填充。

---

## 4. 验证与回归测试

1. **黄金测试集断言**：在 `tests/test_classifier_golden.py` 中覆盖风前提前起跑与风后常规起跑的合成试次，严格断言类别归属与 RT 正负号。
2. **图表哈希防漂移**：更新并锁定图表像素 SHA-256 基线，确保多面板扩展后全量图表渲染整洁稳定。

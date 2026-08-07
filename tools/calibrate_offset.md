# calibrate_offset.py — 环形气流装置刺激角偏移校准

对环形气流装置（两个喷嘴固定在同一个环上）进行**单一全局角度偏移** `delta` 的校准。
整台装置共享一个偏移：`stim_true = stim_nominal + delta`（两个喷嘴随环整体旋转，仍是相对位置）。
`delta` 从行为数据本身反推（硬件已调回、无法实测），并写出校正后的数据副本。

- 脚本：`tools/calibrate_offset.py`
- 依赖：仅 `numpy`、`pandas`、`matplotlib`（`scipy` 不需要）
- **绝不修改原始数据**；所有输出写到 `--output` 目录
- 已通过 `--selftest`（合成已知 delta 的恢复验证）

---

## 1. 核心思路（务必先读）

校准目标：**找一个偏移 δ，让左右两侧（left/right 刺激）的逃跑误差尽量对称地落在目标 −18° 两侧**，即
校正后 `mean_error_left ≈ mean_error_right ≈ −18°`。

- **δ 只用一组数据算不准确**，因此用 `--input` 下**全部**组、全部 session 的 trial 一起 pooled 估计：
  `δ = (mean_error_left + mean_error_right)/2 + expected_error`（使左右两侧平均误差对 −18° 目标做最小二乘居中，即"尽量对称"）。
- 这个 δ 就是**设备偏移**，对**所有**组、**所有**刺激角度统一应用：`stim_corrected = stim_nominal + δ`。
- `expected_error` 是健康群体先验：`response − (stim_true+180)` 的圆平均期望 = −18°（即平均逃跑方向在刺激源对侧 162°）。

---

## 2. 环境要求（必读）

本仓库约定（见 `CLAUDE.md` 的 ENV_CONSTRAINT）：任何运行代码的命令必须在 WSL 内通过以下原子链执行，
且所有路径使用 WSL 格式（`D:\x` → `/mnt/d/x`）：

```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <python 命令>"
```

---

## 3. 对指定目录的数据进行校准

### 3.1 输入目录结构

`--input` 指向的目录下，**每个直接子文件夹 = 一个组（一只蟋蟀）**，组内包含配对的 session 文件：

```
<data>/
├── 2026.8.4/                  ← 组 1（默认按 folder 分组）
│   ├── 0.780cricket_..._session_1_events.csv
│   ├── 0.780cricket_..._session_1_kinematics.csv
│   ├── 0.780cricket_..._session_2_events.csv
│   └── 0.780cricket_..._session_2_kinematics.csv
├── 2026.8.4 4/                ← 组 2
│   └── ...
└── ...
```

- 事件文件按 `{subject}_session_{n}_events.csv` 命名，运动学文件为 `..._kinematics.csv`，自动配对。
- ⚠ 直接放在 `--input` 根下的 CSV **不会被扫描**（根目录被当作「组」的容器）。
- `--group-by subject` 改为按文件名前缀分组；本数据有两个 `0.886cricket` 文件夹，用 subject 会变 5 组，故默认用 `folder`。

### 3.2 运行命令

```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && \
cd /mnt/d/Projects/Cercus-cli && \
python tools/calibrate_offset.py \
  --input  /mnt/d/Data/bw/garbage/cali \
  --output /mnt/d/Data/bw/garbage/cali_corrected \
  --groups 6 \
  --plot"
```

`--groups 6` 与发现数不符时仅告警。`--plot` 默认开启（生成 `calibration_report.png`），`--no-plot` 关闭。

### 3.3 快速自检（合成数据验证数学正确性）

```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && \
cd /mnt/d/Projects/Cercus-cli && python tools/calibrate_offset.py --selftest"
```

植入全局 delta=7.0°，应恢复在 ±3° 内，且 `originals_unchanged / mirror_ok / events_gained_angles / response_angle_trials_used` 均为 `True`，结尾 `PASS`。

---

## 4. 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | `data` | 数据根目录（含各组子文件夹） |
| `--output` | `data_corrected` | 校正输出目录，**必须 ≠ --input** |
| `--groups` | 无 | 期望的组数，仅告警校验 |
| `--group-by` | `folder` | `folder`（子文件夹为一组）或 `subject`（文件名前缀为一组） |
| `--left-angle` | `90.0` | 软件认定的「左侧」喷嘴标称角（度） |
| `--right-angle` | `270.0` | 软件认定的「右侧」喷嘴标称角（度） |
| `--min-disp-mm` | `1.0` | 净位移低于此值的 trial 丢弃（近零向量角度无意义） |
| `--expected-error-deg` | `18.0` | 健康群体先验：`response−(stim+180)` 的圆平均期望值 |
| `--plot` / `--no-plot` | 开 | 是否生成报告图 |
| `--selftest` | 关 | 运行合成正确性自检后退出 |

⚠ 标称角只影响 δ 的基准：同一物理装置下，`δ` 与 `--left-angle/--right-angle` 一起决定校正后的真实角度。
若软件实际标称不是 90/270，请用参数覆盖（δ 会相应平移）。

---

## 5. 输入数据格式

### 5.1 events.csv
列：`event_name, timestamp, session_num, trial_in_session, global_trial_id, details`
其中 `details` 是 CSV 转义的 JSON（pandas 读取后自动还原引号）。

**刺激角解析优先级：**
1. `details` 中的数值角度键（存在即直接用）：`stim_angle` / `wind_angle` / `wind_direction` / `stim_azimuth` / `wind_azimuth` / `angle`
2. 否则回退到 `wind_dir` / `screen_side` 的 `left`/`right` → 映射到 `--left-angle` / `--right-angle`

⚠ 当前 `D:\Data\bw\garbage\cali` 数据**没有**数值角度键，只有 `wind_dir`/`screen_side`(left/right)，走第 2 条路径。

**响应角优先级：**
1. 若 `details` 含 `response_angle`（或 events 有 `response_angle` 列），直接用它作为该 trial 的逃跑方向
2. 否则从 kinematics 计算：`response = degrees(atan2(Σdy, Σdx))`，取 `stim_state==1` 期间（约 0.2 s 的风刺激窗口）的位移积分

### 5.2 kinematics.csv
列：`sys_time, ard_time, dx, dy, dz, stim_state, global_trial_id`
与 events 通过 `global_trial_id` 关联（字符串/整数均可，脚本已容错）。

---

## 6. 输出

输出目录镜像输入结构：

```
data_corrected/
├── {组文件夹}/
│   ├── {subject}_session_{n}_events.csv        # details 增加 stim_angle(标称) 与 stim_angle_corrected(=标称+δ)
│   ├── {subject}_session_{n}_kinematics.csv     # 仅新增列 stim_angle_corrected
├── calibration_report.json                      # 全局 δ、不确定度、每组校正前后对称性指标
└── calibration_report.png                       # 校正前后 error 的极坐标 rose + 直方图对比
```

- events 只改动 `trial_start` 行的 `details`（补 `stim_angle` / `stim_angle_corrected`），其余行/列原样保留。
- kinematics 不改任何角度，只新增 `stim_angle_corrected` 列用于追溯。

### calibration_report.json 关键字段

| 字段 | 含义 |
|---|---|
| `delta_est_deg` | **全局设备偏移** `δ = (mean_error_left + mean_error_right)/2 + expected_error`，用全部数据 pooled |
| `delta_method` | 说明 δ 的估计方式（单一全局偏移、全部组 pooled） |
| `pooled_circ_delta_deg` | 备选：全部 trial 圆平均 + expected_error（n 加权版，供对比） |
| `loo_delta_std_deg` | 留一蟋蟀法(leave-one-cricket-out)重新估计全局 δ 的标准差 = 不确定度 |
| `groups[i].per_group_circ_delta_deg` | 该组单独 pooled 圆平均估计（诊断：看各组一致性，不用于校正） |
| `groups[i].circ_mean_error_deg` | 该组 `wrap180(response − (stim_nominal+180))` 的圆平均 |
| `groups[i].slope / intercept / r` | 回归 `response ~ stim_nominal`（诊断用） |
| `groups[i].sides.{left,right}.mean_error_*_deg` | 校正**前**/**后**（用全局 δ）该侧平均误差 |
| `groups[i].sides.{left,right}.within_5deg_after` | 校正后该侧是否 `|mean_error+18| ≤ 5°`（对称性验收） |
| `nominal_mapping` / `parameters` | 记录所用左右标称角与全部参数，便于追溯 |

---

## 7. 结果解读与注意事项

1. **设备偏移是全局单一值**：所有组共用同一个 δ，校正文件里每组 `stim_angle_corrected = nominal + δ`。
   判断 δ 可靠：`loo_delta_std_deg` 小（各组一致）、且校正后左右两侧 `within_5deg_after=True`。

2. **对称性验收**：真正的目标是校正后左、右两侧的平均误差都回到 **−18±5°**。
   一个全局 δ 能让两侧同时达标的前提是：逃跑确实对侧（≈刺激源+180−18）且两喷嘴正对（相差180°）。

3. **本批 `garbage` 数据实测**（2026-08 校准集）：
   - 全局设备偏移 **δ ≈ −59.7°**，`loo_delta_std ≈ 1.13°`（6 只蟋蟀高度一致 → δ 可靠）。
   - 但校正后左右两侧仍不达标：left 平均误差约 **+50°**、right 约 **−77°**（各组间散布 ±10°）。
     即该数据里左/右逃跑方向仅相差 ~34°（而非对侧的 ~180°），**任何单一 δ 都无法把两侧同时拉回 −18±5°**。
   - 含义：δ（环偏移）能被稳定估计，但此数据的逃跑响应**不符合"对侧162°"先验**，不能用来验证行为对称性。
     需要真实（有效对侧逃跑）数据才能同时满足 δ 估计与 −18±5 验收。

4. **δ 的模 180 歧义**：两喷嘴相对（相差180°）时，δ 与 δ±180° 给出同一组真实喷嘴位置（如 −59.7° ≡ +120.3°）。
   报告以 `-180~180` 呈现。

5. **两条先验互斥（诊断说明）**：环向先验（`response−(stim+180)` 圆平均 = −18°，隐含斜率 1）与回归先验
   （slope≈0.9、intercept≈−1）不能同时成立。以全局 δ（圆平均路径）为准；`regression_implied_delta_deg`
   仅当数据符合回归先验时才 ≈ δ，是诊断、非门控（见 `cross_check_note`）。

6. 原数据绝不改动；如需重跑，删掉旧 `--output` 目录即可（脚本会覆盖写入）。

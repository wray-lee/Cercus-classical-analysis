# calibrate_offset.py — 环形气流装置刺激角偏移校准

对环形气流装置（两个喷嘴固定在同一个环上、**相差180°**）进行**单一全局角度偏移** `delta` 的校准。
整台装置共享一个偏移：`stim_true = stim_nominal + delta`（两喷嘴随环整体旋转，仍相对）。
`delta` 从行为数据本身反推（硬件已调回、无法实测），并写出校正后的数据副本。

- 脚本：`tools/calibrate_offset.py`
- 依赖：`numpy`、`pandas`、`matplotlib`、`scipy`（复用 cercus-cli 管道的 `savgol` 平滑）
- **绝不修改原始数据**；所有输出写到 `--output` 目录
- 已通过 `--selftest`（合成已知 delta 的恢复验证）

---

## 1. 核心思路（务必先读）

- **有效逃避** = cercus-cli 管道定义：`preprocess` → 三元分类 → 逃逸区间 arena 轨迹，
  **`response_type ∈ {Escape, PreWalk}`**（`v_max > 98 mm/s` 爆发，逃逸区间 = 10→10 mm/s 窗口）。
- **逃跑方向** = 逃逸区间内 arena 轨迹端点方向 `atan2(traj_x[-1], traj_y[-1])`
  （= main.py 轨迹绘制的约定，角度从 +y 起算）。
- **坐标帧**：`atan2(x,y)` 帧里 `0° = +y（正前）`，`+90° = +x（右）`。左喷嘴在 **−x 边 = 270°**，
  右喷嘴在 **+x 边 = 90°**（`draw_side_arrows` 画的就是左/右沿）。因此默认
  `--left-angle 270 --right-angle 90`（不是 90/270）。
- **蟋蟀左右逃跑方向不需要相差 180°**（这是正常的）；偏移 δ 的目标是**让左右尽量对称**：
  使左右两侧平均误差尽量对称地落在目标 `−18°` 两侧，即
  `δ = (mean_error_left + mean_error_right)/2 + expected_error`（18）。
- **δ 用 input 下全部组、全部 session 一起 pooled 估计**（单组不准），对**所有**组统一应用。
- **校正方式（格式完全不变）**：events 与原始逐字节相同、不新增任何字段；校正烘进 kinematics ——
  body-frame 的 `(dx, dy)` 每个分量按 `−δ` 旋转，使 arena 轨迹/逃跑方向整体旋转 `−δ`。
  这样 cercus 分析（`single`/`trial-panels`/`population`）在读取校正后 kinematics 时，
  逃跑方向 = 原始方向 − δ，等价于刺激角已校正（格式不变、无新增列）。δ 也写入
  `calibration_report.json` 供下游自行应用。

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

- ⚠ 直接放在 `--input` 根下的 CSV **不会被扫描**（根目录被当作「组」的容器）。
- `--group-by subject` 改为按文件名前缀分组；本数据有两个 `0.886cricket` 文件夹，用 subject 会变 5 组，故默认用 `folder`。

### 3.2 运行命令（全量校正，默认即用全部数据）

```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && \
cd /mnt/d/Projects/Cercus-cli && \
python tools/calibrate_offset.py \
  --input  /mnt/d/Data/bw/garbage/cali \
  --output /mnt/d/Data/bw/garbage/cali_corrected \
  --groups 6 \
  --plot"
```

`--input` 必须指向**父目录**（含各组子文件夹）。默认就用全部 6 组 pooled 出一个全局 δ，
不需要额外 flag。`--groups 6` 仅告警校验；`--plot` 默认开（生成 report.png），`--no-plot` 关。

### 3.3 快速自检

```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && \
cd /mnt/d/Projects/Cercus-cli && python tools/calibrate_offset.py --selftest"
```

植入全局 delta=7.0°，应恢复在 ±3° 内，且 `originals_unchanged / mirror_ok /
events_gained_angles / all_classified_escape` 均为 `True`，结尾 `PASS`。

---

## 4. 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | `data` | 数据根目录（含各组子文件夹） |
| `--output` | `data_corrected` | 校正输出目录，**必须 ≠ --input** |
| `--groups` | 无 | 期望的组数，仅告警校验 |
| `--group-by` | `folder` | `folder`（子文件夹为一组）或 `subject`（文件名前缀为一组） |
| `--left-angle` | `270.0` | 软件认定的左侧喷嘴标称角（escape 帧 atan2(x,y)，0=+y 正前，90=+x 右） |
| `--right-angle` | `90.0` | 软件认定的右侧喷嘴标称角（右沿 = 90） |
| `--min-disp-mm` | `1.0` | 逃逸轨迹净位移低于此值的 trial 丢弃 |
| `--expected-error-deg` | `18.0` | 健康群体先验：`response−(stim+180)` 的圆平均期望值 |
| `--plot` / `--no-plot` | 开 | 是否生成报告图 |
| `--selftest` | 关 | 运行合成正确性自检后退出 |

⚠ 标称角只影响 δ 的基准：`--left-angle/--right-angle` 需与软件实际标称一致（相差 180°），
否则 δ 会平移。本工具的响应帧是 `atan2(x,y)`，左/右喷嘴对应 270°/90°。

---

## 5. 输入数据格式

### 5.1 events.csv
列：`event_name, timestamp, session_num, trial_in_session, global_trial_id, details`
其中 `details` 是 CSV 转义的 JSON（pandas 读取后自动还原引号）。

- 刺激角：优先 `details` 里的数值角度键（`stim_angle / wind_angle / wind_direction /
  stim_azimuth / wind_azimuth / angle`）；否则用 `wind_dir`/`screen_side`(left/right) 映射到 `--left-angle/--right-angle`。
- 若 `details` 含 `response_angle`（或 events 有该列），该 trial 直接用它作为逃跑方向，
  否则走 cercus 逃逸管道（第 5.2 节）。

### 5.2 kinematics.csv
列：`sys_time, ard_time, dx, dy, dz, stim_state, global_trial_id`
- 逃跑方向由 cercus-cli `preprocess`（arena 坐标，`dz` 航向把 body-frame dx/dy 旋到全局）
  + `label_trials`（分类）+ 逃逸区间内轨迹端点方向 `atan2(x,y)` 计算。
- **Escape + PreWalk** 都算有效逃避；`NoResponse`（v_max ≤ 98 mm/s）丢弃。

---

## 6. 输出

输出目录镜像输入结构：

```
data_corrected/
├── {组文件夹}/
│   ├── {subject}_session_{n}_events.csv        # 与原始逐字节相同（格式完全一致，不新增字段）
│   ├── {subject}_session_{n}_kinematics.csv     # 同列；body-frame (dx,dy) 按 −δ 旋转（轨迹/逃跑方向旋转 −δ）
├── calibration_report.json                      # 全局 δ、不确定度、每组校正前后对称性指标
└── calibration_report.png                       # 校正前后 error 的极坐标 rose + 直方图对比
```

- **events 原样复制**（byte-identical），格式与原始完全一致，不新增任何键/列。
- **kinematics 格式不变**（同列 `sys_time,ard_time,dx,dy,dz,stim_state,global_trial_id`），
  但 body-frame `(dx, dy)` 按 `−δ` 旋转，使 arena 轨迹/逃跑方向整体旋转 `−δ`。
  校正后 cercus 分析算出的逃跑方向 = 原始方向 − δ，等价于刺激角已校正。

### calibration_report.json 关键字段

| 字段 | 含义 |
|---|---|
| `delta_est_deg` | **全局设备偏移** `δ = (mean_error_left + mean_error_right)/2 + expected_error`，全部组 pooled |
| `delta_method` | δ 估计方式说明 |
| `pooled_circ_delta_deg` | 备选：全部 trial 圆平均 + expected_error（n 加权版） |
| `loo_delta_std_deg` | 留一蟋蟀法重新估计全局 δ 的标准差 = 不确定度 |
| `groups[i].per_group_circ_delta_deg` | 该组单独估计（诊断：看各组一致性，不用于校正） |
| `groups[i].circ_mean_error_deg` | 该组 `wrap180(response − (stim_nominal+180))` 圆平均 |
| `groups[i].slope / intercept / r` | 回归 `response ~ stim_nominal`（诊断；正确帧下 slope>0，健康先验 ~0.9） |
| `groups[i].sides.{left,right}.mean_error_*_deg` | 校正前/后（用全局 δ）该侧平均误差 |
| `groups[i].sides.{left,right}.within_5deg_after` | 校正后该侧是否 `|mean_error+18| ≤ 5°`（对称性验收） |
| `nominal_mapping` / `parameters` | 记录所用左右标称角与全部参数 |

---

## 7. 结果解读与注意事项

1. **设备偏移是全局单一值**：所有组共用同一个 δ。校正后 kinematics 的逃跑方向 = 原始方向 − δ
   （等价于刺激角已校正，但格式完全不变）。判断 δ 可靠：`loo_delta_std_deg` 小（各组一致）。

2. **对称性验收**：目标是校正后左、右两侧平均误差都回到 **−18±5°**。由于 δ 让两侧误差
   对称地落在 −18 两侧，当两侧误差本身相差不大时即可达标。

3. **本批 `garbage` 数据实测**（2026-08 校准集）：
   - 全局设备偏移 **δ ≈ −13.2°**，`loo_delta_std ≈ 2.9°`（6 只蟋蟀较一致 → δ 可靠）。
   - 校正后左右误差对称地落在 −18 两侧：left ≈ **−75°**、right ≈ **+30°**（各组间散布），
     中点 ≈ −18°（"尽量对称"已达到），但每侧离 −18 仍有 ±50–60° 残余。
   - 含义：这批数据的逃跑方向（相对喷嘴 180° 对侧）本身不满足"对侧 162°"先验，
     左/右逃跑角差约 50–130°（非 180°）。**δ（环偏移）能被稳定估计，但此数据无法同时满足
     左右均进 −18±5°**。需真实（有效对侧逃跑）数据才能同时达标。

4. **δ 的模 180 歧义**：两喷嘴相对（相差180°）时，δ 与 δ±180° 给出同一组真实喷嘴位置。
   报告以 `-180~180` 呈现。

5. **两条先验互斥（诊断说明）**：环向先验（escape = stim+162，隐含斜率 1）与回归先验
   （slope≈0.9、intercept≈−1）不能同时成立。以全局 δ（圆平均路径）为准；回归仅诊断
   （正确帧下 slope 为正，接近先验方向）。

6. 原数据绝不改动；如需重跑，删掉旧 `--output` 目录即可（脚本会覆盖写入）。

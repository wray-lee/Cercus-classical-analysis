# Tickets: full 模式跨范式连线图

依赖 DAG: T2 ← T1；T3 独立；T4 ← {T2,T3}

## T1 refactor(analysis): 提取 population 的 V_max 自适应阈值函数
- 新建 `cercus/analysis/vmax_threshold.py`：从 `population_analysis.py` 原样迁移
  `_compute_kde_valley_threshold` / `_compute_log_gmm_threshold` / `_compute_iqr_gmm_threshold`
  + 级联选择 helper `select_vmax_threshold(all_vmax, cfg) -> tuple[float,str]`。
- `population_analysis.py` 改为导入（薄再导出，保持旧名字可调）。
- 测试：golden——构造双峰 v_max 数组，级联返回 KDE valley 值；单峰退化返回 fallback。
- affected: population_analysis.py, cercus/analysis/vmax_threshold.py, tests/test_vmax_threshold.py

## T2 feat(analysis): full 模式数据侧 `cercus/analysis/full.py`
- `discover_paradigms(input_dir) -> list[(name, dir)]`：子目录含 *_kinematics*.csv 才算范式
  （排除 _EXCLUDED_DIR_NAMES），空目录 warning 跳过。
- `sort_paradigms(names)`：按目录名首整数升序（bv/bw 无符号→最左，字典序）。
- `run_paradigm(name, dir)`：io→preprocess→label_trials → trial 级表 + paradigm 列。
- `aggregate_paradigm_table(input_dir, workers) -> (df, meta)`：范式级 Pool；
  池化 v_max → T1 的 select_vmax_threshold → `is_valid_escape`；meta={threshold, method, paradigms}。
- affected: cercus/analysis/full.py, tests/test_full_analysis.py

## T3 feat(viz): `plot_paradigm_dumbbell` 绘图
- 新模块 `cercus/visualization/paradigm.py`。3 panel（响应概率 / RT / distance，定义见 spec）。
- 画法规式：点=单动物均值（NPG 按范式上色）→ 竖线 → 黑方块=范式均值 ± SD；
  x 刻度下标 n=动物数；NaN 指标的动物不画点。
- 输入：T2 的聚合 df（列 paradigm, subject_id, response_rate, rt_mean, dist_mean）——
  绘图函数自己从 trial 级 df 聚合也可，签名收 trial 级 df（宽接口）。
- affected: cercus/visualization/paradigm.py, cercus/visualization/__init__.py,
  tests/test_paradigm_dumbbell.py

## T4 feat(cli): `full` 命令 + 端到端 + 文档
- `cercus/cli/app.py` 新 command `full`（--input/--output/--workers），调用 T2 聚合，
  写 full_summary.csv + full_meta.json + paradigm_dumbbell.svg。
- README + CLAUDE.md 增补 full 命令条目。
- 真实数据 e2e：/mnt/d/data 跑通（5 范式）。
- affected: cercus/cli/app.py, README.md, CLAUDE.md, tests/test_full_cli.py

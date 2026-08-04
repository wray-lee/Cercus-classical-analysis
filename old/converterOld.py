"""
Legacy Merged CSV to Cercus Standard Converter
专门针对包含 (Trial, AirflowSide, Time_dlc, ZeroBasedTime) 的已合并最终版数据。
严格复用每个 Trial 首帧 (iloc[0]) 作为基准进行旋转。
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

def convert_legacy_to_standard(data_dir: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描目录下所有的 csv
    all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    target_files = []

    # 自动识别含有 'Trial' 和 'ZeroBasedTime' 的目标合并文件
    for f in all_csvs:
        try:
            # 仅读取表头进行快速校验
            cols = pd.read_csv(f, nrows=0).columns
            if 'Trial' in cols and 'ZeroBasedTime' in cols and 'Theta[deg]' in cols:
                target_files.append(f)
        except Exception:
            continue

    if not target_files:
        print("未找到包含 Trial 和 ZeroBasedTime 的合并版 CSV 文件，请检查路径。")
        return

    for file_path in target_files:
        subject_prefix = os.path.basename(file_path).replace('.csv', '').replace('.CSV', '')
        print(f"\n--- 开始转换合并数据集: {subject_prefix} ---")

        df_full = pd.read_csv(file_path)

        session_num = 1
        base_name = f"{subject_prefix}_session_{session_num}"
        events_path = output_dir / f"{base_name}_events.csv"
        kin_path = output_dir / f"{base_name}_kinematics.csv"

        events_records = []
        kin_records = []

        events_records.append({
            "event_name": "session_config", "timestamp": 0.0, "session_num": session_num,
            "trial_in_session": 0, "global_trial_id": 0, "details": "{}"
        })

        # 按 Trial 切割数据，进行独立旋转映射
        for trial_id, df_trial in df_full.groupby('Trial'):
            # 确保时间轴顺序正确
            df_trial = df_trial.sort_values('Time[ms]').reset_index(drop=True)
            if df_trial.empty:
                continue

            airflow_side = str(df_trial.iloc[0]['AirflowSide'])
            stim_type = str(df_trial.iloc[0].get('StimType', 'Unknown'))
            trial_id_int = int(trial_id)

            # -------------------------------------------------------------
            # 核心：直接使用当前数据块的首帧朝向进行刚性旋转 (复现 V 字型)
            # -------------------------------------------------------------
            body_deg_init = df_trial.iloc[0]['Theta[deg]']
            body_rad_init = np.radians(body_deg_init)

            cos_val = np.cos(body_rad_init)
            sin_val = np.sin(body_rad_init)

            dX = df_trial['X[mm]'].diff().fillna(0).values
            dY = df_trial['Y[mm]'].diff().fillna(0).values

            # 严格套用旧代码矩阵
            rot_x = dX * cos_val + dY * sin_val
            rot_y = -dX * sin_val + dY * cos_val

            # -------------------------------------------------------------
            # 组装 Cercus 数据格式
            # -------------------------------------------------------------
            df_kin = pd.DataFrame()
            df_kin['sys_time'] = df_trial['Time[ms]'] / 1000.0
            df_kin['ard_time'] = df_trial['Time[ms]']

            # Cercus 绘图底层会执行 -dx 和 -dy，此处提前取负使得结果完美抵消
            df_kin['dx'] = -rot_x
            df_kin['dy'] = -rot_y
            df_kin['dz'] = 0.0

            # 使用已有的 ZeroBasedTime，>=0 意味着刺激发生
            df_kin['stim_state'] = (df_trial['ZeroBasedTime'] >= 0).astype(int)
            df_kin['global_trial_id'] = trial_id_int
            kin_records.append(df_kin)

            trial_start_sys = df_trial.iloc[0]['Time[ms]'] / 1000.0
            events_records.append({
                "event_name": "trial_start", "timestamp": trial_start_sys,
                "session_num": session_num, "trial_in_session": trial_id_int, "global_trial_id": trial_id_int,
                "details": json.dumps({"type": f"{stim_type}_{airflow_side}", "side": airflow_side})
            })

            trial_stop_sys = df_trial.iloc[-1]['Time[ms]'] / 1000.0
            events_records.append({
                "event_name": "trial_stop", "timestamp": trial_stop_sys,
                "session_num": session_num, "trial_in_session": trial_id_int, "global_trial_id": trial_id_int,
                "details": "{}"
            })

        # 落地输出
        if events_records and kin_records:
            df_events = pd.DataFrame(events_records, columns=["event_name", "timestamp", "session_num", "trial_in_session", "global_trial_id", "details"])
            df_events.to_csv(events_path, index=False)

            df_all_kin = pd.concat(kin_records, ignore_index=True)
            df_all_kin['sys_time'] = df_all_kin['sys_time'].map('{:.6f}'.format)
            df_all_kin['ard_time'] = df_all_kin['ard_time'].astype(int)
            df_all_kin.to_csv(kin_path, index=False, columns=["sys_time", "ard_time", "dx", "dy", "dz", "stim_state", "global_trial_id"])

            print(f"✅ 转换完毕: \n ├── {events_path.name}\n └── {kin_path.name}")

if __name__ == "__main__":
    LEGACY_DATA_DIR = r"D:\OH DATA\StopToEscape_Result_csv"
    OUTPUT_DIR = Path(f"{LEGACY_DATA_DIR}/cercus_standard_data")
    convert_legacy_to_standard(LEGACY_DATA_DIR, OUTPUT_DIR)
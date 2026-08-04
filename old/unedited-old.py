#%% Stop-to-Escape Trajectories
import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

# :small_blue_diamond: Trajectory 대상 폴더 설정
target_folder = r'D:\OH DATA\StopToEscape_Result_csv'

# :small_blue_diamond: CSV 파일 리스트 (대소문자 확장자 모두 포함)
list_csv = natsorted(glob.glob(os.path.join(target_folder, '*.csv')) +
                     glob.glob(os.path.join(target_folder, '*.CSV')))

# :small_blue_diamond: 파일 개수 확인
print(f"총 CSV 파일 개수: {len(list_csv)}")
if len(list_csv) == 0:
    print(":x: CSV 파일을 찾을 수 없습니다. 경로 또는 확장자 확인 필요.")
else:
    # :small_blue_diamond: 전체 trajectory 시각화
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for csvpath in list_csv:
        df_tmp = pd.read_csv(csvpath)
        x_diff = df_tmp['X[mm]'].diff()
        y_diff = df_tmp['Y[mm]'].diff()

        # 초기 각도 변환
        body_deg_init = df_tmp.iloc[0]['Theta[deg]']
        body_rad_init = body_deg_init * (math.pi / 180)

        # :white_check_mark: 회전 좌표 계산 (방향 반전 수정됨)
        cos = np.cos(body_rad_init)
        sin = np.sin(body_rad_init)
        rot_x = df_tmp['X[mm]'] * cos + df_tmp['Y[mm]'] * sin
        rot_y = -df_tmp['X[mm]'] * sin + df_tmp['Y[mm]'] * cos

        # 회전 기준점 0으로 보정
        df_tmp['Rotate_X'] = rot_x - rot_x.iloc[0]
        df_tmp['Rotate_Y'] = rot_y - rot_y.iloc[0]

        # 색상 설정
        airflow_side = df_tmp['AirflowSide'].iloc[0]
        if airflow_side == 'Right':
            color = 'r'
        elif airflow_side == 'Left':
            color = 'b'
        else:
            color = 'gray'

        # 전체 plot 추가
        ax.plot(df_tmp['Rotate_X'], df_tmp['Rotate_Y'], color=color)

    ax.set_title("All Stop-to-Escape Trajectories")
    ax.set_xlabel("Rotated X (mm)")
    ax.set_ylabel("Rotated Y (mm)")
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()  # :fire: 필수: 전체 trajectory 시각화 출력

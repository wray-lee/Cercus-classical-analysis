import pandas as pd
import numpy as np
import glob
import math

# 固化的底层硬件基准缩放比
ANCHOR_VAL = 60 * math.pi / 4000

def rescue_csv(file_path):
    df = pd.read_csv(file_path)
    if 'dx' not in df.columns:
        return
        
    v_old = df[['dx', 'dy', 'dz']].values
    
    if len(v_old) == 0 or np.all(v_old == 0):
        return
    
    # 1. 逆向剥离：反除标尺并取整，无损还原硬件捕捉的原始整型脉冲 (N, 3)
    v_raw = np.round(v_old / ANCHOR_VAL)
    
    # 2. 矩阵重构：利用数据映射的几何刚性，最小二乘法倒推回被覆盖的 3x3 旧矩阵
    X, _, _, _ = np.linalg.lstsq(v_raw, v_old, rcond=None)
    M_old = X.T
    
    # 3. 维度修补：将 Z 轴（矩阵第三列）因算法截断丢失的权重进行物理翻倍补偿
    M_new = M_old.copy()
    M_new[:, 2] *= 2.0
    
    # 4. 正向投影：用修复并补偿后的新矩阵，重新计算绝对真实的物理坐标
    v_new = np.dot(v_raw, M_new.T)
    
    # 5. 覆盖保存
    df[['dx', 'dy', 'dz']] = v_new
    fixed_path = file_path.replace('.csv', '_fixed.csv')
    df.to_csv(fixed_path, index=False)
    print(f"数据逆向重构并修补完成: {fixed_path}")

# 批量处理当前数据目录下所有未修复的文件
for path in glob.glob("data/*_kinematics.csv"):
    if "_fixed" not in path:
        rescue_csv(path)
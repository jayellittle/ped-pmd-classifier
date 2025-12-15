# pyright: basic

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

from config import CONFIDENCE_THRESHOLD, MIN_TRAJECTORY_LEN


def calculate_y_dist_to_line(head_positions):
    if len(head_positions) < 3:
        return 0.0, 0.0

    start_point = head_positions[0]
    end_point = head_positions[-1]
    x_start, y_start = start_point
    x_end, y_end = end_point

    if abs(x_end - x_start) < 1e-6:
        return 0.0, 0.0

    # y = mx + c
    m = (y_end - y_start) / (x_end - x_start)
    c = y_start - m * x_start

    vertical_distances = []
    for point in head_positions:
        x_i, y_i = point
        y_line = m * x_i + c

        distance = y_i - y_line
        vertical_distances.append(distance)

    return np.var(vertical_distances), np.std(vertical_distances)


def calculate_trajectory_straightness_ratio(head_positions):
    if len(head_positions) < 2:
        return 0.0

    actual_distance = 0.0
    for i in range(1, len(head_positions)):
        x1, y1 = head_positions[i - 1]
        x2, y2 = head_positions[i]
        segment_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        actual_distance += segment_distance

    start_x, start_y = head_positions[0]
    end_x, end_y = head_positions[-1]
    straight_distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    if straight_distance == 0:
        return 0.0 if actual_distance == 0 else float("inf")

    ratio = actual_distance / straight_distance

    return ratio


def calculate_trajectory_frequency(head_positions, fps):
    if len(head_positions) < MIN_TRAJECTORY_LEN or fps == 0:
        return {"dominant_freq": 0.0, "walking_band_energy": 0.0}

    # Detrended Y Signal
    start_point = head_positions[0]
    end_point = head_positions[-1]
    x_start, y_start = start_point
    x_end, y_end = end_point

    detrended_y = []
    if abs(x_end - x_start) < 1e-6:  # Case of vertical movement
        y_coords = [p[1] for p in head_positions]
        detrended_y = np.diff(y_coords)
    else:
        # y = mx + c
        m = (y_end - y_start) / (x_end - x_start)
        c = y_start - m * x_start
        for point in head_positions:
            x_i, y_i = point
            y_line = m * x_i + c
            detrended_y.append(y_i - y_line)

    if len(detrended_y) == 0:
        return {"dominant_freq": 0.0, "walking_band_energy": 0.0}

    # FFT
    N = len(detrended_y)
    yf = fft(np.array(detrended_y))
    xf = fftfreq(N, 1 / fps)  # Hz

    # Frequency Spectrum Analysis　(Positive Only)
    yf_positive = 2.0 / N * np.abs(yf[0 : N // 2])
    xf_positive = xf[0 : N // 2]

    # Feature Extraction
    # (a) Dominant Frequency: Search peaks (Except slow movements under 1Hz)
    try:
        relevant_indices = np.where(xf_positive >= 1.0)
        peak_index = np.argmax(yf_positive[relevant_indices])
        dominant_freq = xf_positive[relevant_indices][peak_index]
    except ValueError:
        dominant_freq = 0.0  # No peak

    # (b) Walking Band Energy: Around 1~3Hz
    walking_band_indices = np.where((xf_positive >= 1.0) & (xf_positive <= 3.0))
    walking_band_energy = np.sum(yf_positive[walking_band_indices] ** 2)

    return {"dominant_freq": dominant_freq, "walking_band_energy": walking_band_energy}


def calculate_angle(p1, p2, p3):
    """Return angle on p2"""
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])

    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm_product == 0:
        return 0.0

    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def calculate_arm_angle_variance(all_kpts, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    관절 좌표에 선형 보간법(Linear Interpolation)을 적용한 후 팔 각도 분산을 계산합니다.
    """
    if not all_kpts or len(all_kpts) < MIN_TRAJECTORY_LEN:
        return 0.0

    # 1. 데이터를 Pandas DataFrame으로 변환하기 위해 리스트로 추출
    data_list = []
    for kpts in all_kpts:
        # kpts shape: (17, 3) -> [x, y, conf]
        row = {}
        
        # 왼쪽 팔 (Shoulder: 5, Elbow: 7, Wrist: 9)
        row['l_sh_x'], row['l_sh_y'], row['l_sh_c'] = kpts[5]
        row['l_el_x'], row['l_el_y'], row['l_el_c'] = kpts[7]
        row['l_wr_x'], row['l_wr_y'], row['l_wr_c'] = kpts[9]
        
        # 오른쪽 팔 (Shoulder: 6, Elbow: 8, Wrist: 10)
        row['r_sh_x'], row['r_sh_y'], row['r_sh_c'] = kpts[6]
        row['r_el_x'], row['r_el_y'], row['r_el_c'] = kpts[8]
        row['r_wr_x'], row['r_wr_y'], row['r_wr_c'] = kpts[10]
        
        data_list.append(row)

    df = pd.DataFrame(data_list)

    # 2. 신뢰도가 낮은 좌표를 NaN(결측치)으로 변경 (Masking)
    # 관절별로 confidence 컬럼을 확인하여 좌표를 날림
    joints = [
        ('l_sh', 'l_sh_c'), ('l_el', 'l_el_c'), ('l_wr', 'l_wr_c'),
        ('r_sh', 'r_sh_c'), ('r_el', 'r_el_c'), ('r_wr', 'r_wr_c')
    ]
    
    for prefix, conf_col in joints:
        mask = df[conf_col] < confidence_threshold
        df.loc[mask, [f'{prefix}_x', f'{prefix}_y']] = np.nan

    # 3. 선형 보간법 적용 (Interpolation)
    # NaN으로 비어있는 구간을 앞뒤 값을 이용해 직선으로 채움
    coord_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    df[coord_cols] = df[coord_cols].interpolate(method='linear', limit_direction='both')

    # 4. 보간된 좌표로 각도 계산
    all_detected_angles = []

    for _, row in df.iterrows():
        # 왼쪽 팔 각도 계산
        if not np.isnan(row['l_sh_x']) and not np.isnan(row['l_el_x']) and not np.isnan(row['l_wr_x']):
            p1 = (row['l_sh_x'], row['l_sh_y'])
            p2 = (row['l_el_x'], row['l_el_y'])
            p3 = (row['l_wr_x'], row['l_wr_y'])
            angle = calculate_angle(p1, p2, p3)
            all_detected_angles.append(angle)

        # 오른쪽 팔 각도 계산
        if not np.isnan(row['r_sh_x']) and not np.isnan(row['r_el_x']) and not np.isnan(row['r_wr_x']):
            p1 = (row['r_sh_x'], row['r_sh_y'])
            p2 = (row['r_el_x'], row['r_el_y'])
            p3 = (row['r_wr_x'], row['r_wr_y'])
            angle = calculate_angle(p1, p2, p3)
            all_detected_angles.append(angle)

    # 데이터가 너무 적으면 0 반환
    if len(all_detected_angles) < 10:
        return 0.0

    return np.var(all_detected_angles)

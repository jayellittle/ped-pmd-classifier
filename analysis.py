# pyright: basic

import numpy as np
from scipy.fft import fft, fftfreq

from config import CONFIDENCE_THRESHOLD


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
    if len(head_positions) < 15 or fps == 0:
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
    all_detected_angles = []

    for kpts in all_kpts:
        left_shoulder, left_elbow, left_wrist = kpts[5], kpts[7], kpts[9]
        right_shoulder, right_elbow, right_wrist = kpts[6], kpts[8], kpts[10]

        if (
            left_shoulder[2] > confidence_threshold
            and left_elbow[2] > confidence_threshold
            and left_wrist[2] > confidence_threshold
        ):
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            all_detected_angles.append(left_angle)

        if (
            right_shoulder[2] > confidence_threshold
            and right_elbow[2] > confidence_threshold
            and right_wrist[2] > confidence_threshold
        ):
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            all_detected_angles.append(right_angle)

    if len(all_detected_angles) < 10:
        return 0.0

    return np.var(all_detected_angles)

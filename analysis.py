# pyright: basic

import numpy as np


def calculate_y_dist_to_line(head_positions):
    if len(head_positions) < 3:
        return 0.0, 0.0

    start_point = head_positions[0]
    end_point = head_positions[1]
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

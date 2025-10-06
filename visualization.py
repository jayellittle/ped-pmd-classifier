# pyright: basic

import cv2
import numpy as np
from config import CONFIDENCE_THRESHOLD, SKELETON_COLORS, SKELETON_CONNECTIONS


def draw_skeleton(frame, keypoints, confidence_threshold=CONFIDENCE_THRESHOLD):
    for connection in SKELETON_CONNECTIONS:
        kpt1_idx, kpt2_idx = connection
        kpt1 = keypoints[kpt1_idx]
        kpt2 = keypoints[kpt2_idx]

        if kpt1[2] > confidence_threshold and kpt2[2] > confidence_threshold:
            pt1 = (int(kpt1[0]), int(kpt1[1]))
            pt2 = (int(kpt2[0]), int(kpt2[1]))
            color = SKELETON_COLORS.get(connection, (0, 255, 0))
            cv2.line(frame, pt1, pt2, color, 2)


def draw_head_trajectory(frame, head_positions, color=(0, 0, 255)):
    if len(head_positions) < 2:
        return

    for i in range(1, len(head_positions)):
        pt1 = (int(head_positions[i - 1][0]), int(head_positions[i - 1][1]))
        pt2 = (int(head_positions[i][0]), int(head_positions[i][1]))

        # Connect line if shorter than 100px (To prevent tracking error)
        distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        if distance < 100:
            cv2.line(frame, pt1, pt2, color, 2)  # Width = 2

        # Circles for head position
        for pos in head_positions:
            pt = (int(pos[0]), int(pos[1]))
            cv2.circle(frame, pt, 2, color, -1)


def draw_head_line(frame, head_positions, color=(255, 200, 0)):
    if len(head_positions) < 2:
        return

    start_pt = (int(head_positions[0][0]), int(head_positions[0][1]))
    end_pt = (int(head_positions[-1][0]), int(head_positions[-1][1]))
    cv2.line(frame, start_pt, end_pt, color, 1)  # Width = 1

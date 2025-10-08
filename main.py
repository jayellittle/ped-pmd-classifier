# pyright: basic

import glob
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict


from analysis import (
    calculate_trajectory_frequency,
    calculate_trajectory_straightness_ratio,
    calculate_y_dist_to_line,
)
from config import CONFIDENCE_THRESHOLD, INPUT_DIR, MIN_TRAJECTORY_LEN, OUTPUT_DIR
from visualization import draw_head_line, draw_head_trajectory, draw_skeleton


os.makedirs(OUTPUT_DIR, exist_ok=True)
video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
model = YOLO("yolov8x-pose.pt")


def process_videos():
    analysis_results = []
    for video_path in video_files:
        print("=================================================")
        print(f"Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        base_filename = os.path.basename(video_path)
        filename_without_ext = os.path.splitext(base_filename)[0]

        output_video_name = filename_without_ext.replace("input_", "output_") + ".mp4"
        output_video_path = os.path.join(OUTPUT_DIR, output_video_name)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

        person_data = defaultdict(lambda: {"head_positions": []})

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, persist=True)
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                if results[0].keypoints.data is not None:
                    all_keypoints = results[0].keypoints.data.cpu().numpy()

                for track_id, kpts in zip(track_ids, all_keypoints):
                    draw_skeleton(frame, kpts)

                    for i, kpt in enumerate(kpts):
                        if kpt[2] > CONFIDENCE_THRESHOLD:
                            cv2.circle(
                                frame, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1
                            )

                    # Using Indices of Nose, Eyes, Ears to make a Head Index
                    head_keypoints_indices = [0, 1, 2, 3, 4]
                    valid_kpts = []
                    for idx in head_keypoints_indices:
                        kpt_x, kpt_y, kpt_conf = kpts[idx]
                        if kpt_conf > CONFIDENCE_THRESHOLD:
                            valid_kpts.append((kpt_x, kpt_y))

                    if valid_kpts:
                        x_coords = [k[0] for k in valid_kpts]
                        y_coords = [k[1] for k in valid_kpts]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        head_center_x = (x_min + x_max) / 2
                        head_center_y = (y_min + y_max) / 2

                        person_data[track_id]["head_positions"].append(
                            (head_center_x, head_center_y)
                        )

                    # Draw head trajectory of people tracked
                    draw_head_trajectory(frame, person_data[track_id]["head_positions"])

                    # Draw a line of head trajectory from start to end point
                    draw_head_line(frame, person_data[track_id]["head_positions"])

                    cv2.circle(
                        frame,
                        (int(head_center_x), int(head_center_y)),
                        5,
                        (0, 0, 255),
                        -1,
                    )
                    # Show ID of people tracked
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (int(head_center_x), int(head_center_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                    )
            out.write(frame)

        for track_id, data in person_data.items():
            if len(data["head_positions"]) > MIN_TRAJECTORY_LEN:
                y_var, y_std = calculate_y_dist_to_line(data["head_positions"])
                straightness_ratio = calculate_trajectory_straightness_ratio(
                    data["head_positions"]
                )
                freq_features = calculate_trajectory_frequency(
                    data["head_positions"], fps
                )

                analysis_results.append(
                    {
                        "video_name": base_filename,
                        "track_id": track_id,
                        "y_var": y_var,
                        "y_std": y_std,
                        "straightness_ratio": straightness_ratio,
                        "dominant_freq": freq_features["dominant_freq"],
                        "walking_band_energy": freq_features["walking_band_energy"],
                    }
                )

        cap.release()
        out.release()
        print(f"Finished processing video: {base_filename}")

    if analysis_results:
        df = pd.DataFrame(analysis_results)
        df.to_csv("analysis_results.csv", index=False, encoding="utf-8-sig")
        print(
            f"All feature data saved to 'analysis_results.csv'. ({len(analysis_results)} records)"
        )
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    print("Starting video processing...")
    process_videos()
    print("All videos have been processed.")

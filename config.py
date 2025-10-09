# pyright: basic

VERSION = "v3"
INPUT_DIR = "video_inputs"
OUTPUT_DIR = f"results/{VERSION}/video_outputs"
ANALYSIS_DIR = f"results/{VERSION}"

CONFIDENCE_THRESHOLD = 0.5

MIN_TRAJECTORY_LEN = 15

SKELETON_CONNECTIONS = [
    # Face
    (0, 1),
    (0, 2),  # nose to eyes
    (1, 3),
    (2, 4),  # eyes to ears
    # Body
    (5, 6),  # shoulders
    (5, 11),
    (6, 12),  # shoulders to hips
    (11, 12),  # hips
    # Arms
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    # Legs
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    # Neck
    (5, 0),
    (6, 0),  # shoulders to nose
]

SKELETON_COLORS = {
    # Face: Blue
    (0, 1): (255, 0, 0),
    (0, 2): (255, 0, 0),
    (1, 3): (255, 0, 0),
    (2, 4): (255, 0, 0),
    # Body: Green
    (5, 6): (0, 255, 0),
    (5, 11): (0, 255, 0),
    (6, 12): (0, 255, 0),
    (11, 12): (0, 255, 0),
    # Left Arm: Cyan
    (5, 7): (255, 255, 0),
    (7, 9): (255, 255, 0),
    # Right Arm: Magenta
    (6, 8): (255, 0, 255),
    (8, 10): (255, 0, 255),
    # Left Leg: Orange
    (11, 13): (0, 165, 255),
    (13, 15): (0, 165, 255),
    # Right Leg: Pink
    (12, 14): (203, 192, 255),
    (14, 16): (203, 192, 255),
    # Neck: White
    (5, 0): (255, 255, 255),
    (6, 0): (255, 255, 255),
}

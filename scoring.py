import numpy as np

# ---- Define Target Poses ----
TARGET_POSES = {
    "Triangle Pose": {
        "left_hip": (160, 180),  # Angles for key joints of Triangle Pose (example)
        "right_hip": (160, 180),
        "left_knee": (160, 180),
        "right_knee": (160, 180),
        "left_shoulder": (160, 180),
        "right_shoulder": (160, 180),
    },
    "Warrior Pose": {
        "left_hip": (160, 180),
        "right_hip": (160, 180),
        "left_knee": (160, 180),
        "right_knee": (160, 180),
        "left_shoulder": (160, 180),
        "right_shoulder": (160, 180),
    },
    "Tree Pose": {
        "left_hip": (160, 180),
        "right_hip": (160, 180),
        "left_knee": (160, 180),
        "right_knee": (160, 180),
        "left_shoulder": (160, 180),
        "right_shoulder": (160, 180),
    }
}

# ---- Pose Evaluation ----
def evaluate_all_poses(angles):
    """
    Compares the angles to the target poses to calculate the score.
    :param angles: Dictionary with calculated joint angles.
    :return: Best matching pose name, score and feedback.
    """
    best_pose = "No pose detected"
    best_score = 0
    feedback = []

    for pose_name, target_angles in TARGET_POSES.items():
        pose_score = 0
        for joint_name, target_range in target_angles.items():
            if joint_name in angles:
                angle = angles[joint_name]
                # Check if the angle is within the target range
                if target_range[0] <= angle <= target_range[1]:
                    pose_score += 1

        # If current pose score is better than the previous best, update
        if pose_score > best_score:
            best_score = pose_score
            best_pose = pose_name

    # Give feedback based on score
    if best_score > 0:
        feedback.append(f"Pose detected: {best_pose}")
        feedback.append(f"Score: {best_score}/10")
    else:
        feedback.append("No valid pose detected.")

    return best_pose, best_score, feedback

# ---- Grading System ----
def grade_pose(score):
    """
    Converts raw score into a final grade for the pose.
    :param score: Raw pose score.
    :return: Final grade (0-10).
    """
    if score >= 9:
        return 10
    elif score >= 8:
        return 9
    elif score >= 7:
        return 8
    elif score >= 6:
        return 7
    elif score >= 5:
        return 6
    elif score >= 4:
        return 5
    elif score >= 3:
        return 4
    elif score >= 2:
        return 3
    elif score >= 1:
        return 2
    else:
        return 1
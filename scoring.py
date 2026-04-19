import numpy as np

# ---- Anatomically Correct Target Poses ----
# Format: "joint": (target_angle_degrees, tolerance_degrees)
# Joints: left/right_elbow, left/right_shoulder, left/right_hip, left/right_knee
TARGET_POSES = {
    "Warrior II": {
        "description": "Front knee bent 90 degrees, arms extended parallel to floor.",
        "angles": {
            "right_knee":    (90,  15),
            "left_knee":     (175, 12),
            "left_shoulder": (90,  15),
            "right_shoulder":(90,  15),
            "left_elbow":    (175, 12),
            "right_elbow":   (175, 12),
        }
    },
    "Tree Pose": {
        "description": "One leg straight, other knee bent outward with foot on inner thigh.",
        "angles": {
            "left_knee":  (175, 12),
            "right_knee": (60,  20),
            "left_hip":   (175, 12),
            "right_hip":  (60,  20),
        }
    },
    "Triangle Pose": {
        "description": "Legs wide apart, torso tilted sideways, both arms extended.",
        "angles": {
            "left_knee":     (175, 12),
            "right_knee":    (175, 12),
            "left_shoulder": (90,  15),
            "right_shoulder":(90,  15),
            "left_hip":      (100, 20),
            "right_hip":     (100, 20),
        }
    },
}

# Scoring constants
ANGLE_TOLERANCE = 15.0  # degrees of perfect grace zone (shared default)
MAX_ERROR       = 35.0  # degrees beyond tolerance that results in 0 contribution
DETECTION_THRESHOLD = 6.0  # minimum score out of 10 to be "recognized"


def _joint_score(target_angle, tolerance, actual_angle):
    """Returns 0.0 – 1.0 score for a single joint with smooth degradation."""
    error = abs(target_angle - actual_angle)
    if error <= tolerance:
        return 1.0
    excess = error - tolerance
    return max(0.0, 1.0 - (excess / MAX_ERROR))


def evaluate_all_poses(current_angles):
    """
    Scan all poses and return the best matching one.
    Returns: (pose_name, score_out_of_10, feedback_list)
    """
    best_pose_name = "No pose detected"
    best_score     = 0.0
    best_feedback  = []

    for pose_name, pose_data in TARGET_POSES.items():
        target_angles = pose_data["angles"]
        joint_scores  = []
        feedback      = []

        for joint, (target, tolerance) in target_angles.items():
            if joint in current_angles:
                js = _joint_score(target, tolerance, current_angles[joint])
                joint_scores.append(js)
                if js < 0.5:
                    # Tell user which joint to fix
                    feedback.append(f"Fix {joint.replace('_', ' ')}")

        if not joint_scores:
            continue

        raw_score = round((sum(joint_scores) / len(joint_scores)) * 10, 1)

        if raw_score > best_score:
            best_score     = raw_score
            best_pose_name = pose_name
            best_feedback  = feedback

    if best_score < DETECTION_THRESHOLD:
        return "No pose detected", 0.0, []

    return best_pose_name, best_score, best_feedback
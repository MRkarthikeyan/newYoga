import numpy as np

# ---- Anatomically Correct Target Poses ----
# Format: "joint": (target_angle_degrees, tolerance_degrees)
# Joints: left/right_elbow, left/right_shoulder, left/right_hip, left/right_knee
TARGET_POSES = {
    # ── WARRIOR II ──────────────────────────────
    "Warrior II (Right)": {
        "description": "Right knee bent 90 degrees, arms extended parallel to floor.",
        "angles": {
            "right_knee":    (100, 25),  # Bent
            "left_knee":     (175, 20),  # Straight
            "left_shoulder": (90,  20),  # Arms horizontal
            "right_shoulder":(90,  20),
            "left_elbow":    (175, 20),
            "right_elbow":   (175, 20),
        }
    },
    "Warrior II (Left)": {
        "description": "Left knee bent 90 degrees, arms extended parallel to floor.",
        "angles": {
            "left_knee":     (100, 25),  # Bent
            "right_knee":    (175, 20),  # Straight
            "left_shoulder": (90,  20),  # Arms horizontal
            "right_shoulder":(90,  20),
            "left_elbow":    (175, 20),
            "right_elbow":   (175, 20),
        }
    },

    # ── TREE POSE ───────────────────────────────
    "Tree Pose (Right)": {
        "description": "Left leg straight, right knee bent outward with foot on inner thigh.",
        "angles": {
            "left_knee":  (175, 20),     # Standing leg straight
            "right_knee": (45,  30),     # Bent sharply
            "left_hip":   (175, 20),     # Standing torso straight
            "right_hip":  (135, 25),     # Knee points out/down
        }
    },
    "Tree Pose (Left)": {
        "description": "Right leg straight, left knee bent outward with foot on inner thigh.",
        "angles": {
            "right_knee": (175, 20),     # Standing leg straight
            "left_knee":  (45,  30),     # Bent sharply
            "right_hip":  (175, 20),     # Standing torso straight
            "left_hip":   (135, 25),     # Knee points out/down
        }
    },

    # ── TRIANGLE POSE ───────────────────────────
    "Triangle Pose (Right)": {
        "description": "Legs apart, torso tilted to the right, arms extended up and down.",
        "angles": {
            "left_knee":     (175, 20),  # Both legs straight
            "right_knee":    (175, 20),
            "right_hip":     (90,  30),  # Torso tilted horizontal over right leg
            "left_shoulder": (90,  25),  # Left arm reaches up (90 deg to horizontal torso)
            "left_elbow":    (175, 20),
            "right_elbow":   (175, 20),
        }
    },
    "Triangle Pose (Left)": {
        "description": "Legs apart, torso tilted to the left, arms extended up and down.",
        "angles": {
            "left_knee":     (175, 20),  # Both legs straight
            "right_knee":    (175, 20),
            "left_hip":      (90,  30),  # Torso tilted horizontal over left leg
            "right_shoulder":(90,  25),  # Right arm reaches up
            "left_elbow":    (175, 20),
            "right_elbow":   (175, 20),
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
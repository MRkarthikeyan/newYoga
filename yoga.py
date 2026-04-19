
import cv2
import mediapipe as mp
import numpy as np
import time
from scoring import evaluate_all_poses

# ---- LCD Setup ----
try:
    from rpi_lcd import LCD
    lcd = LCD()
    lcd_available = True
except ImportError:
    lcd = None
    lcd_available = False
    print("Warning: rpi_lcd not installed or not running on Raspberry Pi. LCD output disabled.")

# ---- Initialize MediaPipe Pose ----
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---- Constants ----
HOLD_DURATION = 10.0       # Seconds to hold the pose
DETECTION_THRESHOLD = 3.0  # Lowered threshold to start the pose hold

# ---- Application States ----
STATE_WAITING = "WAITING"   # No valid pose detected, waiting for competitor
STATE_HOLDING = "HOLDING"   # Pose detected, countdown is active
STATE_LOCKED = "LOCKED"    # Score is frozen and displayed

# ---- Camera Initialization ----
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not initialized.")
        return None
    return cap

# ---- Calculate Angle Function ----
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# ---- Pose Detection Logic ----
def main():
    cap = initialize_camera()

    if not cap:
        return  # Exit if camera isn't initialized

    state = STATE_WAITING
    hold_start_time = 0.0
    score_buffer = []          # Stores all raw scores during the hold
    locked_score = 0           # Final graded score
    locked_pose_name = ""      # Pose name when locked
    detected_pose_during_hold = ""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # ---- Pose Detection ----
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image.flags.writeable = True

            detected_pose = "No pose detected"
            raw_score = 0.0
            feedback = []

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                try:
                    # Calculate joint angles
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate angles
                    current_angles = {
                        "left_elbow": calculate_angle(l_shoulder, l_elbow, l_wrist),
                        "right_elbow": calculate_angle(r_shoulder, r_elbow, r_wrist),
                        "left_shoulder": calculate_angle(l_hip, l_shoulder, l_elbow),
                        "right_shoulder": calculate_angle(r_hip, r_shoulder, r_elbow),
                        "left_hip": calculate_angle(l_shoulder, l_hip, l_knee),
                        "right_hip": calculate_angle(r_shoulder, r_hip, r_knee),
                        "left_knee": calculate_angle(l_hip, l_knee, l_ankle),
                        "right_knee": calculate_angle(r_hip, r_knee, r_ankle)
                    }

                    # Detect pose and score
                    detected_pose, raw_score, feedback = evaluate_all_poses(current_angles)

                except Exception as e:
                    print(f"Error in calculating angles: {e}")
                    pass

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # ---- State Machine Logic ----
            now = time.time()

            if state == STATE_WAITING:
                if detected_pose != "No pose detected" and raw_score >= DETECTION_THRESHOLD:
                    # A valid pose detected! Start the hold timer.
                    state = STATE_HOLDING
                    hold_start_time = now
                    score_buffer = [raw_score]
                    detected_pose_during_hold = detected_pose

            elif state == STATE_HOLDING:
                elapsed = now - hold_start_time
                if detected_pose == "No pose detected" or raw_score < DETECTION_THRESHOLD:
                    # Pose not held correctly, reset
                    state = STATE_WAITING
                    score_buffer = []
                    hold_start_time = 0.0
                else:
                    score_buffer.append(raw_score)
                    detected_pose_during_hold = detected_pose

                    if elapsed >= HOLD_DURATION:
                        avg_raw = sum(score_buffer) / len(score_buffer)
                        locked_score = int(round(avg_raw))  # Graded score
                        locked_pose_name = detected_pose_during_hold
                        state = STATE_LOCKED
                        print(f"\n>>> SCORE LOCKED: {locked_pose_name} = {locked_score}/10 <<<\n")

                        # --- Update LCD ---
                        if lcd_available:
                            try:
                                lcd.text(f"Pose: {locked_pose_name}", 1)
                                lcd.text(f"Score: {locked_score}/10", 2)
                            except Exception as e:
                                print(f"LCD Error: {e}")

            # ---- UI Rendering ----
            h, w, _ = image.shape

            if state == STATE_WAITING:
                cv2.putText(image, "WAITING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif state == STATE_HOLDING:
                remaining = max(0, HOLD_DURATION - (now - hold_start_time))
                cv2.putText(image, f"HOLDING: {detected_pose_during_hold}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(image, f"Time left: {remaining:.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif state == STATE_LOCKED:
                cv2.putText(image, f"{locked_pose_name} {locked_score}/10", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow('Yoga Pose Competition Judge', image)

            # Key mappings
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                state = STATE_WAITING
                score_buffer = []
                hold_start_time = 0.0
                locked_score = 0
                locked_pose_name = ""
                if lcd_available:
                    lcd.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

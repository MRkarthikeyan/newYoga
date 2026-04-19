import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import threading
from datetime import datetime
from scoring import TARGET_POSES, evaluate_all_poses

# ---- Voice Announcement Setup ----
try:
    import pyttsx3
    _tts_available = True
except ImportError:
    _tts_available = False
    print("Info: pyttsx3 not installed. Voice disabled. Run: pip install pyttsx3")

def announce_score(pose_name, score):
    """Speak the final score in a background thread so it doesn't freeze the feed."""
    if not _tts_available:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(f"{pose_name}. Score: {score} out of 10.")
        engine.runAndWait()
    except Exception as e:
        print(f"Audio Error: {e}")

# ---- LCD Setup (Raspberry Pi only) ----
try:
    from rpi_lcd import LCD
    lcd = LCD()
    lcd_available = True
except ImportError:
    lcd = None
    lcd_available = False

# ---- MediaPipe Init ----
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

# =====================================================================
# CONFIGURATION — edit these to tune the system
# =====================================================================
HOLD_DURATION       = 5.0   # seconds to hold pose before score is locked
WARMUP_DURATION     = 3.0   # countdown shown before hold timer starts
DETECTION_THRESHOLD = 6.0   # min score /10 required to trigger the hold
PRESENCE_DURATION   = 1.5   # seconds a person must be in frame before activation
MIRROR_MODE         = True   # flip feed horizontally (mirror effect)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_log.csv")

# ---- Application States ----
STATE_NO_PERSON = "NO_PERSON"  # No human body detected
STATE_WAITING   = "WAITING"    # Person present, no valid pose yet
STATE_WARMUP    = "WARMUP"     # Valid pose detected — showing 3-2-1 countdown
STATE_HOLDING   = "HOLDING"    # Hold timer running
STATE_LOCKED    = "LOCKED"     # Score frozen, waiting for reset


# =====================================================================
# UTILITIES
# =====================================================================

def log_score(pose_name, score):
    """Append scored entry to CSV session log."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "pose", "score"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pose_name, score])


def apply_grading_band(score):
    """Convert a raw decimal score to a strict integer band (0-10)."""
    thresholds = [9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5]
    for i, t in enumerate(thresholds):
        if score >= t:
            return 10 - i
    return 0


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not initialized.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def calculate_angle(a, b, c):
    """2D angle at joint b, given three [x,y] points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# =====================================================================
# DRAWING HELPERS
# =====================================================================

# Maps joint name → MediaPipe landmark enum for skeleton coloring
_JOINT_LM = {
    "left_elbow":    mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow":   mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder":mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_hip":      mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip":     mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee":     mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee":    mp_pose.PoseLandmark.RIGHT_KNEE,
}


def draw_skeleton(image, results, bad_joints):
    """
    Draw full skeleton green, then overlay RED circles on joints that need fixing.
    bad_joints is a list of feedback strings like "Fix left knee".
    """
    if not results.pose_landmarks:
        return

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 220, 80), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2),
    )

    h, w, _ = image.shape
    for tip in bad_joints:
        # tip format: "Fix left knee" → "left_knee"
        joint_key = tip.replace("Fix ", "").replace(" ", "_")
        if joint_key in _JOINT_LM:
            lm = results.pose_landmarks.landmark[_JOINT_LM[joint_key]]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 12, (0, 0, 255), -1)
            cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)


def draw_progress_bar(image, x, y, w, h_bar, progress, color=(0, 220, 80)):
    cv2.rectangle(image, (x, y), (x + w, y + h_bar), (70, 70, 70), -1)
    fill = int(w * min(max(progress, 0), 1))
    if fill > 0:
        cv2.rectangle(image, (x, y), (x + fill, y + h_bar), color, -1)


def draw_ui(image, state, **kw):
    h, w, _ = image.shape
    fps = kw.get("fps", 0)

    # ---- FPS (always visible, bottom-left) ----
    cv2.putText(image, f"FPS:{int(fps)}", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

    # ---- Mirror indicator ----
    mirror_label = "[M] Mirror: ON" if kw.get("mirror", True) else "[M] Mirror: OFF"
    cv2.putText(image, mirror_label, (w - 190, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    if state == STATE_NO_PERSON:
        cv2.rectangle(image, (0, 0), (w, 80), (45, 45, 45), -1)
        cv2.putText(image, "Step into the frame...", (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)

    elif state == STATE_WAITING:
        detected = kw.get("detected_pose", "")
        raw      = kw.get("raw_score", 0)
        cv2.rectangle(image, (0, 0), (w, 95), (30, 100, 160), -1)
        cv2.putText(image, "WAITING FOR POSE...", (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        if detected and detected != "No pose detected":
            cv2.putText(image, f"Close: {detected} ({raw}/10)  need >={DETECTION_THRESHOLD}",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, f"Hold a pose with score >= {DETECTION_THRESHOLD}/10 to start",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    elif state == STATE_WARMUP:
        countdown = kw.get("countdown", WARMUP_DURATION)
        pose_name = kw.get("pose_name", "")
        progress  = 1.0 - (countdown / WARMUP_DURATION)
        cv2.rectangle(image, (0, 0), (w, 100), (0, 140, 200), -1)
        cv2.putText(image, f"GET READY: {pose_name}", (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Hold! Starting in {countdown:.1f}s...", (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        draw_progress_bar(image, 10, 85, w - 20, 10, progress, color=(0, 220, 220))

    elif state == STATE_HOLDING:
        remaining = kw.get("remaining", 0)
        progress  = kw.get("progress", 0)
        pose_name = kw.get("pose_name", "")
        raw_score = kw.get("raw_score", 0)
        feedback  = kw.get("feedback", [])
        cv2.rectangle(image, (0, 0), (w, 100), (0, 155, 255), -1)
        cv2.putText(image, f"HOLDING: {pose_name}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Time left: {remaining:.1f}s", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Live: {raw_score}/10", (w - 175, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        draw_progress_bar(image, 10, 85, w - 20, 10, progress, color=(0, 220, 80))
        # Show up to 3 fix-tips on screen
        if feedback:
            y_off = 125
            cv2.putText(image, "Fix:", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2, cv2.LINE_AA)
            for tip in feedback[:3]:
                y_off += 26
                cv2.putText(image, f"  {tip}", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1, cv2.LINE_AA)

    elif state == STATE_LOCKED:
        locked_score   = kw.get("locked_score", 0)
        locked_pose    = kw.get("locked_pose", "")
        competitor_num = kw.get("competitor_num", 1)
        cv2.rectangle(image, (0, 0), (w, 130), (0, 150, 0), -1)
        cv2.putText(image, f"SCORE LOCKED  — Competitor #{competitor_num}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(image, locked_pose, (15, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{locked_score}/10", (w - 210, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, "Press 'R' to reset for next competitor", (15, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1, cv2.LINE_AA)


# =====================================================================
# MAIN
# =====================================================================

def main():
    cap = initialize_camera()
    if not cap:
        return

    # --- State variables ---
    state                    = STATE_NO_PERSON
    presence_start_time      = 0.0
    warmup_start_time        = 0.0
    hold_start_time          = 0.0
    score_buffer             = []
    locked_score             = 0
    locked_pose_name         = ""
    detected_pose_during_hold = ""
    competitor_num           = 1
    pTime                    = 0
    mirror                   = MIRROR_MODE

    print("=" * 52)
    print("  YOGA POSE COMPETITION JUDGE  (Enhanced v2)")
    print("=" * 52)
    print(f"  Hold Duration   : {int(HOLD_DURATION)}s")
    print(f"  Warmup Countdown: {int(WARMUP_DURATION)}s")
    print(f"  Min Threshold   : {DETECTION_THRESHOLD}/10")
    print(f"  Mirror Mode     : {'ON' if mirror else 'OFF'}")
    print(f"  Voice (TTS)     : {'ON' if _tts_available else 'OFF (pip install pyttsx3)'}")
    print(f"  Session Log     : {LOG_FILE}")
    print("  Keys: [R] Reset  [M] Mirror  [Q] Quit")
    print("=" * 52)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            if mirror:
                image = cv2.flip(image, 1)

            # ---- Pose Detection ----
            image.flags.writeable = False
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            detected_pose  = "No pose detected"
            raw_score      = 0.0
            feedback       = []
            person_present = results.pose_landmarks is not None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                def lm(part):
                    return [lms[part.value].x, lms[part.value].y]

                try:
                    current_angles = {
                        "left_elbow":    calculate_angle(lm(mp_pose.PoseLandmark.LEFT_SHOULDER),  lm(mp_pose.PoseLandmark.LEFT_ELBOW),    lm(mp_pose.PoseLandmark.LEFT_WRIST)),
                        "right_elbow":   calculate_angle(lm(mp_pose.PoseLandmark.RIGHT_SHOULDER), lm(mp_pose.PoseLandmark.RIGHT_ELBOW),   lm(mp_pose.PoseLandmark.RIGHT_WRIST)),
                        "left_shoulder": calculate_angle(lm(mp_pose.PoseLandmark.LEFT_HIP),       lm(mp_pose.PoseLandmark.LEFT_SHOULDER), lm(mp_pose.PoseLandmark.LEFT_ELBOW)),
                        "right_shoulder":calculate_angle(lm(mp_pose.PoseLandmark.RIGHT_HIP),      lm(mp_pose.PoseLandmark.RIGHT_SHOULDER),lm(mp_pose.PoseLandmark.RIGHT_ELBOW)),
                        "left_hip":      calculate_angle(lm(mp_pose.PoseLandmark.LEFT_SHOULDER),  lm(mp_pose.PoseLandmark.LEFT_HIP),      lm(mp_pose.PoseLandmark.LEFT_KNEE)),
                        "right_hip":     calculate_angle(lm(mp_pose.PoseLandmark.RIGHT_SHOULDER), lm(mp_pose.PoseLandmark.RIGHT_HIP),     lm(mp_pose.PoseLandmark.RIGHT_KNEE)),
                        "left_knee":     calculate_angle(lm(mp_pose.PoseLandmark.LEFT_HIP),       lm(mp_pose.PoseLandmark.LEFT_KNEE),     lm(mp_pose.PoseLandmark.LEFT_ANKLE)),
                        "right_knee":    calculate_angle(lm(mp_pose.PoseLandmark.RIGHT_HIP),      lm(mp_pose.PoseLandmark.RIGHT_KNEE),    lm(mp_pose.PoseLandmark.RIGHT_ANKLE)),
                    }
                    detected_pose, raw_score, feedback = evaluate_all_poses(current_angles)
                except Exception as e:
                    print(f"Angle error: {e}")

                draw_skeleton(image, results, feedback)

            # ---- State Machine ----
            now = time.time()

            if state == STATE_NO_PERSON:
                if person_present:
                    if presence_start_time == 0.0:
                        presence_start_time = now
                    elif (now - presence_start_time) >= PRESENCE_DURATION:
                        state = STATE_WAITING
                        presence_start_time = 0.0
                        print("Person detected — ready.")
                else:
                    presence_start_time = 0.0

            elif state == STATE_WAITING:
                if not person_present:
                    state = STATE_NO_PERSON
                    presence_start_time = 0.0
                elif detected_pose != "No pose detected" and raw_score >= DETECTION_THRESHOLD:
                    state = STATE_WARMUP
                    warmup_start_time = now
                    detected_pose_during_hold = detected_pose
                    print(f"Pose '{detected_pose}' detected! Warmup starting...")

            elif state == STATE_WARMUP:
                elapsed_warmup = now - warmup_start_time
                if not person_present or raw_score < DETECTION_THRESHOLD:
                    # Broke pose during warmup — go back to waiting
                    state = STATE_WAITING
                    warmup_start_time = 0.0
                    print("Pose lost during warmup. Back to waiting.")
                elif elapsed_warmup >= WARMUP_DURATION:
                    state = STATE_HOLDING
                    hold_start_time = now
                    score_buffer = [raw_score]
                    print(f"Holding '{detected_pose_during_hold}'...")

            elif state == STATE_HOLDING:
                elapsed = now - hold_start_time
                if not person_present or raw_score < DETECTION_THRESHOLD:
                    state = STATE_WAITING
                    score_buffer = []
                    hold_start_time = 0.0
                    print("Pose broken! Resetting hold timer.")
                else:
                    score_buffer.append(raw_score)
                    detected_pose_during_hold = detected_pose
                    if elapsed >= HOLD_DURATION:
                        avg_raw      = sum(score_buffer) / len(score_buffer)
                        locked_score = apply_grading_band(avg_raw)
                        locked_pose_name = detected_pose_during_hold
                        state = STATE_LOCKED
                        print(f"\n>>> SCORE LOCKED: {locked_pose_name} = {locked_score}/10 "
                              f"(avg {avg_raw:.1f} over {len(score_buffer)} frames) <<<\n")
                        log_score(locked_pose_name, locked_score)
                        threading.Thread(
                            target=announce_score,
                            args=(locked_pose_name, locked_score),
                            daemon=True
                        ).start()
                        if lcd_available:
                            try:
                                lcd.text(f"{locked_pose_name[:16]}", 1)
                                lcd.text(f"Score: {locked_score}/10", 2)
                            except Exception as e:
                                print(f"LCD Error: {e}")

            # STATE_LOCKED: frozen — wait for 'R'

            # ---- FPS ----
            cTime = time.time()
            fps   = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            # ---- Draw UI ----
            common = dict(fps=fps, mirror=mirror)
            if state == STATE_NO_PERSON:
                draw_ui(image, state, **common)
            elif state == STATE_WAITING:
                draw_ui(image, state, detected_pose=detected_pose, raw_score=raw_score, **common)
            elif state == STATE_WARMUP:
                countdown = max(0, WARMUP_DURATION - (now - warmup_start_time))
                draw_ui(image, state, pose_name=detected_pose_during_hold, countdown=countdown, **common)
            elif state == STATE_HOLDING:
                elapsed   = now - hold_start_time
                remaining = max(0, HOLD_DURATION - elapsed)
                progress  = elapsed / HOLD_DURATION
                draw_ui(image, state,
                        pose_name=detected_pose_during_hold,
                        remaining=remaining, progress=progress,
                        raw_score=raw_score, feedback=feedback, **common)
            elif state == STATE_LOCKED:
                draw_ui(image, state,
                        locked_score=locked_score, locked_pose=locked_pose_name,
                        competitor_num=competitor_num, **common)

            cv2.imshow("Yoga Pose Competition Judge", image)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                state = STATE_WAITING
                score_buffer = []
                hold_start_time = 0.0
                warmup_start_time = 0.0
                locked_score = 0
                locked_pose_name = ""
                competitor_num += 1
                print(f">>> RESET — Ready for Competitor #{competitor_num} <<<")
                if lcd_available:
                    try:
                        lcd.clear()
                    except Exception:
                        pass
            elif key == ord('m'):
                mirror = not mirror
                print(f"Mirror mode: {'ON' if mirror else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()

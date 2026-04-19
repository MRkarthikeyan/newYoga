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

def lcd_show(line1, line2=""):
    """
    Safely write two lines to the LCD.
    Clears the display first so old text never bleeds through.
    Falls back silently if LCD is unavailable.
    """
    if not lcd_available:
        return
    try:
        lcd.clear()
        time.sleep(0.05)       # brief I2C settle after clear
        lcd.text(line1[:16], 1)
        if line2:
            lcd.text(line2[:16], 2)
    except Exception as e:
        print(f"LCD Error: {e}")

# ---- MediaPipe Init ----
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

# =====================================================================
# CONFIGURATION — edit these to tune the system
# =====================================================================
ANALYSIS_DURATION   = 5.0   # seconds the system auto-analyses the pose
DETECTION_THRESHOLD = 6.0   # min score /10 to consider a pose recognised
PRESENCE_DURATION   = 1.5   # seconds a person must be in frame before scan starts
MIRROR_MODE         = True   # flip feed horizontally (mirror effect)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_log.csv")

# ---- Application States ----
STATE_NO_PERSON = "NO_PERSON"  # No human body detected
STATE_WAITING   = "WAITING"    # Person present — waiting 1 s before analysis
STATE_ANALYZING = "ANALYZING"  # Auto 5-second pose scan is running
STATE_LOCKED    = "LOCKED"     # Result frozen, waiting for reset


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

    elif state == STATE_ANALYZING:
        elapsed   = kw.get("elapsed", 0)
        remaining = kw.get("remaining", ANALYSIS_DURATION)
        progress  = kw.get("progress", 0)
        pose_name = kw.get("detected_pose", "Scanning...")
        raw_score = kw.get("raw_score", 0)
        feedback  = kw.get("feedback", [])
        # Pulsing purple/violet header
        cv2.rectangle(image, (0, 0), (w, 100), (130, 0, 200), -1)
        cv2.putText(image, "ANALYSING POSE...", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        # Live best-match label
        if pose_name and pose_name != "No pose detected":
            cv2.putText(image, f"Detecting: {pose_name}  ({raw_score}/10)", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 200, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, "No pose recognised yet — hold a pose!", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 140, 255), 1, cv2.LINE_AA)
        # Countdown bar (fills up as time runs)
        cv2.putText(image, f"{remaining:.1f}s left", (w - 145, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        draw_progress_bar(image, 10, 85, w - 20, 10, progress, color=(180, 80, 255))
        # Fix tips
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
        pose_found     = kw.get("pose_found", True)

        if pose_found:
            # Green header — pose was recognised
            cv2.rectangle(image, (0, 0), (w, 130), (0, 150, 0), -1)
            cv2.putText(image, f"RESULT  — Competitor #{competitor_num}", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(image, locked_pose, (15, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{locked_score}/10", (w - 210, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 255, 255), 4, cv2.LINE_AA)
        else:
            # Red header — no pose detected
            cv2.rectangle(image, (0, 0), (w, 130), (0, 0, 180), -1)
            cv2.putText(image, f"RESULT  — Competitor #{competitor_num}", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "No Pose Detected", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "0/10", (w - 150, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.8, (200, 200, 255), 4, cv2.LINE_AA)

        cv2.putText(image, "Press 'R' to analyse next competitor", (15, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


# =====================================================================
# MAIN
# =====================================================================

def main():
    cap = initialize_camera()
    if not cap:
        return

    # --- State variables ---
    state               = STATE_NO_PERSON
    presence_start_time = 0.0
    analysis_start_time = 0.0        # when the 5-second scan started
    score_buffer        = []         # (pose_name, raw_score) tuples during analysis
    locked_score        = 0
    locked_pose_name    = ""
    pose_found          = False      # True if a valid pose was identified
    competitor_num      = 1
    pTime               = 0
    mirror              = MIRROR_MODE

    print("=" * 52)
    print("  YOGA POSE COMPETITION JUDGE  (Enhanced v3)")
    print("=" * 52)
    print(f"  Analysis Window : {int(ANALYSIS_DURATION)}s (auto-start on entry)")
    print(f"  Min Threshold   : {DETECTION_THRESHOLD}/10")
    print(f"  Mirror Mode     : {'ON' if mirror else 'OFF'}")
    print(f"  Voice (TTS)     : {'ON' if _tts_available else 'OFF (pip install pyttsx3)'}")
    print(f"  Session Log     : {LOG_FILE}")
    print("  Keys: [R] Reset  [M] Mirror  [Q] Quit")
    print("=" * 52)

    # Show startup message on LCD
    lcd_show("Enter into frame", "Yoga Judge Ready")

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
                        # Person has been in frame long enough — start the analysis immediately
                        state = STATE_ANALYZING
                        analysis_start_time = now
                        score_buffer = []
                        presence_start_time = 0.0
                        print(f"Person detected — analysis started ({int(ANALYSIS_DURATION)}s window)")
                        lcd_show("Analysing...", f"Hold for {int(ANALYSIS_DURATION)}s")
                else:
                    presence_start_time = 0.0

            elif state == STATE_WAITING:
                # Brief intermediate state (currently unused but kept for future use)
                if person_present:
                    state = STATE_ANALYZING
                    analysis_start_time = now
                    score_buffer = []
                else:
                    state = STATE_NO_PERSON
                    presence_start_time = 0.0

            elif state == STATE_ANALYZING:
                elapsed = now - analysis_start_time

                # Always collect scores — even "No pose detected" frames (score 0)
                if detected_pose != "No pose detected" and raw_score >= DETECTION_THRESHOLD:
                    score_buffer.append((detected_pose, raw_score))

                if elapsed >= ANALYSIS_DURATION:
                    # ---- Analysis window over — compute result ----
                    if score_buffer:
                        # Pick the most frequently detected pose among top-scoring frames
                        from collections import Counter
                        pose_counts = Counter(p for p, _ in score_buffer)
                        best_pose_name = pose_counts.most_common(1)[0][0]
                        # Average score only for frames where that pose was detected
                        relevant_scores = [s for p, s in score_buffer if p == best_pose_name]
                        avg_raw      = sum(relevant_scores) / len(relevant_scores)
                        locked_score = apply_grading_band(avg_raw)
                        locked_pose_name = best_pose_name
                        pose_found   = True
                        print(f"\n>>> RESULT: {locked_pose_name} = {locked_score}/10 "
                              f"({len(relevant_scores)} valid frames out of {len(score_buffer)}) <<<\n")
                        log_score(locked_pose_name, locked_score)
                        threading.Thread(
                            target=announce_score,
                            args=(locked_pose_name, locked_score),
                            daemon=True
                        ).start()
                        lcd_show(locked_pose_name, f"Score: {locked_score}/10")
                        # No valid pose detected during the entire window
                        locked_score     = 0
                        locked_pose_name = "No Pose Detected"
                        pose_found       = False
                        print("\n>>> RESULT: No pose detected <<<\n")
                        log_score("No Pose Detected", 0)
                        threading.Thread(
                            target=announce_score,
                            args=("No pose detected", 0),
                            daemon=True
                        ).start()
                        lcd_show("No Pose Detected", "Score: 0/10")

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
                draw_ui(image, state, **common)
            elif state == STATE_ANALYZING:
                elapsed_a   = now - analysis_start_time
                remaining_a = max(0, ANALYSIS_DURATION - elapsed_a)
                progress_a  = elapsed_a / ANALYSIS_DURATION
                draw_ui(image, state,
                        elapsed=elapsed_a, remaining=remaining_a, progress=progress_a,
                        detected_pose=detected_pose, raw_score=raw_score,
                        feedback=feedback, **common)
            elif state == STATE_LOCKED:
                draw_ui(image, state,
                        locked_score=locked_score, locked_pose=locked_pose_name,
                        pose_found=pose_found, competitor_num=competitor_num, **common)

            cv2.imshow("Yoga Pose Competition Judge", image)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                state = STATE_NO_PERSON
                score_buffer = []
                analysis_start_time = 0.0
                presence_start_time = 0.0
                locked_score = 0
                locked_pose_name = ""
                pose_found = False
                competitor_num += 1
                print(f">>> RESET — Ready for Competitor #{competitor_num} <<<")
                lcd_show("Enter into frame", f"Competitor #{competitor_num}")
            elif key == ord('m'):
                mirror = not mirror
                print(f"Mirror mode: {'ON' if mirror else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()

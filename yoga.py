import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import threading
from datetime import datetime
from scoring import TARGET_POSES, evaluate_all_poses

# ---- TTS ----
try:
    import pyttsx3
    _tts_available = True
except ImportError:
    _tts_available = False

import os

def announce_score(pose_name, score):
    try:
        # Run espeak directly as a system command in the background
        # -s 150 is the speed (words per minute)
        # 2>/dev/null hides the terminal warnings espeak usually prints
        os.system(f'espeak -s 150 "{pose_name}. Score: {score} out of 10." 2>/dev/null')
    except Exception as e:
        print(f"Audio Error: {e}")

# ---- LCD ----
try:
    from rpi_lcd import LCD
    lcd = LCD()
    lcd_available = True
    print("Info: LCD initialized successfully.")
except Exception as e:
    lcd = None
    lcd_available = False
    print(f"Warning: LCD disabled — {e}")

def lcd_show(line1, line2=""):
    if not lcd_available:
        return
    try:
        lcd.clear()
        time.sleep(0.05)
        lcd.text(line1[:16], 1)
        if line2:
            lcd.text(line2[:16], 2)
    except Exception as e:
        print(f"LCD Error: {e}")

# ---- MediaPipe ----
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

# =====================================================================
# CONFIG
# =====================================================================
DETECT_DURATION   = 5.0   # Step 1: seconds pose must be held at ≥70% to confirm
SCORE_DURATION    = 5.0   # Step 2: seconds to collect averaged score
DETECT_THRESHOLD  = 7.0   # 70% confidence needed to start/continue detection timer
PRESENCE_DURATION = 1.5   # seconds person must be in frame before detection starts
MIRROR_MODE       = True
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_log.csv")

# ---- States ----
STATE_NO_PERSON = "NO_PERSON"  # No human detected
STATE_DETECTING = "DETECTING"  # Step 1: 5s detection countdown (resets on pose break)
STATE_SCORING   = "SCORING"    # Step 2: 5s scoring window (collects average)
STATE_LOCKED    = "LOCKED"     # Final score displayed

# =====================================================================
# UTILITIES
# =====================================================================
def log_score(pose_name, score):
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "pose", "score"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pose_name, score])

def apply_grading_band(score):
    for i, t in enumerate([9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5]):
        if score >= t:
            return 10 - i
    return 0

def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    angle = np.abs(np.degrees(
        np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ))
    return 360 - angle if angle > 180 else angle

# =====================================================================
# SKELETON DRAWING
# =====================================================================
_JLM = {
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
    if not results.pose_landmarks:
        return
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,220,80), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=2),
    )
    h, w, _ = image.shape
    for tip in bad_joints:
        key = tip.replace("Fix ", "").replace(" ", "_")
        if key in _JLM:
            lm = results.pose_landmarks.landmark[_JLM[key]]
            cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 12, (0,0,255), -1)
            cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 12, (255,255,255), 2)

def draw_bar(img, x, y, bw, bh, progress, color=(0,220,80)):
    cv2.rectangle(img, (x,y), (x+bw, y+bh), (70,70,70), -1)
    fill = int(bw * min(max(progress, 0), 1))
    if fill > 0:
        cv2.rectangle(img, (x,y), (x+fill, y+bh), color, -1)

# =====================================================================
# UI RENDERING
# =====================================================================
def draw_ui(image, state, **kw):
    h, w, _ = image.shape
    fps    = kw.get("fps", 0)
    mirror = kw.get("mirror", True)
    cv2.putText(image, f"FPS:{int(fps)}", (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,255), 1)
    cv2.putText(image, "[M]Mirror:ON" if mirror else "[M]Mirror:OFF",
                (w-160, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    if state == STATE_NO_PERSON:
        cv2.rectangle(image, (0,0), (w,80), (45,45,45), -1)
        cv2.putText(image, "Step into the frame...", (15,52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180,180,180), 2, cv2.LINE_AA)

    elif state == STATE_DETECTING:
        pose_name  = kw.get("pose_name", "")
        raw_score  = kw.get("raw_score", 0.0)
        remaining  = kw.get("remaining", DETECT_DURATION)
        progress   = kw.get("progress", 0.0)
        feedback   = kw.get("feedback", [])
        conf_pct   = int(raw_score * 10)

        cv2.rectangle(image, (0,0), (w,105), (30,100,160), -1)

        if pose_name:
            # Pose detected above threshold — show name, confidence, countdown
            cv2.putText(image, f"DETECTING: {pose_name}", (15,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Confidence: {conf_pct}%", (15,68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,255,180), 1, cv2.LINE_AA)
            cv2.putText(image, f"{remaining:.1f}s", (w-110,68),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2, cv2.LINE_AA)
            draw_bar(image, 10, 88, w-20, 10, progress, color=(0,200,255))
            # Fix tips
            y = 128
            for tip in feedback[:3]:
                cv2.putText(image, f"  {tip}", (10,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1, cv2.LINE_AA)
                y += 24
        else:
            # No valid pose yet
            cv2.putText(image, "Detecting Pose...", (15,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Hold a pose clearly (need >=70% confidence)", (15,68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
            draw_bar(image, 10, 88, w-20, 10, 0, color=(0,200,255))

    elif state == STATE_SCORING:
        pose_name  = kw.get("pose_name", "")
        remaining  = kw.get("remaining", SCORE_DURATION)
        progress   = kw.get("progress", 0.0)
        avg        = kw.get("avg_so_far", 0.0)

        cv2.rectangle(image, (0,0), (w,105), (0,140,255), -1)
        cv2.putText(image, f"SCORING: {pose_name}", (15,38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Hold! {remaining:.1f}s left", (15,72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Avg: {avg:.1f}/10", (w-185,72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)
        draw_bar(image, 10, 88, w-20, 10, progress, color=(0,220,80))

    elif state == STATE_LOCKED:
        score = kw.get("locked_score", 0)
        pose  = kw.get("locked_pose", "")
        num   = kw.get("competitor_num", 1)
        cv2.rectangle(image, (0,0), (w,130), (0,150,0), -1)
        cv2.putText(image, f"FINAL SCORE — Competitor #{num}", (15,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1, cv2.LINE_AA)
        cv2.putText(image, pose, (15,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{score}/10", (w-205,118),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(image, "Press 'R' for next competitor", (15,118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1, cv2.LINE_AA)

# =====================================================================
# MAIN
# =====================================================================
def main():
    cap = init_camera()
    if not cap:
        return

    state               = STATE_NO_PERSON
    presence_start      = 0.0
    detect_start        = 0.0
    current_detect_pose = ""   # pose being timed in DETECTING
    score_start         = 0.0
    score_buffer        = []
    locked_score        = 0
    locked_pose         = ""
    competitor_num      = 1
    pTime               = 0
    mirror              = MIRROR_MODE

    print("=" * 52)
    print("  YOGA POSE COMPETITION JUDGE  (v4)")
    print("=" * 52)
    print(f"  Step 1 — Detection : {int(DETECT_DURATION)}s (resets if pose breaks/changes)")
    print(f"  Step 2 — Scoring   : {int(SCORE_DURATION)}s (average score computed)")
    print(f"  Threshold          : {DETECT_THRESHOLD}/10 ({int(DETECT_THRESHOLD*10)}% confidence)")
    print("  Keys: [R] Reset  [M] Mirror  [Q] Quit")
    print("=" * 52)

    lcd_show("Enter into frame", "Yoga Judge v4")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                break

            if mirror:
                image = cv2.flip(image, 1)

            # ── Pose Detection ──────────────────────────────────────
            image.flags.writeable = False
            results = pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            detected_pose  = "No pose detected"
            raw_score      = 0.0
            feedback       = []
            person_present = results.pose_landmarks is not None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                def lm(p): return [lms[p.value].x, lms[p.value].y]
                try:
                    angles = {
                        "left_elbow":    calc_angle(lm(mp_pose.PoseLandmark.LEFT_SHOULDER),  lm(mp_pose.PoseLandmark.LEFT_ELBOW),    lm(mp_pose.PoseLandmark.LEFT_WRIST)),
                        "right_elbow":   calc_angle(lm(mp_pose.PoseLandmark.RIGHT_SHOULDER), lm(mp_pose.PoseLandmark.RIGHT_ELBOW),   lm(mp_pose.PoseLandmark.RIGHT_WRIST)),
                        "left_shoulder": calc_angle(lm(mp_pose.PoseLandmark.LEFT_HIP),       lm(mp_pose.PoseLandmark.LEFT_SHOULDER), lm(mp_pose.PoseLandmark.LEFT_ELBOW)),
                        "right_shoulder":calc_angle(lm(mp_pose.PoseLandmark.RIGHT_HIP),      lm(mp_pose.PoseLandmark.RIGHT_SHOULDER),lm(mp_pose.PoseLandmark.RIGHT_ELBOW)),
                        "left_hip":      calc_angle(lm(mp_pose.PoseLandmark.LEFT_SHOULDER),  lm(mp_pose.PoseLandmark.LEFT_HIP),      lm(mp_pose.PoseLandmark.LEFT_KNEE)),
                        "right_hip":     calc_angle(lm(mp_pose.PoseLandmark.RIGHT_SHOULDER), lm(mp_pose.PoseLandmark.RIGHT_HIP),     lm(mp_pose.PoseLandmark.RIGHT_KNEE)),
                        "left_knee":     calc_angle(lm(mp_pose.PoseLandmark.LEFT_HIP),       lm(mp_pose.PoseLandmark.LEFT_KNEE),     lm(mp_pose.PoseLandmark.LEFT_ANKLE)),
                        "right_knee":    calc_angle(lm(mp_pose.PoseLandmark.RIGHT_HIP),      lm(mp_pose.PoseLandmark.RIGHT_KNEE),    lm(mp_pose.PoseLandmark.RIGHT_ANKLE)),
                    }
                    detected_pose, raw_score, feedback = evaluate_all_poses(angles)
                except Exception as e:
                    print(f"Angle error: {e}")
                draw_skeleton(image, results, feedback)

            # ── State Machine ───────────────────────────────────────
            now = time.time()

            # ── NO_PERSON ──
            if state == STATE_NO_PERSON:
                if person_present:
                    if presence_start == 0.0:
                        presence_start = now
                    elif (now - presence_start) >= PRESENCE_DURATION:
                        state = STATE_DETECTING
                        detect_start = now
                        current_detect_pose = ""
                        presence_start = 0.0
                        print("Person detected — Step 1: detecting pose...")
                        lcd_show("Detecting Pose", "Hold a pose...")
                else:
                    presence_start = 0.0

            # ── DETECTING (Step 1) ──
            elif state == STATE_DETECTING:
                if not person_present:
                    # Person left — go back to no person
                    state = STATE_NO_PERSON
                    detect_start = 0.0
                    current_detect_pose = ""
                    presence_start = 0.0
                    lcd_show("Enter into frame", "Yoga Judge v4")

                elif raw_score < DETECT_THRESHOLD or detected_pose == "No pose detected":
                    # Confidence dropped below 70% — reset detection timer
                    if current_detect_pose:
                        print(f"Pose dropped below 70% — detection timer reset.")
                    detect_start = now
                    current_detect_pose = ""

                else:
                    # Confidence >= 70%
                    if detected_pose != current_detect_pose:
                        # Pose changed — reset timer for new pose
                        print(f"Pose: '{detected_pose}' ({raw_score}/10) — timer reset.")
                        detect_start = now
                        current_detect_pose = detected_pose
                        lcd_show("Detecting:", detected_pose[:16])

                    elapsed = now - detect_start
                    if elapsed >= DETECT_DURATION:
                        # Pose held for 5s at ≥70% — confirmed! Start scoring.
                        state = STATE_SCORING
                        score_start = now
                        score_buffer = [raw_score]
                        print(f"\n>>> POSE CONFIRMED: {current_detect_pose} — Step 2: scoring...\n")
                        lcd_show("Pose Confirmed!", current_detect_pose[:16])

            # ── SCORING (Step 2) ──
            elif state == STATE_SCORING:
                elapsed = now - score_start

                if not person_present or raw_score < DETECT_THRESHOLD or detected_pose == "No pose detected":
                    # Pose broken — Step 3: go back to Step 1
                    print("Pose broken during scoring! Back to Step 1.")
                    state = STATE_DETECTING
                    detect_start = now
                    current_detect_pose = ""
                    score_buffer = []
                    lcd_show("Detecting Pose", "Hold a pose...")
                else:
                    score_buffer.append(raw_score)

                    if elapsed >= SCORE_DURATION:
                        # 5s scoring complete — compute average
                        avg_raw      = sum(score_buffer) / len(score_buffer)
                        locked_score = apply_grading_band(avg_raw)
                        locked_pose  = current_detect_pose
                        state        = STATE_LOCKED
                        print(f"\n>>> FINAL: {locked_pose} = {locked_score}/10 "
                              f"(avg {avg_raw:.2f} from {len(score_buffer)} frames) <<<\n")
                        log_score(locked_pose, locked_score)
                        lcd_show(locked_pose, f"Score: {locked_score}/10")
                        threading.Thread(
                            target=announce_score,
                            args=(locked_pose, locked_score),
                            daemon=True
                        ).start()

            # STATE_LOCKED — frozen until 'R'

            # ── FPS ─────────────────────────────────────────────────
            cTime = time.time()
            fps   = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            # ── Draw UI ──────────────────────────────────────────────
            common = dict(fps=fps, mirror=mirror)

            if state == STATE_NO_PERSON:
                draw_ui(image, state, **common)

            elif state == STATE_DETECTING:
                has_pose = bool(current_detect_pose) and raw_score >= DETECT_THRESHOLD
                elapsed  = now - detect_start if has_pose else 0
                remaining = max(0, DETECT_DURATION - elapsed)
                progress  = (elapsed / DETECT_DURATION) if has_pose else 0.0
                draw_ui(image, state,
                        pose_name=current_detect_pose if has_pose else "",
                        raw_score=raw_score, remaining=remaining,
                        progress=progress, feedback=feedback, **common)

            elif state == STATE_SCORING:
                elapsed   = now - score_start
                remaining = max(0, SCORE_DURATION - elapsed)
                progress  = elapsed / SCORE_DURATION
                avg       = sum(score_buffer) / len(score_buffer) if score_buffer else 0.0
                draw_ui(image, state,
                        pose_name=current_detect_pose, remaining=remaining,
                        progress=progress, avg_so_far=avg, **common)

            elif state == STATE_LOCKED:
                draw_ui(image, state,
                        locked_score=locked_score, locked_pose=locked_pose,
                        competitor_num=competitor_num, **common)

            cv2.imshow("Yoga Pose Competition Judge", image)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                state = STATE_NO_PERSON
                detect_start = 0.0
                current_detect_pose = ""
                score_buffer = []
                score_start = 0.0
                presence_start = 0.0
                locked_score = 0
                locked_pose = ""
                competitor_num += 1
                print(f">>> RESET — Competitor #{competitor_num} <<<")
                lcd_show("Enter into frame", f"Competitor #{competitor_num}")
            elif key == ord('m'):
                mirror = not mirror
                print(f"Mirror: {'ON' if mirror else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession log: {LOG_FILE}")

if __name__ == "__main__":
    main()

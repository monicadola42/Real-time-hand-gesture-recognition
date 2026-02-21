import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque, Counter
from time import time

# ---------- CONFIG ----------
REF_FOLDER = Path(r"E:\PYTHON\ref_gestures")
REF_FOLDER.mkdir(exist_ok=True)
CAM_INDEX = 0
MAX_HANDS = 1
MIN_DETECTION_CONF = 0.6
MIN_TRACKING_CONF = 0.6
SMOOTH_WINDOW = 7
SIMILARITY_THRESHOLD = 0.68
SAVE_PREFIX = "ref_"
# ----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_landmarks_from_bgr(bgr_image, hands_processor):
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    res = hands_processor.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

def normalize_landmarks(pts):
    pts = pts.copy()
    wrist = pts[0]
    pts -= wrist
    tips = pts[[4, 8, 12, 16, 20]]
    scale = np.mean(np.linalg.norm(tips, axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    ref_vec = pts[9]
    angle = np.arctan2(ref_vec[1], ref_vec[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    pts = pts.dot(R.T)
    return pts.flatten()

def similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def build_ref_db(hands_processor):
    db = []
    files = sorted([p for p in REF_FOLDER.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to read", p)
            continue
        lm = get_hand_landmarks_from_bgr(img, hands_processor)
        if lm is None:
            print("No hand detected in ref:", p.name)
            continue
        desc = normalize_landmarks(lm)
        db.append({"name": p.name, "desc": desc, "img": img})
        print("Loaded ref:", p.name)
    print("Reference DB size:", len(db))
    return db

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS,
                        min_detection_confidence=MIN_DETECTION_CONF,
                        min_tracking_confidence=MIN_TRACKING_CONF) as hands:

        ref_db = build_ref_db(hands)

        label_queue = deque(maxlen=SMOOTH_WINDOW)
        score_queue = deque(maxlen=SMOOTH_WINDOW)
        prev_time = time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            lm = get_hand_landmarks_from_bgr(frame, hands)
            best_label = "No hand"
            best_score = 0.0
            best_img = None   # <-- store most similar uploaded photo

            if lm is not None:
                res = hands.process(frame_rgb)
                if res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                desc_live = normalize_landmarks(lm)
                if ref_db:
                    scores = []
                    for entry in ref_db:
                        s = similarity(desc_live, entry["desc"])
                        scores.append((entry["name"], s, entry["img"]))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    best_label, best_score, best_img = scores[0]

            display_label = best_label if best_score >= SIMILARITY_THRESHOLD else "Unknown"
            label_queue.append(display_label)
            score_queue.append(best_score)

            most_common = Counter(label_queue).most_common(1)[0][0]
            avg_score = sum(score_queue) / len(score_queue) if len(score_queue) > 0 else 0.0

            # FPS
            cur_time = time()
            dt = cur_time - prev_time
            fps = (1.0 / dt) if dt > 0 else fps
            prev_time = cur_time

            # ------------ SHOW CAMERA + SIMILAR UPLOADED PHOTO ------------
            h, w = frame.shape[:2]

            if best_img is not None and best_score >= SIMILARITY_THRESHOLD:
                ref_resized = cv2.resize(best_img, (w, h))
            else:
                ref_resized = np.zeros_like(frame)  # black if no good match

            combined = np.hstack([frame, ref_resized])

            cv2.putText(combined, f"Gesture: {most_common}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(combined, f"Score: {avg_score:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(combined, "Left: live camera  Right: most similar uploaded photo",
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("Gesture → Similar Uploaded Photo", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            # still allow saving new reference from live
            if key == ord('s') and lm is not None:
                ts = int(time() * 1000)
                fname = REF_FOLDER / f"{SAVE_PREFIX}{ts}.jpg"
                cv2.imwrite(str(fname), frame)
                desc = normalize_landmarks(lm)
                ref_db.append({"name": fname.name, "desc": desc, "img": frame.copy()})
                print("Saved reference:", fname.name)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
import cv2
import time
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from attendance_logic import load_marked_students, mark_attendance, resolve_known_identity, update_confirmation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = BASE_DIR / "face_recognizer.yml"
LABELS_PATH = BASE_DIR / "labels.npy"
LANDMARKER_PATH = BASE_DIR / "face_landmarker.task"
ATTENDANCE_FOLDER = BASE_DIR / "attendance"
ATTENDANCE_PATH = ATTENDANCE_FOLDER / "attendance.csv"
FRAME_SIZE = (640, 480)
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
CONFIDENCE_THRESHOLD = 80.0
CONFIRM_FRAMES = 8
MIN_FACE_SIZE = 80
SINGLE_PERSON_MODE = True
FACE_LOST_SECONDS = 1.5
WINDOW_TITLE = "Face Recognition Attendance"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 220, 0)
RED = (0, 0, 230)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
BLACK = (0, 0, 0)

# ================= CHECK FILES =================
for p in [MODEL_PATH, RECOGNIZER_PATH, LABELS_PATH, LANDMARKER_PATH]:
    if not p.exists():
        print(f"ERROR: Required file not found: {p}")
        sys.exit(1)

ATTENDANCE_FOLDER.mkdir(exist_ok=True)
try:
    today = datetime.now().strftime("%Y-%m-%d")
    marked_students = load_marked_students(ATTENDANCE_PATH, today)
except (OSError, ValueError) as e:
    print(f"ERROR: Cannot read attendance file {ATTENDANCE_PATH}: {e}")
    print("Check file permissions and ensure its header is Name, Date, Time.")
    sys.exit(1)

# ================= LOAD LABELS / LBPH =================
try:
    labels_data = np.load(LABELS_PATH, allow_pickle=True).item()
    if not isinstance(labels_data, dict) or not labels_data:
        raise ValueError("label file must contain a non-empty dictionary")
    labels = {}
    for key, value in labels_data.items():
        label_id = int(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("each label must map to a non-empty string")
        labels[label_id] = value.strip()
except (OSError, ValueError, TypeError, AttributeError, EOFError) as e:
    print(f"ERROR: Could not read a valid label map from {LABELS_PATH}: {e}")
    print("Regenerate the local model files with: python train_faces.py")
    sys.exit(1)

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(RECOGNIZER_PATH))
except Exception as e:
    print("ERROR loading LBPH recognizer:", e)
    sys.exit(1)

# ================= MEDIAPIPE FACE LANDMARKER =================
try:
    base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    TESSELATION = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    CONTOURS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
except Exception as e:
    print("ERROR starting MediaPipe Face Landmarker:", e)
    sys.exit(1)

# ================= HELPERS =================
def put_text(frame, text, pos, color=WHITE, scale=0.6, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x, y), FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_box(frame, x, y, w, h, name, score, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    ty = y - 10 if y >= 35 else y + h + 25
    put_text(frame, name, (x, ty), color, 0.62)
    put_text(frame, f"Score: {score:.1f}", (x, ty + 25), color, 0.48)


def draw_mesh(frame, result):
    if not result.face_landmarks:
        return
    h, w = frame.shape[:2]
    for face in result.face_landmarks:
        for c in TESSELATION:
            a, b = face[c.start], face[c.end]
            p1 = (int(a.x * w), int(a.y * h))
            p2 = (int(b.x * w), int(b.y * h))
            if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                cv2.line(frame, p1, p2, CYAN, 1, cv2.LINE_AA)
        for c in CONTOURS:
            a, b = face[c.start], face[c.end]
            p1 = (int(a.x * w), int(a.y * h))
            p2 = (int(b.x * w), int(b.y * h))
            if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                cv2.line(frame, p1, p2, YELLOW, 1, cv2.LINE_AA)
        for lm in face:
            px, py = int(lm.x * w), int(lm.y * h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 1, YELLOW, -1, cv2.LINE_AA)

# ================= CAMERA / YUNET =================
backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
webcam = cv2.VideoCapture(0, backend)
if not webcam.isOpened():
    landmarker.close()
    print("ERROR: Could not open webcam.")
    sys.exit(1)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH), "", FRAME_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
)

# ================= ATTENDANCE STATE =================
candidate_name = None
candidate_count = 0
locked_name = None
locked_score = None
last_face_time = 0.0
previous_time = time.time()
start_time = time.monotonic()

print("\n==============================================")
print(" FACE RECOGNITION ATTENDANCE SYSTEM")
print(" YuNet + LBPH + MediaPipe Face Mesh")
print("==============================================")
print("Loaded students:")
for k, v in labels.items():
    print(f"  {k} -> {v}")
print(f"Already marked today: {sorted(marked_students)}")
print("Press ESC or Q to exit.\n")

# ================= MAIN LOOP =================
try:
    while True:
        now = datetime.now()
        display_date = now.strftime("%d-%m-%Y")
        display_time = now.strftime("%H:%M:%S")

        try:
            ret, frame = webcam.read()
        except cv2.error as e:
            print(f"ERROR: Webcam read failed: {e}")
            break
        if not ret:
            print("ERROR: Webcam returned no frame; stopping safely.")
            break
        h, w = frame.shape[:2]

        # Keep a clean copy for YuNet + LBPH.
        # Detection/recognition must happen BEFORE drawing the mesh.
        recognition_frame = frame.copy()

        # YuNet detection on the original camera image.
        detector.setInputSize((w, h))
        _, faces = detector.detect(recognition_frame)

        # MediaPipe landmarks are drawn after detection.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        landmark_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        draw_mesh(frame, landmark_result)
        face_count = 0 if faces is None else len(faces)

        # Recognition
        if face_count == 0:
            if locked_name is None:
                candidate_name = None
                candidate_count = 0
        elif SINGLE_PERSON_MODE and face_count > 1:
            if locked_name is None:
                candidate_name = None
                candidate_count = 0
        else:
            face = faces[0]
            x, y, fw, fh = map(int, face[:4])
            x = max(0, x); y = max(0, y)
            x2 = min(w, x + fw); y2 = min(h, y + fh)

            if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE and x2 > x and y2 > y:
                crop = recognition_frame[y:y2, x:x2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)
                    try:
                        label, score = recognizer.predict(gray)
                    except cv2.error:
                        label, score = -1, 999.0
                    name = resolve_known_identity(
                        label, score, labels, CONFIDENCE_THRESHOLD
                    )

                    if locked_name is not None:
                        if name == locked_name:
                            # Refresh the lock only when the current face still matches
                            # the recognized identity; a different person must not inherit it.
                            last_face_time = time.time()
                            draw_box(
                                frame, x, y, fw, fh,
                                f"{locked_name} - LOCKED", score, GREEN
                            )
                        else:
                            # Drop the old identity immediately if the face no longer
                            # matches, then require the new identity to pass confirmation.
                            locked_name = None
                            locked_score = None
                            candidate_name = None
                            candidate_count = 0
                            draw_box(
                                frame, x, y, fw, fh,
                                "IDENTITY CHANGED - VERIFY", score, YELLOW
                            )
                    elif name is not None:
                        candidate_name, candidate_count, confirmed = update_confirmation(
                            candidate_name, candidate_count, name, CONFIRM_FRAMES
                        )
                        draw_box(frame, x, y, fw, fh, f"Checking: {name}", score, YELLOW)

                        if confirmed:
                            locked_name = candidate_name
                            locked_score = score
                            last_face_time = time.time()
                            candidate_name = None
                            candidate_count = 0

                            if locked_name not in marked_students:
                                mark_date = datetime.now().strftime("%Y-%m-%d")
                                mark_time = datetime.now().strftime("%H:%M:%S")
                                try:
                                    was_written = mark_attendance(
                                        ATTENDANCE_PATH, locked_name, mark_date, mark_time
                                    )
                                except (OSError, ValueError) as e:
                                    print(f"ERROR: Could not save attendance: {e}")
                                    put_text(frame, "SAVE ERROR - CHECK FILE", (15, 150), RED, 0.55)
                                else:
                                    marked_students.add(locked_name)
                                    if was_written:
                                        print(f"Attendance marked: {locked_name} at {mark_time}")
                                    else:
                                        print(f"Already marked today: {locked_name}")
                            else:
                                print(f"Already marked today: {locked_name}")
                    else:
                        candidate_name = None
                        candidate_count = 0
                        draw_box(frame, x, y, fw, fh, "UNKNOWN", score, RED)
                else:
                    # An empty crop is not a consecutive confirmation frame.
                    if locked_name is None:
                        candidate_name = None
                        candidate_count = 0
            else:
                # A too-small or invalid face must reset confirmation.
                if locked_name is None:
                    candidate_name = None
                    candidate_count = 0

        # Unlock only after face disappears
        if SINGLE_PERSON_MODE and locked_name is not None:
            if time.time() - last_face_time > FACE_LOST_SECONDS:
                print(f"Face lost. Unlocking {locked_name}.")
                locked_name = None
                locked_score = None
                candidate_name = None
                candidate_count = 0

        # FPS
        current = time.time()
        fps = 1.0 / (current - previous_time) if current > previous_time else 0
        previous_time = current

        # Status
        if locked_name is not None:
            status = "ATTENDANCE MARKED" if locked_name in marked_students else "RECOGNIZED"
            status_color = GREEN
        elif candidate_name is not None:
            status = f"CONFIRMING {candidate_count}/{CONFIRM_FRAMES}"
            status_color = YELLOW
        elif face_count == 0:
            status = "NO FACE DETECTED"
            status_color = RED
        elif face_count > 1 and SINGLE_PERSON_MODE:
            status = "MULTIPLE FACES - WAIT"
            status_color = RED
        else:
            status = "READY - LOOK AT CAMERA"
            status_color = YELLOW

        # Information panel
        cv2.rectangle(frame, (0, 0), (w, 125), BLACK, -1)
        put_text(frame, "FACE RECOGNITION ATTENDANCE", (15, 28), CYAN, 0.68)
        put_text(frame, f"Date: {display_date}", (15, 58), WHITE, 0.52)
        put_text(frame, f"Time: {display_time}", (175, 58), WHITE, 0.52)
        put_text(frame, f"Faces: {face_count}", (335, 58), WHITE, 0.52)
        put_text(frame, f"FPS: {int(fps)}", (445, 58), WHITE, 0.52)
        put_text(frame, f"Status: {status}", (15, 90), status_color, 0.56)

        if locked_name is not None:
            put_text(frame, f"Name: {locked_name}", (15, 118), GREEN, 0.54)
        elif candidate_name is not None:
            put_text(frame, f"Candidate: {candidate_name}", (15, 118), YELLOW, 0.52)

        # Bottom information
        put_text(frame, f"Marked Today: {len(marked_students)}", (15, h - 42), GREEN, 0.55)
        put_text(frame, "ESC/Q = Exit", (w - 135, h - 42), WHITE, 0.50)
        put_text(frame, "Mesh: MediaPipe | Recognition: YuNet + LBPH", (15, h - 15), WHITE, 0.42)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key in (ord("q"), ord("Q")):
            break
finally:
    webcam.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("\nAttendance system stopped.")
    print(f"Attendance file: {ATTENDANCE_PATH}")

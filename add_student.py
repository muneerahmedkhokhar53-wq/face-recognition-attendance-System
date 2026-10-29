import cv2
import sys
from pathlib import Path
import numpy as np
from attendance_logic import is_valid_student_name

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
FACES_DIR = BASE_DIR / "faces"
FRAME_SIZE = (640, 480)
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
SAMPLES = 40
MIN_FACE_SIZE = 80

if not MODEL_PATH.exists():
    print(f"ERROR: {MODEL_PATH} not found.")
    sys.exit(1)

name = input("Enter student name or ID (letters, numbers, spaces, . _ -; max 64 chars): ").strip()
if not is_valid_student_name(name):
    print("ERROR: Invalid name/ID. Use letters, numbers, spaces, dots, underscores, or hyphens (max 64 characters).")
    print("Path separators and special characters are not allowed.")
    sys.exit(1)

student_dir = FACES_DIR / name
student_dir.mkdir(parents=True, exist_ok=True)
existing = list(student_dir.glob("*.jpg"))
start_number = len(existing)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH), "", FRAME_SIZE,
    SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
)

saved = 0
print(f"Capturing {SAMPLES} images for {name}.")
print("Keep only this student in front of camera. Move slightly left/right/up/down.")
print("Press Q or ESC to cancel.")

try:
    while saved < SAMPLES:
        ok, frame = camera.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None and len(faces) == 1:
            x, y, fw, fh = [int(v) for v in faces[0][:4]]
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)

            if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE and x2 > x and y2 > y:
                crop = frame[y:y2, x:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)

                filename = student_dir / f"{start_number + saved + 1:04d}.jpg"
                if not cv2.imwrite(str(filename), gray):
                    print(f"ERROR: Could not save enrollment image: {filename}")
                    break
                saved += 1

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                status = "FACE OK - SAVED"
            else:
                status = "MOVE CLOSER"
        elif faces is not None and len(faces) > 1:
            status = "ONLY ONE PERSON"
        else:
            status = "FACE NOT DETECTED"

        cv2.putText(frame, f"{name}  Saved: {saved}/{SAMPLES}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, status, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0) if saved else (0,0,255), 2)
        cv2.imshow("Add Student", frame)

        key = cv2.waitKey(80) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Saved {saved} images in: {student_dir}")
if saved == SAMPLES:
    print("Next: python train_faces.py")

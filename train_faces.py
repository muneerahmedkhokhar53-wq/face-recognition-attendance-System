import cv2
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
RECOGNIZER_PATH = BASE_DIR / "face_recognizer.yml"
LABELS_PATH = BASE_DIR / "labels.npy"

if not FACES_DIR.exists():
    print("ERROR: faces folder not found.")
    sys.exit(1)

student_dirs = sorted([p for p in FACES_DIR.iterdir() if p.is_dir()],
                      key=lambda p: p.name.lower())

if not student_dirs:
    print("ERROR: No student folders found.")
    sys.exit(1)

images = []
label_ids = []
labels = {}

for label_id, student_dir in enumerate(student_dirs):
    files = sorted([p for p in student_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not files:
        print(f"WARNING: {student_dir.name} has no images.")
        continue

    labels[label_id] = student_dir.name
    print(f"{label_id} -> {student_dir.name}: {len(files)} images")

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        images.append(image)
        label_ids.append(label_id)

if not images:
    print("ERROR: No usable enrollment images were found.")
    print("Check that faces/<student-name-or-id>/ contains readable JPG, JPEG, or PNG images.")
    sys.exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)

print(f"Training with {len(images)} images...")
recognizer.train(images, np.array(label_ids, dtype=np.int32))
try:
    recognizer.write(str(RECOGNIZER_PATH))
    np.save(LABELS_PATH, labels, allow_pickle=True)
except (OSError, cv2.error, ValueError) as e:
    print(f"ERROR: Could not save trained model files: {e}")
    sys.exit(1)

print("TRAINING COMPLETE")
print(f"Created: {RECOGNIZER_PATH}")
print(f"Created: {LABELS_PATH}")
print("Run: python face_reco_mesh_attendance_fixed.py")

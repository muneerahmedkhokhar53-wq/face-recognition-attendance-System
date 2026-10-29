# Local Webcam Validation Guide

Use this checklist after installing the dependencies and creating the local model files described in the [README](../README.md). It checks the full webcam-to-CSV workflow, which GitHub Actions cannot exercise because CI has no access to your camera or personal enrollment data.

## Before you start

- Use only your own face or participants who have given informed consent.
- Keep enrollment images, `face_recognizer.yml`, `labels.npy`, and attendance output on your local machine. Do not upload them to an issue, pull request, or public repository.
- Enroll at least two consented test identities with distinct labels if you want to test identity switching.
- Ensure the local model files exist and were trained from the current enrollment set.
- Close other applications that may be using the webcam.

## 1. Run automated tests first

From the repository root, with your virtual environment activated:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q attendance_logic.py face_reco_mesh_attendance_fixed.py add_student.py train_faces.py
```

Expected: all unit tests pass and the syntax check exits without an error. These checks validate helper logic and syntax; they do not measure recognition accuracy.

## 2. Start the webcam application

```powershell
python face_reco_mesh_attendance_fixed.py
```

Confirm that the camera window opens, the face mesh is drawn, and the status panel updates. Press `Q` or `Esc` to close the application.

If the program reports a missing model file, follow the setup steps in the README. Do not copy identity-specific files from old commits or other people's repositories.

## 3. Manual acceptance checklist

Run each scenario separately. Inspect only your local `attendance/attendance.csv`.

| Check | Steps | Expected result |
| --- | --- | --- |
| Known identity | Look toward the camera with a clear, well-lit face for at least the configured confirmation period. | The same known label is observed across consecutive frames; the app locks the identity and writes one row if not already marked today. |
| Unknown identity | Use a person who is not enrolled, with their consent, or test without a face in view. | The app should not assign a known student label to an unknown face and should not create an attendance row for it. |
| Identity switch | After identity A is locked, move out of view long enough for the lock to clear, then show the separately enrolled identity B. Also observe the direct switch behavior if the face stays in view. | Identity A must not be transferred to B. The app should clear the old lock when the recognized label changes and require B to pass confirmation before marking. |
| Multiple faces | With consent, let two faces appear in frame at once. | In single-person mode, the app should wait rather than select a face to mark. |
| Duplicate attendance | Recognize the same enrolled identity again on the same date, restarting the app if needed. | The CSV should contain no second row for that same name/date. The app may report that the person was already marked. |
| Face disappears | Once an identity is locked, move fully out of frame and wait more than the configured face-lost interval. | The lock should clear; a later appearance must be confirmed again. |
| Exit | Press `Q` and then repeat with `Esc`. | The window closes and the application exits cleanly. |
| Camera unavailable | Close the app, make the camera unavailable using normal operating-system controls, then start it again. Restore camera access afterwards. | The app should report that the webcam cannot be opened and exit without entering the recognition loop. |

**Important:** the identity-switch case depends on the camera and recognition model correctly distinguishing the enrolled faces. A passing manual check is not a statistical accuracy evaluation.

## 4. Inspect the CSV safely

Open `attendance/attendance.csv` locally. It should have the columns:

```csv
Name,Date,Time
```

For each name and date, there should be at most one attendance row. Do not publish screenshots or CSV excerpts containing real names, IDs, timestamps, or other personal information. If you need to share a bug report, replace these with fictional labels and dates.

## 5. Record your test results

Copy this template into a private note and fill it in after testing:

- Date/time of test:
- Operating system:
- Python version:
- OpenCV / MediaPipe versions:
- Webcam model (optional):
- Lighting and approximate camera distance:
- Automated tests: PASS / FAIL
- Known identity: PASS / FAIL / NOT TESTED
- Unknown identity: PASS / FAIL / NOT TESTED
- Identity switch: PASS / FAIL / NOT TESTED
- Multiple faces: PASS / FAIL / NOT TESTED
- Duplicate attendance: PASS / FAIL / NOT TESTED
- Face disappearance / unlock: PASS / FAIL / NOT TESTED
- Clean exit / camera unavailable: PASS / FAIL / NOT TESTED
- Notes on false matches, missed detections, or delays:

Do not claim an accuracy percentage based on a few demonstrations. A defensible evaluation needs a consented, representative test set, a documented protocol, and counts of false accepts and false rejects.

## Troubleshooting

- **Camera does not open:** check OS camera permissions, close other apps using it, and try the correct camera index if your device has multiple cameras.
- **Known person is always UNKNOWN:** check that the person was enrolled and the model files were regenerated after enrollment; inspect lighting, face size, pose, and the configured LBPH threshold.
- **Wrong or unstable labels:** collect more varied consented enrollment samples, retrain, and repeat the test. Do not lower the threshold blindly; validate false matches as well as missed matches.
- **CSV cannot be saved:** close the file in spreadsheet software, check folder permissions, and ensure the local `attendance/` directory is writable.
- **The window closes unexpectedly:** run from the terminal and save the error message, but remove personal data before sharing it publicly.

# Face Recognition Attendance System

A Python computer-vision prototype for recording attendance with a webcam. The project combines OpenCV, YuNet face detection, LBPH face recognition, and MediaPipe Face Mesh.

> **Status:** Educational prototype. Recognition accuracy depends on image quality, lighting, enrollment data, camera position, and threshold configuration. Do not use this project as the sole basis for high-stakes attendance or identity decisions.

## Features

- Webcam-based face detection and recognition pipeline
- Face enrollment using labelled samples
- Attendance records containing a student identifier and timestamp
- OpenCV-based image processing, with YuNet, LBPH, and MediaPipe components
- Designed as a local computer-vision learning project

## Technology

- Python
- OpenCV
- YuNet face detector
- LBPH face recognizer (requires an OpenCV build with the contrib face module)
- MediaPipe Face Mesh

## Prerequisites

- Python version compatible with the installed OpenCV and MediaPipe builds
- Webcam
- A Windows, macOS, or Linux environment with camera access

## Setup

Clone the repository:

```bash
git clone https://github.com/muneerahmedkhokhar53-wq/face-recognition-attendance-System.git
cd face-recognition-attendance-System
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the dependencies declared by the project, if the dependency file is available and valid:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the source scripts and model prerequisites before launching. The recognition pipeline expects locally prepared model assets and enrollment data; these are intentionally not bundled as personal or generated artifacts in the public repository. Do not assume the webcam workflow is ready until those prerequisites have been configured.

## Run

Inspect the Python entry-point file and configuration before running the application. The main webcam pipeline is:

```bash
python face_reco_mesh_attendance_fixed.py
```

Follow the source scripts for enrollment and training steps. Exact dependency and model setup should be validated in the target environment before use.

## Typical workflow

1. Obtain informed consent from each participant before collecting face images.
2. Prepare enrollment images locally and run the project's enrollment/training scripts.
3. Configure the required model assets locally.
4. Start the webcam attendance workflow.
5. Review uncertain matches and attendance records manually.
6. Keep only records that are necessary and securely delete them when no longer needed.

## Privacy and limitations

- **Never commit real student records, enrollment images, trained identity recognizers, label mappings, camera captures, or credentials.** The repository's `.gitignore` excludes common local data paths and generated identity artifacts.
- Attendance spreadsheets, CSV files, face images, recognizer files, and label maps should remain on a controlled local machine unless there is a documented, consented reason to store them elsewhere.
- Removing a file from the latest version does not erase copies from earlier Git history or existing clones. If real personal or biometric-derived data was committed, review repository history and access, and consider making the repository private and rewriting history as appropriate.
- Recognition may fail or misidentify people, particularly with poor lighting, occlusion, pose changes, or low-quality enrollment images.
- LBPH confidence values are not calibrated identity probabilities. Choose thresholds using representative validation samples and measure false-accept and false-reject rates.
- Face Mesh landmarks are not proof of identity.
- This repository should not be treated as a production security system or an unbiased biometric solution.

## Recommended validation before real use

- Test with consented, representative samples under different lighting and camera conditions.
- Record false matches, missed detections, and processing latency; do not claim accuracy without a documented test set.
- Add automated tests for attendance logging, duplicate attendance handling, and malformed or missing input data.
- Document the tested Python/OpenCV/MediaPipe versions and a clean-install procedure.

## License

Check the repository for a license file before redistributing or reusing this project.

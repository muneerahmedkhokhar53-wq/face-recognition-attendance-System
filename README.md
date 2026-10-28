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
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies from the repository's dependency file, if present. For a reproducible setup, pin and document the exact versions that have been tested in your environment rather than assuming the latest versions are compatible.

```bash
python -m pip install --upgrade pip
# Then install the project's declared dependencies, for example:
python -m pip install -r requirements.txt
```

## Run

Inspect the repository's Python entry-point file and configuration before running the application. Launch the script intended as the main entry point, for example:

```bash
python face-recognition-attendance-System.py
```

If the entry-point filename or required arguments differ in the current version, follow the source files in the repository.

## Typical workflow

1. Prepare a small set of clear enrollment images for each consenting participant.
2. Run the enrollment step implemented by the project.
3. Start the webcam attendance workflow.
4. Review the resulting records and manually correct uncertain matches.
5. Back up only the records that are needed, and securely delete data when it is no longer required.

## Privacy and limitations

- Obtain informed consent before collecting or processing face images or face-derived data.
- Keep enrollment images, face encodings, and attendance records out of public commits. Never commit real student data, camera captures, or credentials.
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

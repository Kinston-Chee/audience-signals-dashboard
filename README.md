# Audience Engagement Signals Dashboard

A beginner-friendly computer vision tutorial for building an audience engagement signals dashboard with OpenCV and MediaPipe.

The project teaches juniors how to turn webcam frames into simple, observable visual signals such as face direction, gaze direction, and eye openness. It is designed as a club workshop resource, so the notebooks build up gradually from webcam basics to a final dashboard demo.

## What This Project Covers

- OpenCV webcam capture and drawing
- MediaPipe face detection and face mesh landmarks
- Basic yaw and pitch estimation for head direction
- Simple signal classification for facing front, looking away, and eye openness
- Multi-face aggregation with privacy-by-design discussion
- Time-series smoothing for less noisy visual signals
- Dashboard overlays with OpenCV
- Demo and reliability checklist for presenting the project

## Repository Structure

```text
.
|-- notebooks/
|   |-- Audience_Signals_Dashboard.ipynb
|   |-- README.md
|   `-- tutorial_series/
|       |-- 00_Orientation_and_Setup.ipynb
|       |-- 01_OpenCV_Webcam_Basics.ipynb
|       `-- ...
|-- requirements.txt
`-- README.md
```

## Getting Started

Use Python 3.10 or newer. A webcam is required for the live OpenCV lessons.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks from the `notebooks/` folder.

For a quick overview, start with:

```text
notebooks/Audience_Signals_Dashboard.ipynb
```

For the full workshop path, follow:

```text
notebooks/tutorial_series/00_Orientation_and_Setup.ipynb
notebooks/tutorial_series/01_OpenCV_Webcam_Basics.ipynb
...
notebooks/tutorial_series/10_Final_Integration_Rebuild_The_Full_Project.ipynb
```

## Ethics Note

This project focuses on observable computer vision signals. It should not be used to infer emotions, attention, boredom, learning quality, or other internal mental states. Treat the dashboard as a technical teaching demo, not as a judgement system for people.

## For Club Juniors

Run one notebook at a time, read the notes before each code cell, and press `q` to close OpenCV webcam windows when a lesson asks you to stop the live feed.

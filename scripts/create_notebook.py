import json
import os

# Define the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

# --- Notebook Content Generation ---

# 1. Title & Intro
add_markdown("""# Project: Audience Engagement Signals Dashboard (5-Point Analysis)

**Goal:** Build a real-time computer vision system that breaks down audience behavior into 5 granular observable signals.

**Core Learning:** "Complex behavior is built from simple, observable primitives."

### \u26a0\ufe0f ETHICAL DISCLAIMER
- We detect **Face Direction** (Physical)
- We detect **Gaze Direction** (Physical)
- We detect **Eye Openness** (Physical)
- We do **NOT** infer "boredom" or "attention". We just report the metrics.
""")

add_markdown("""## 1. The 5 Observable Signals
We will track these 5 states per person:
1.  **Face Front**: Head is roughly facing the camera.
2.  **Face Away**: Head is turned left/right/up/down.
3.  **Eyes Front**: Irises are centered in the eyes.
4.  **Eyes Away**: Irises are looking to the side.
5.  **Eyes Open/Close**: Detecting blinks or sleepiness.
""")

# 2. Setup
add_markdown("""## 2. Setup
**PowerShell Install Command:**
```powershell
py -3.12 -m pip install opencv-python mediapipe numpy matplotlib scipy
```
""")

add_code("""!pip install opencv-python mediapipe numpy matplotlib scipy
""")

add_code("""import cv2
import mediapipe as mp
import numpy as np
import time
import collections
from scipy.spatial import distance as dist

print(f"OpenCV: {cv2.__version__}")
print(f"MediaPipe: {mp.__version__}")
""")

# Step 1: Basics
add_markdown("""## 3. Step 1: Eye Aspect Ratio (EAR) for Blinks
To detect "Eye Open/Close", we use the **Eye Aspect Ratio (EAR)** formula.
It measures how "open" the eye is by comparing vertical dist vs horizontal dist of landmarks.

`EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`
""")

add_code("""def calculate_EAR(eye_landmarks):
    # Vertical lines
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal line
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    if C == 0: return 0
    ear = (A + B) / (2.0 * C)
    return ear
""")

# Step 2: Gaze
add_markdown("""## 4. Step 2: Head & Iris Logic
We separate Head direction from Eye direction.
""")

add_code("""def get_5_point_stats(landmarks, frame_w, frame_h):
    # --- 1. EYE OPENNESS (EAR) FIRST ---
    # Helper to get (x,y)
    def pt(idx): return (landmarks[idx].x, landmarks[idx].y)
    
    # Left Eye: 33, 160, 158, 133, 153, 144
    left_eye_pts = [pt(33), pt(160), pt(158), pt(133), pt(153), pt(144)]
    ear = calculate_EAR(left_eye_pts)
    is_eye_open = ear > 0.20
    
    # --- 2. HEAD DIRECTION (YAW + PITCH) ---
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    chin = landmarks[152]
    forehead = landmarks[10]
    
    # Yaw (Horizontal)
    face_width = abs(right_cheek.x - left_cheek.x)
    nose_dist_x = abs(nose.x - left_cheek.x)
    yaw_ratio = 0.5
    if face_width > 0: yaw_ratio = nose_dist_x / face_width
    
    # Pitch (Vertical)
    face_height = abs(chin.y - forehead.y)
    nose_dist_y = abs(chin.y - nose.y)
    pitch_ratio = 0.5
    if face_height > 0: pitch_ratio = nose_dist_y / face_height
    
    # Logic: Face is "Front" only if BOTH Yaw and Pitch are centered
    is_yaw_front = 0.35 < yaw_ratio < 0.65
    is_pitch_front = 0.30 < pitch_ratio < 0.70 # Tune if needed
    
    is_face_front = is_yaw_front and is_pitch_front
    
    # --- 3. IRIS DIRECTION ---
    iris_ratio = 0.0 # Default to 0 if closed
    is_eye_front = False
    
    if is_eye_open:
        # Only calculate Iris if eyes are open
        L_in = landmarks[133]
        L_out = landmarks[33]
        L_iris = landmarks[468]
        
        eye_width = abs(L_in.x - L_out.x)
        iris_dist = abs(L_iris.x - L_out.x)
        
        if eye_width > 0: iris_ratio = iris_dist / eye_width
        
        is_eye_front = 0.35 < iris_ratio < 0.65
    
    return {
        "Face Front": is_face_front,
        "Face Away": not is_face_front,
        "Eyes Front": is_eye_front,
        "Eyes Away": not is_eye_front, # Will be True if closed (iris=0). This is technically correct (not looking at screen).
        "Eyes Open": is_eye_open,
        "Debug": (yaw_ratio, pitch_ratio, iris_ratio, ear)
    }
""")

# Dashboard
add_markdown("""## 5. Main Dashboard
Displaying all 5 signals.
""")

add_code("""cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

mp_face_mesh = mp.solutions.face_mesh

print("Starting 5-Signal Dashboard... 'q' to quit.")

with mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, c = frame.shape
        
        # Aggregation for this frame
        frame_counts = {
            "Face Front": 0, "Face Away": 0,
            "Eyes Front": 0, "Eyes Away": 0,
            "Eyes Open": 0, "Eyes Closed": 0
        }
        
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                stats = get_5_point_stats(face.landmark, w, h)
                
                # Update Counts
                if stats["Face Front"]: frame_counts["Face Front"] += 1
                else: frame_counts["Face Away"] += 1
                
                if stats["Eyes Front"]: frame_counts["Eyes Front"] += 1
                else: frame_counts["Eyes Away"] += 1
                
                if stats["Eyes Open"]: frame_counts["Eyes Open"] += 1
                else: frame_counts["Eyes Closed"] += 1
                
                # Draw Box
                x_min = min([l.x for l in face.landmark]) * w
                x_max = max([l.x for l in face.landmark]) * w
                y_min = min([l.y for l in face.landmark]) * h
                y_max = max([l.y for l in face.landmark]) * h
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)
                
                # Draw Text Status
                status_txt = []
                if stats["Face Front"]: status_txt.append("Face:Front")
                else: status_txt.append("Face:Away")
                
                # Only show eye status if face is front (heuristic) or show always? Show always as requested.
                if stats["Eyes Front"]: status_txt.append("Eye:Front")
                else: status_txt.append("Eye:Away")
                
                if not stats["Eyes Open"]: status_txt.append("BLINK")
                
                for i, txt in enumerate(status_txt):
                    cv2.putText(frame, txt, (int(x_min), int(y_min)-10-(i*15)), font, 0.5, (0,255,0), 1)

                # Debug Values
                yaw, pitch, iris, ear = stats["Debug"]
                cv2.putText(frame, f"Y:{yaw:.2f} P:{pitch:.2f} I:{iris:.2f} E:{ear:.2f}", (int(x_min), int(y_max)+15), font, 0.4, (200,200,200), 1)

        # Dashboard UI
        cv2.rectangle(frame, (0,0), (200, 160), (0,0,0), -1)
        
        y = 20
        cv2.putText(frame, "Real-time Signals", (10,y), font, 0.6, (255,255,255), 1); y+=20
        cv2.putText(frame, f"Face Front: {frame_counts['Face Front']}", (10,y), font, 0.5, (0,255,0), 1); y+=20
        cv2.putText(frame, f"Face Away:  {frame_counts['Face Away']}", (10,y), font, 0.5, (0,255,255), 1); y+=20
        cv2.putText(frame, f"Eyes Front: {frame_counts['Eyes Front']}", (10,y), font, 0.5, (0,255,0), 1); y+=20
        cv2.putText(frame, f"Eyes Away:  {frame_counts['Eyes Away']}", (10,y), font, 0.5, (0,255,255), 1); y+=20
        cv2.putText(frame, f"Eyes Open:  {frame_counts['Eyes Open']}", (10,y), font, 0.5, (255,255,255), 1); y+=20
        cv2.putText(frame, f"Eyes Close:  {frame_counts['Eyes Closed']}", (10,y), font, 0.5, (0,0,255), 1); y+=20
        
        cv2.imshow('5-Point Signal Dashboard', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
""")

# Export
add_markdown("""## 6. Export Note
You can log these 6 integers per frame to CSV for "Engagement Analysis" later.
""")

# Write the notebook
with open(os.path.join(r"d:\\Python\\Audience Engagement Signals Dashboard", "Audience_Signals_Dashboard.ipynb"), "w") as f:
    json.dump(notebook, f, indent=4)

print("Notebook generated successfully.")

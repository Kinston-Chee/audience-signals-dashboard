import json
import os

OUTPUT_DIR = "audience_signals_tutorial"

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(notebook, f, indent=4)
    print(f"Generated {filename}")

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.split("\n")]}

# --- Notebook 04: Geometry (Yaw/Pitch) ---
nb04 = [
    md("# 04. Geometry for Head Direction (Yaw & Pitch)\n\n**Goal:** Use math to detect if a face is looking straight or turning away."),
    md("## 1. The Strategy: Ratios (Heuristics)\nTrue 3D pose estimation is hard. We use a simpler 2D method: **Ratios**.\n- If the Nose is exactly between the cheeks, you are facing forward.\n- If the Nose is closer to the left cheek, you are looking left."),
    md("## 2. Yaw (Left/Right)\n`Yaw Ratio = (Nose - LeftCheek) / (RightCheek - LeftCheek)`\n- **0.5**: Centered.\n- **< 0.4**: Looking Left.\n- **> 0.6**: Looking Right."),
    md("## 3. Pitch (Up/Down)\n`Pitch Ratio = (Nose - Chin) / (Forehead - Chin)`\n- **0.5**: Centered.\n- **< 0.4**: Looking Down.\n- **> 0.6**: Looking Up."),
    code("import cv2\nimport mediapipe as mp\n\nmp_face_mesh = mp.solutions.face_mesh\nface_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)\ncap = cv2.VideoCapture(0)\n\nprint(\"Move your head! 'q' to quit\")\n\nwhile True:\n    ret, frame = cap.read()\n    if not ret: break\n    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n    results = face_mesh.process(rgb)\n    h, w, c = frame.shape\n    \n    if results.multi_face_landmarks:\n        for face in results.multi_face_landmarks:\n            # Keypoints\n            nose = face.landmark[1]\n            left_cheek = face.landmark[234]\n            right_cheek = face.landmark[454]\n            chin = face.landmark[152]\n            forehead = face.landmark[10]\n            \n            # Calc Yaw (Horizontal)\n            face_width = abs(right_cheek.x - left_cheek.x)\n            if face_width > 0:\n                yaw = abs(nose.x - left_cheek.x) / face_width\n            else: yaw = 0.5\n            \n            # Calc Pitch (Vertical)\n            face_height = abs(chin.y - forehead.y)\n            if face_height > 0:\n                pitch = abs(chin.y - nose.y) / face_height\n            else: pitch = 0.5\n            \n            # Draw it\n            nx, ny = int(nose.x * w), int(nose.y * h)\n            cv2.putText(frame, f\"Yaw: {yaw:.2f}\", (nx, ny-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)\n            cv2.putText(frame, f\"Pitch: {pitch:.2f}\", (nx, ny+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)\n\n    cv2.imshow('Lesson 04', frame)\n    if cv2.waitKey(1) & 0xFF == ord('q'): break\n\ncap.release()\ncv2.destroyAllWindows()")
]
create_notebook("04_Geometry_for_HeadDirection_Yaw_Pitch_Basics.ipynb", nb04)

# --- Notebook 05: Classification Logic ---
nb05 = [
    md("# 05. Signal Classification\n\n**Goal:** Turn math ratios into discrete Logic Signals (True/False)."),
    md("## 1. Defining States\nWe define 3 core states:\n1.  **Face Front**: Yaw and Pitch are both \"Centered\".\n2.  **Eyes Front**: Iris is centered.\n3.  **Eyes Open**: Not blinking."),
    md("## 2. Thresholds\n- Yaw: 0.35 to 0.65\n- Pitch: 0.30 to 0.70\n- Iris: 0.35 to 0.65\n- EAR (Eye Aspect Ratio): > 0.20 (Open)"),
    code("def get_signals(yaw, pitch, ear, iris):\n    # 1. Face Logic\n    is_face_front = (0.35 < yaw < 0.65) and (0.30 < pitch < 0.70)\n    \n    # 2. Eye Logic\n    is_eye_open = ear > 0.20\n    is_eye_front = False\n    if is_eye_open:\n        is_eye_front = (0.35 < iris < 0.65)\n    \n    return {\n        \"Face Front\": is_face_front,\n        \"Eyes Front\": is_eye_front,\n        \"Eyes Open\": is_eye_open\n    }"),
    md("### \ud83d\udee0\ufe0f Engineer Thinking\nWhy separate Face and Eyes? Because you can face the screen but look away (Side-eye)!"),
    code("# Run this to test the logic with random numbers\nprint(get_signals(0.5, 0.5, 0.3, 0.5)) # Should be All True\nprint(get_signals(0.9, 0.5, 0.3, 0.5)) # Face Front False (Yaw too high)\nprint(get_signals(0.5, 0.5, 0.1, 0.5)) # Eyes Open False (Blinking)")
]
create_notebook("05_Signal_Classification_Facing_Down_Away.ipynb", nb05)

# --- Notebook 06: Aggregation ---
nb06 = [
    md("# 06. Multi-Face Aggregation\n\n**Goal:** Handle a crowd. Count the percentage of engagement."),
    md("## 1. Privacy By Design\nWe iterate through ALL faces in a frame, but we only save the **Counts**. We throw away the specific face data immediately."),
    md("## 2. The Logic\n```python\ncounts = {\"Face Front\": 0, ...}\nfor face in faces:\n    signals = get_signals(face)\n    if signals[\"Face Front\"]: counts[\"Face Front\"] += 1\n```"),
    code("import cv2\nimport mediapipe as mp\n# ... (Code to loop through results.multi_face_landmarks and summarize totals) ...\n# [Simplified for this tutorial]\nprint(\"Code Concept: Loop through 'multi_face_landmarks' list.\")")
]
create_notebook("06_MultiFace_Aggregation_and_PrivacyByDesign.ipynb", nb06)

# --- Notebook 07: Smoothing ---
nb07 = [
    md("# 07. Time Series Smoothing\n\n**Goal:** Fix the \"Flickering\" problem."),
    md("## 1. The Problem\nIn one frame, yaw might be 0.64 (Front). Next frame: 0.66 (Away). The signal toggles rapidly. This is annoying."),
    md("## 2. The Solution: Moving Average\nWe keep a history of the last 30 frames (approx 1 second). We take the average.\n- If 29 frames say \"Front\" and 1 says \"Away\", the average is still \"Front\"."),
    code("import collections\n\n# Create a buffer of fixed length\nhistory = collections.deque(maxlen=10)\n\n# Simulate Data\nimport random\nfor i in range(20):\n    val = 1 if i < 15 else 0 # Sudden drop\n    history.append(val)\n    \n    # Calculate Smooth Average\n    avg = sum(history) / len(history)\n    print(f\"Raw: {val} | Smooth: {avg:.2f}\")"),
    md("### \ud83d\udee0\ufe0f Engineer Thinking\nLarger window = Smoother data, but **Higher Latency**. 30 frames is a good sweet spot.")
]
create_notebook("07_TimeSeries_Logging_and_Smoothing.ipynb", nb07)

print("Batch 2 (04-07) Generated Successfully.")

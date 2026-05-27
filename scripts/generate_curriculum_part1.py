import json
import os

OUTPUT_DIR = "audience_signals_tutorial"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

# --- Notebook 00: Orientation ---
nb00 = [
    md("# 00. Orientation & Setup\n\n**Goal:** Get your environment ready and understand the *Ethical* goals of this project."),
    md("## 1. What We Are Building\nWe are building a **Real-time Audience Engagement Settings Dashboard** using your webcam.\n\n### \u26a0\ufe0f The Golden Rule (Ethics)\nWe use Computer Vision to detect **Physical Signals** (Head direction, Eye openness), NOT internal states (Boredom, Attention).\n\n- **OK:** \"The head is turned 40 degrees left.\"\n- **NOT OK:** \"The user is not paying attention.\"\n\nWe aggregate data (e.g., \"30% of the audience is looking away\") to protect individual privacy."),
    md("## 2. Tools We Use\n1.  **OpenCV**: For reading the webcam and drawing boxes (The \"Eyes\").\n2.  **MediaPipe**: For detecting Face Landmarks (The \"Brain\").\n3.  **NumPy**: For high-speed math (The \"Calculator\")."),
    md("## 3. Installation\nRun the command below in your terminal (PowerShell) to install dependencies."),
    md("```powershell\npy -3.12 -m pip install opencv-python mediapipe numpy matplotlib scipy\n```"),
    code("!pip install opencv-python mediapipe numpy matplotlib scipy"),
    md("## 4. One-Click Check\nRun the cell below. If it prints version numbers without error, you are ready!"),
    code("import cv2\nimport mediapipe as mp\nimport numpy as np\nimport scipy\n\nprint(f\"OpenCV Version: {cv2.__version__}\")\nprint(f\"MediaPipe Version: {mp.__version__}\")\nprint(f\"NumPy Version: {np.__version__}\")\nprint(\"\\u2705 Setup Complete!\")"),
    md("### \ud83d\udee0\ufe0f Debugging\n- **ImportError / ModuleNotFound**: You didn't install the library or are using the wrong python kernel.\n- **DLL load failed**: Missing C++ redistributables on Windows. Try installing `msvc-runtime`.")
]
create_notebook("00_Orientation_and_Setup.ipynb", nb00)

# --- Notebook 01: OpenCV Basics ---
nb01 = [
    md("# 01. OpenCV Webcam Basics\n\n**Goal:** Create a \"Hello World\" camera app that runs in a loop."),
    md("## 1. The Computer Vision Loop\nVideos are just fast sequences of images (Frames). We need a `while` loop that:\n1.  **Reads** a frame from the camera.\n2.  **Processes** it (draws stuff).\n3.  **Displays** it.\n4.  **Waits** for a quit key."),
    md("## 2. Opening the Camera"),
    code("import cv2\n\n# '0' is usually the default webcam\ncap = cv2.VideoCapture(0)\n\nif not cap.isOpened():\n    print(\"Error: Could not open camera.\")\nelse:\n    print(\"Camera detected!\")\n    cap.release()"),
    md("## 3. The Main Loop (Run this!)"),
    code("import cv2\nimport time\n\ncap = cv2.VideoCapture(0)\n\nprint(\"Press 'q' to quit the window.\")\n\nwhile True:\n    # 1. Read\n    ret, frame = cap.read()\n    if not ret:\n        print(\"Failed to grab frame\")\n        break\n        \n    # 2. Process (Draw a rectangle)\n    # cv2.rectangle(img, start_point, end_point, color_bgr, thickness)\n    cv2.rectangle(frame, (50, 50), (250, 250), (255, 0, 0), 2)\n    \n    # Add Text\n    cv2.putText(frame, \"Hello OpenCV\", (50, 40), \n                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)\n    \n    # 3. Display\n    cv2.imshow('Lesson 01', frame)\n    \n    # 4. Wait (1ms) and check for 'q' key\n    # 0xFF is a hex mask to get the last 8 bits\n    if cv2.waitKey(1) & 0xFF == ord('q'):\n        break\n\ncap.release()\ncv2.destroyAllWindows()\nprint(\"Closed.\")"),
    md("### \ud83d\udee0\ufe0f Engineer Thinking\n- **Latency**: Everything inside the `while` loop adds delay. Keep it fast.\n- **BGR Colors**: OpenCV uses Blue-Green-Red, not RGB. `(255, 0, 0)` is Blue, not Red.")
]
create_notebook("01_OpenCV_Webcam_Basics.ipynb", nb01)

# --- Notebook 02: MediaPipe Basics ---
nb02 = [
    md("# 02. MediaPipe Face Detection\n\n**Goal:** Detect faces in real-time and draw bounding boxes."),
    md("## 1. What is MediaPipe?\nIt's a library by Google that gives us pre-trained AI models. We don't train AI; we **use** it."),
    md("## 2. Face Mesh vs Face Detection\n- **Face Detection**: Fast, just gives a box.\n- **Face Mesh**: Slower (slightly), gives 468 points (Mouth, Eyes, Nose).\n\nWe will use **Face Mesh** because strict Gaze Tracking requires knowing exactly where the eyes are."),
    code("import cv2\nimport mediapipe as mp\n\n# Initialize the Mesh model\nmp_face_mesh = mp.solutions.face_mesh\n\n# Setup the model with options\nface_mesh = mp_face_mesh.FaceMesh(\n    max_num_faces=3,          # Track up to 3 people\n    refine_landmarks=True,    # CRITICAL: Gives us Iris landmarks\n    min_detection_confidence=0.5,\n    min_tracking_confidence=0.5\n)"),
    md("## 3. The Detection Loop"),
    code("cap = cv2.VideoCapture(0)\n\nwhile True:\n    ret, frame = cap.read()\n    if not ret: break\n    \n    # MediaPipe needs RGB, OpenCV gives BGR\n    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n    \n    # INFERENCE (The magic happens here)\n    results = face_mesh.process(rgb_frame)\n    \n    # If faces found...\n    if results.multi_face_landmarks:\n        count = len(results.multi_face_landmarks)\n        cv2.putText(frame, f\"Faces: {count}\", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)\n        \n        for landmarks in results.multi_face_landmarks:\n             # Draw a simple dot on the 'Nose' (Index 1)\n             # Landmarks are Normalized (0.0 to 1.0). Convert to pixels.\n             h, w, c = frame.shape\n             nose = landmarks.landmark[1]\n             cx, cy = int(nose.x * w), int(nose.y * h)\n             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)\n\n    cv2.imshow('Lesson 02', frame)\n    if cv2.waitKey(1) & 0xFF == ord('q'):\n        break\n\ncap.release()\ncv2.destroyAllWindows()"),
    md("### \ud83d\udee0\ufe0f Debugging\n- **Performance**: Note the FPS. MediaPipe is heavy. Ensure you have light!\n- **Faces not found**: Detection requires the whole face to be visible.")
]
create_notebook("02_MediaPipe_Face_Detection_Basics.ipynb", nb02)

# --- Notebook 03: Landmarks & Coordinates ---
nb03 = [
    md("# 03. Landmarks & Coordinates\n\n**Goal:** Understand the 468-point Face Mesh map."),
    md("## 1. Normalized Coordinates\nMediaPipe returns `x, y, z` values between **0.0 and 1.0**.\n- `x=0`: Left edge.\n- `x=0.5`: Center.\n- `x=1.0`: Right edge.\n\nTo draw them, we multiply by image width/height."),
    md("## 2. Key Landmarks map\nWe don't need all 468 points. We mostly care about:\n- **Nose Tip**: Index `1`\n- **Chin**: Index `152`\n- **Left Eye Outer**: Index `33`\n- **Right Eye Outer**: Index `263`\n- **Left Iris Center**: Index `468`\n- **Right Iris Center**: Index `473`"),
    code("import cv2\nimport mediapipe as mp\n\n# Setup\nmp_face_mesh = mp.solutions.face_mesh\nface_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)\ncap = cv2.VideoCapture(0)\n\nprint(\"Showing Key Landmarks... 'q' to quit\")\n\nwhile True:\n    ret, frame = cap.read()\n    if not ret: break\n    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n    results = face_mesh.process(rgb)\n    h, w, c = frame.shape\n    \n    if results.multi_face_landmarks:\n        for face in results.multi_face_landmarks:\n            \n            # Helper to draw a point\n            def draw_pt(idx, color, label):\n                pt = face.landmark[idx]\n                px, py = int(pt.x * w), int(pt.y * h)\n                cv2.circle(frame, (px, py), 4, color, -1)\n                cv2.putText(frame, label, (px+5, py), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)\n            \n            # Draw main components\n            draw_pt(1, (0, 255, 0), \"Nose\")\n            draw_pt(152, (0, 255, 0), \"Chin\")\n            draw_pt(33, (255, 0, 0), \"L_Eye\")\n            draw_pt(263, (255, 0, 0), \"R_Eye\")\n            draw_pt(468, (0, 255, 255), \"L_Iris\")\n\n    cv2.imshow('Lesson 03', frame)\n    if cv2.waitKey(1) & 0xFF == ord('q'): break\n\ncap.release()\ncv2.destroyAllWindows()")
]
create_notebook("03_MediaPipe_FaceMesh_Landmarks_and_Coordinates.ipynb", nb03)

print("Batch 1 (00-03) Generated Successfully.")

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

# --- Notebook 08: UI Design ---
nb08 = [
    md("# 08. The Dashboard UI\n\n**Goal:** Create a professional overlay."),
    md("## 1. Transparency (Alpha Blending)\nWe draw a semi-transparent black box behind the text to make it readable on any background."),
    code("import cv2\nimport numpy as np\n\ndef draw_panel(image, x, y, w, h, alpha=0.6):\n    overlay = image.copy()\n    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,0,0), -1)\n    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)"),
    md("## 2. Drawing Bars\nA \"Health Bar\" for attention."),
    code("def draw_bar(image, x, y, pct, color):\n    width = int(pct * 2) # Scale 100% to 200px\n    cv2.rectangle(image, (x, y), (x+width, y+10), color, -1)\n    cv2.rectangle(image, (x, y), (x+200, y+10), (100,100,100), 1)\n\n# Try it on a dummy image\nimg = np.zeros((300, 400, 3), dtype=np.uint8)\nimg = draw_panel(img, 10, 10, 250, 150)\ndraw_bar(img, 20, 50, 80, (0,255,0)) # 80%\ncv2.putText(img, \"Attention: 80%\", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)")
]
create_notebook("08_OpenCV_Dashboard_Overlay_Bars_and_Text.ipynb", nb08)

# --- Notebook 09: Hands Heuristic (Optional) ---
nb09 = [
    md("# 09. (Optional) Phone Posture Heuristic\n\n**Goal:** Detect \"Looking Down\" + \"Hands Up\" = Likely Phone Use."),
    md("## 1. Multi-Graph Strategy\nRunning Face Mesh + Hand Tracking doubles CPU usage. We might skip frames."),
    md("## 2. The Logic\n`Is_Phone_Use = (Pitch < 0.4) AND (Hand_Y > Chin_Y)`."),
    code("import mediapipe as mp\nmp_hands = mp.solutions.hands\nhands = mp_hands.Hands(max_num_hands=2)\n# ... (Code to process hands.process(rgb)) ... \nprint(\"Concept: Combining two graphs.\")")
]
create_notebook("09_Optional_Hands_Module_for_PhonePosture_Heuristic.ipynb", nb09)

# --- Notebook 10: Final Integration ---
nb10 = [
    md("# 10. Final Project: Audience Engagement Dashboard\n\n**Goal:** Put it all together."),
    md("## 1. The Architecture\n1.  **Capture**\n2.  **Detect** (Face + Iris)\n3.  **Signal Logic** (Yaw/Pitch/EAR)\n4.  **Aggregation** (Count totals)\n5.  **Smoothing** (Buffer)\n6.  **Display** (UI Overlay)\n7.  **Log** (CSV)"),
    md("## 2. Run the Full App"),
    code("# Run the accompanying 'create_notebook.py' script (or the project file we built together)\n# to generate the final .ipynb file.")
]
create_notebook("10_Final_Integration_Rebuild_The_Full_Project.ipynb", nb10)

# --- Notebook 11: Demo Script ---
nb11 = [
    md("# 11. AGM Demo Script & Reliability\n\n**Goal:** Presenting your work to stakeholders."),
    md("## 1. The Script (1 Minute)\n> \"Good morning. This tool measures aggregate audience engagement signals.\"\n> \"It uses computer vision to detect head direction and eye openness locally on the device.\"\n> \"It does not identify individuals.\"\n> \"Here you see the metrics: 80% of the audience is facing the screen.\""),
    md("## 2. Reliability Checklist\n- [ ] Clean Camera Lens.\n- [ ] Good Lighting (Face detected?).\n- [ ] Close other Camera Apps (Teams/Zoom).\n- [ ] Restart Kernel before demo starts.")
]
create_notebook("11_AGM_Demo_Script_and_Reliability_Checklist.ipynb", nb11)

print("Batch 3 (08-11) Generated Successfully.")

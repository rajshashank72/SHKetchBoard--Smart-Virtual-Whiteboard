# main.py
import cv2
import numpy as np
from flask import Flask, Response
from hand_tracking import HandTracker
from gesture_features import get_finger_states, detect_gesture
import math
import random

app = Flask(__name__)

# Initialize once to avoid recreation
tracker = HandTracker()
color_palette = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), 
    (255, 0, 255), (0, 255, 255)
]

def generate_frames():
    cap = cv2.VideoCapture(0)  # For EC2: Use virtual cam or test video
    canvas = None
    draw_color = (0, 255, 255)
    thickness = 15
    prev_x, prev_y = None, None

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Image enhancement
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=10)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 40)
        frame = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        if canvas is None:
            canvas = np.zeros_like(frame)

        processed_frame = tracker.process_frame(frame)
        landmarks = tracker.results.multi_hand_landmarks

        if landmarks:
            for hand_landmarks in landmarks:
                h, w, _ = frame.shape
                finger_states = get_finger_states(hand_landmarks.landmark, h, w)
                gesture = detect_gesture(finger_states)

                if gesture == 'fist':
                    tip = hand_landmarks.landmark[8]
                    tip_pos = (int(tip.x * w), int(tip.y * h))
                    cv2.circle(canvas, tip_pos, 40, (0, 0, 0), -1)
                    prev_x, prev_y = None, None
                elif gesture == 'two_fingers':
                    draw_color = random.choice(color_palette)

                # Drawing logic
                mcp = hand_landmarks.landmark[6]
                pip = hand_landmarks.landmark[7]
                tip = hand_landmarks.landmark[8]
                angle = calculate_angle(
                    (int(mcp.x * w), int(mcp.y * h)),
                    (int(pip.x * w), int(pip.y * h)),
                    (int(tip.x * w), int(tip.y * h))
                )

                if angle > 145 and not gesture:
                    if prev_x is None:
                        prev_x, prev_y = tip_pos
                    cv2.line(canvas, (prev_x, prev_y), tip_pos, 
                            draw_color, thickness, cv2.LINE_AA)
                    prev_x, prev_y = tip_pos
                else:
                    prev_x, prev_y = None, None

        combined = cv2.addWeighted(processed_frame, 1, canvas, 2, 0)
        ret, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <head><title>Smart Whiteboard</title></head>
      <body>
        <h1>SHKetchBoard - Virtual Whiteboard</h1>
        <img src="/video_feed" width="800">
      </body>
    </html>
    """

def calculate_angle(a, b, c):
    """Returns angle (in degrees) between three points."""
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

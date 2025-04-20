# %%
import cv2
import numpy as np
from hand_tracking import HandTracker
from gesture_features import get_finger_states, detect_gesture
import math
import random

color_palette = [
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (255, 0, 255),   # Purple
    (0, 255, 255)    # Yellow
]


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_angle(a, b, c):
    """Returns angle (in degrees) between three points."""
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)



tracker = HandTracker()
canvas = None

draw_color = (0, 255, 255)
thickness = 15
prev_x, prev_y = None, None

while True:
    success, frame = cap.read()
    # 💡 Enhance color & brightness
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=10)  # alpha=contrast, beta=brightness

    # Optional: Boost saturation (convert to HSV and back)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 40)  # Increase saturation
    enhanced_hsv = cv2.merge([h, s, v])
    frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    
    if not success:
        break
      
    if canvas is None:
        canvas = np.zeros_like(frame)

    processed_frame = tracker.process_frame(frame)
    landmarks = tracker.results.multi_hand_landmarks
    gesture = None

    if landmarks:
        for hand_landmarks in landmarks:
            h, w, _ = frame.shape

            # 👉 Gesture detection
            finger_states = get_finger_states(hand_landmarks.landmark, h, w)
            gesture = detect_gesture(finger_states)

            if gesture == 'fist':
                # ✊ Erase using black circle at index fingertip
                tip = hand_landmarks.landmark[8]
                tip_pos = (int(tip.x * w), int(tip.y * h))
                cv2.circle(canvas, tip_pos, 40, (0, 0, 0), -1)
                prev_x, prev_y = None, None

            elif gesture == 'two_fingers':
                # ✌️ Change draw color
                draw_color = random.choice(color_palette)
                    # At the top of your script (near imports)
  

            # 👉 Drawing logic based on index finger angle
            mcp = hand_landmarks.landmark[6]
            pip = hand_landmarks.landmark[7]
            tip = hand_landmarks.landmark[8]

            mcp_pos = (int(mcp.x * w), int(mcp.y * h))
            pip_pos = (int(pip.x * w), int(pip.y * h))
            tip_pos = (int(tip.x * w), int(tip.y * h))

            angle = calculate_angle(mcp_pos, pip_pos, tip_pos)

            if angle > 145 and gesture is None:
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = tip_pos
                cv2.line(canvas, (prev_x, prev_y), tip_pos, draw_color, thickness, cv2.LINE_AA)
                prev_x, prev_y = tip_pos
            else:
                prev_x, prev_y = None, None

    combined = cv2.addWeighted(processed_frame, 1, canvas, 2, 0)
    cv2.imshow("Smart Virtual Whiteboard", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# %%


# %%




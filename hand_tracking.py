import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7):
        # Initialize MediaPipe's Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands, 
            min_detection_confidence=detection_confidence
        )

        # Utility for drawing landmarks
        self.mp_draw = mp.solutions.drawing_utils

        # To store the processed results for each frame
        self.results = None

    def process_frame(self, frame, draw_landmarks=True, draw_circle=True, show_coordinates=True):
        # Flip the frame horizontally for natural interaction (like a mirror)
        frame = cv2.flip(frame, 1)

        # Convert BGR frame to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands and landmarks
        self.results = self.hands.process(rgb_frame)

        # If any hand is detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                
                # Draw hand skeleton if enabled
                if draw_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index fingertip (Landmark ID = 8)
                h, w, _ = frame.shape
                index_finger = hand_landmarks.landmark[8]
                cx, cy = int(index_finger.x * w), int(index_finger.y * h)

                # Draw a circle on the index fingertip if enabled
                if draw_circle:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

                # Show coordinates near the fingertip if enabled
                if show_coordinates:
                    cv2.putText(frame, f'{cx}, {cy}', (cx + 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Return the processed frame
        return frame

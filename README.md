# 🖐️ Smart Virtual Whiteboard

The **Smart Virtual Whiteboard** is a gesture-controlled drawing application that uses **OpenCV** and **MediaPipe** to detect hand landmarks in real time. Users can draw, erase, clear the canvas, and switch colors using simple hand gestures — no physical tools required!

## ✨ Features

- 🖊️ Draw using your index finger
- ✌️ Change brush color with two-finger gesture
- ✋ Clear the entire canvas with an open palm
- ✊ Erase using a closed fist
- Real-time hand tracking with **MediaPipe**
- Smooth and responsive drawing experience

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**

## 📸 Gesture Controls

| Gesture | Action             |
|--------|--------------------|
| ☝️ One finger | Draw mode           |
| ✌️ Two fingers | Change color        |
| ✋ Open palm | Clear canvas       |
| ✊ Fist | Erase mode         |

## 📁 Project Structure

main.py
gesture_features
hand_tracking


## 🚀 Getting Started

### 1. Clone the Repository

##2. Install required Libraries
pip install opencv-python mediapipe numpy

##3. Run the Code


🧠 How It Works
Detects 21 hand landmarks using MediaPipe.

Uses index finger tip (landmark ID 8) to track motion.

Identifies hand gestures based on finger states (up/down).

Performs drawing, erasing, clearing, and color switching accordingly.

All drawing is done on a separate canvas layered over the live webcam feed.

🧰 Future Improvements
UI buttons to select brush size and colors

Export/save drawing as image

Add voice commands or audio feedback

Multi-hand drawing (collaborative mode)

🙌 Contributing
Pull requests and feedback are welcome. If you have ideas or suggestions, feel free to open an issue or fork the project!

👤 Author
Shashank Soni
www.linkedin.com/in/shashanksoni72 • GitHub
Drop a heart if you like it!


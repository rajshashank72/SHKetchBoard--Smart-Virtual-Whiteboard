import math

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calculate_angle(a, b, c):
    """
    Returns angle (in degrees) between three points.
    Angle is calculated at point 'b' using vectors ba and bc.
    """
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    cosine_angle = (
        (ba[0] * bc[0] + ba[1] * bc[1]) /
        (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6)
    )
    return math.degrees(math.acos(max(-1, min(1, cosine_angle))))


def get_finger_states(landmarks, h, w):
    """
    Returns a list of 5 boolean values representing if each finger is open.
    Order: [Thumb, Index, Middle, Ring, Pinky]

    Adds a margin to reduce false positives due to slight bends.
    """

    finger_states = []

    # Thumb: uses horizontal (x-axis) comparison for right hand
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    finger_states.append(thumb_tip.x > thumb_mcp.x)

    # Fingers: use y-axis positions with buffer margin to improve reliability
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    mcps = [5, 9, 13, 17]

    margin = 20  # pixels - tweakable

    for tip_id, pip_id, mcp_id in zip(tips, pips, mcps):
        tip_y = landmarks[tip_id].y * h
        pip_y = landmarks[pip_id].y * h
        mcp_y = landmarks[mcp_id].y * h

        # Finger is "open" only if tip is clearly above both pip and mcp
        is_open = tip_y < pip_y - margin and tip_y < mcp_y - margin
        finger_states.append(is_open)

    return finger_states


def detect_gesture(finger_states):
    """
    Detects gesture from finger states.
    Returns: 'fist', 'palm', 'two_fingers', or None
    """
    thumb, index, middle, ring, pinky = finger_states

    if all(not state for state in finger_states):
        return 'fist'
    elif all(state for state in finger_states):
        return 'palm'
    elif finger_states[1] and finger_states[2] and not any(finger_states[3:]):
        return 'two_fingers'
    else:
        return None

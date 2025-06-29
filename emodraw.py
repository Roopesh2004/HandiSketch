import cv2
import mediapipe as mp
import numpy as np
import math
import os
from datetime import datetime

# Ask for label name to save drawings in dataset/label folder
label = input("Enter shape label: ").strip().lower()
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

# Mediapipe hand tracking initialization
mp_hands = mp.solutions.hands  # hand tracking module from mediapipe
hands = mp_hands.Hands(min_detection_confidence=0.7)  # detect hands with good accuracy
mp_drawing = mp.solutions.drawing_utils  # for drawing hand landmarks

# Canvas setup (800x600) with 3 channels (RGB), all black (zeros)
canvas_w, canvas_h = 800, 600
canvas = np.zeros((canvas_w, canvas_h, 3), dtype=np.uint8)

# Open webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(3, canvas_w)  # width
cap.set(4, canvas_h)  # height

# To store previous finger position (so we can draw a continuous line)
prev_x, prev_y = 0, 0

# Eraser mode (False by default)
eraser_mode = False

# Infinite loop till webcam runs
while cap.isOpened():
    ret, frame = cap.read()  # grab one frame from webcam
    frame = cv2.flip(frame, 1)  # flip it horizontally like mirror
    frame = cv2.resize(frame, (canvas.shape[1], canvas.shape[0]))
# Convert frame to RGB (required by mediapipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB for mediapipe
    result = hands.process(rgb)  # get hand landmarks if hand is in frame

    # If a hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of index finger tip and thumb tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert from ratio to pixel values
            h, w, _ = frame.shape
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate distance between index finger tip and thumb tip
            distance = math.hypot(tx - ix, ty - iy)

            # If fingers are close (pinch gesture), then draw
            if distance < 40:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                if eraser_mode:
                    # Eraser: draw a black line (erase) with thick size
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), 40)
                else:
                    # Drawing: red colored line (dark red)
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 200), 5)

                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = 0, 0

    # Combine camera and canvas using transparency
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(combined, f"Eraser: {'ON' if eraser_mode else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if eraser_mode else (0, 255, 0), 2)

    cv2.imshow("EmoDraw", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break  # Quit program
    elif key == ord('e'):
        eraser_mode = not eraser_mode  # Toggle eraser ON/OFF
    elif key == ord('s'):
        # Save current canvas with timestamp as filename
        filename = datetime.now().strftime("%Y%m%d_%H%M%S.png")
        cv2.imwrite(os.path.join(save_dir, filename), canvas)
        print(f"Saved to {os.path.join(save_dir, filename)}")

# Release webcam and destroy window after quit
cap.release()
cv2.destroyAllWindows()

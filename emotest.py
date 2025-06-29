import cv2
import mediapipe as mp
import numpy as np
import math
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("bestemo.keras")

# Labels
class_labels = ['bird', 'butterfly', 'flower', 'house', 'mountain', 'sky', 'star', 'sun', 'tree']

# Emoji image folder
emoji_path = "images"  # <- Make sure to place images like emojis/sun.jpg or sun.png here

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Canvas setup
canvas_h, canvas_w = 480, 640
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, canvas_w)
cap.set(4, canvas_h)

# States
prev_x, prev_y = 0, 0
eraser_mode = False
predicted_label = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (canvas_w, canvas_h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            h, w, _ = frame.shape
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            distance = math.hypot(tx - ix, ty - iy)

            if distance < 40:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy
                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), 40)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 255), 5)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = 0, 0

    # Combine canvas and live video
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display prediction if available
    if predicted_label is not None:
        cv2.putText(combined, f"Prediction: {predicted_label}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Try to load emoji image with supported extensions
        emoji_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = os.path.join(emoji_path, f"{predicted_label}{ext}")
            if os.path.exists(test_path):
                emoji_file = test_path
                break

        if emoji_file:
            emoji = cv2.imread(emoji_file, cv2.IMREAD_UNCHANGED)
            if emoji is not None:
                emoji = cv2.resize(emoji, (150, 150))
                x_offset, y_offset = 50, 50
                h, w = emoji.shape[:2]
                if emoji.shape[2] == 4:
                    # Has alpha channel
                    alpha = emoji[:, :, 3] / 255.0
                    for c in range(3):
                        combined[y_offset:y_offset + h, x_offset:x_offset + w, c] = (
                            alpha * emoji[:, :, c] +
                            (1 - alpha) * combined[y_offset:y_offset + h, x_offset:x_offset + w, c]
                        )
                else:
                    combined[y_offset:y_offset + h, x_offset:x_offset + w] = emoji
        else:
            print(f"No emoji image found for label: {predicted_label}")

    # Display eraser status
    cv2.putText(combined, f"Eraser: {'ON' if eraser_mode else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if eraser_mode else (0, 255, 0), 2)

    # Show final image
    cv2.imshow("EmoDraw", combined)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('e'):
        eraser_mode = not eraser_mode
    elif key == ord('s'):
        save_path = f"draw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(save_path, canvas)
        print(f"Drawing saved at {save_path}")
    elif key == ord('p'):
        # Predict
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        img_array = resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]
        print("Predicted:", predicted_label)

# Cleanup
cap.release()
cv2.destroyAllWindows()

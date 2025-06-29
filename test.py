from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("emodraw.h5")
class_labels = ['bird', 'butterfly', 'flower', 'house', 'mountain', 'sky', 'star', 'sun', 'tree']

img = cv2.imread("dataset/bird/bird7.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
img_array = img.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=(0, -1))

prediction = model.predict(img_array)
print("Predicted:", class_labels[np.argmax(prediction)])

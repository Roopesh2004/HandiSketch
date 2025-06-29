import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
TARGET_COUNT = 40  # Desired number of images per label
DATASET_PATH = 'dataset'

# Augmentation techniques
augmenter = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest'
)

for label in os.listdir(DATASET_PATH):
    label_dir = os.path.join(DATASET_PATH, label)
    images = [img for img in os.listdir(label_dir) if img.endswith(('.png', '.jpg'))]

    current_count = len(images)
    print(f"📂 {label}: {current_count} images")

    if current_count >= TARGET_COUNT:
        print(f"✅ {label} already has {current_count} images.\n")
        continue

    extra_needed = TARGET_COUNT - current_count
    print(f"🔁 Generating {extra_needed} new images for '{label}'...\n")

    # Loop through existing images and generate new ones
    img_index = 0
    while extra_needed > 0:
        img_path = os.path.join(label_dir, images[img_index % len(images)])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))  # Resize to match model input

        # Expand dimensions to match ImageDataGenerator input
        img = np.expand_dims(img, axis=0)

        # Generate one new image
        for batch in augmenter.flow(img, batch_size=1, save_to_dir=label_dir, save_prefix='aug', save_format='png'):
            extra_needed -= 1
            break

        img_index += 1

print("\n🎉 Augmentation complete! All labels now have at least 40 images.")

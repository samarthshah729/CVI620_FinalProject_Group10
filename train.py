import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dense, Flatten, Dropout, Cropping2D

# ----------- LOAD DATASET -------------
base_path = "data"
csv_path = os.path.join(base_path, "driving_log.csv")
img_folder = os.path.join(base_path, "IMG")

samples = []

if not os.path.exists(csv_path):
    print("ERROR: driving_log.csv not found at", csv_path)
    exit()

print("✓ Dataset found:", os.path.abspath(base_path))

with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  
    for line in reader:
        samples.append(line)

print("Total samples loaded:", len(samples))


# ----------- IMAGE PREPROCESSING -------------
def preprocess(img):
    img = img[60:135, :, :]  # crop sky + hood
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


# ----------- GENERATOR  -------------
def generator(samples, batch_size=32):
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch = samples[offset:offset + batch_size]

            images = []
            angles = []

            for line in batch:
                img_name = line[0].split('/')[-1]
                img_path = os.path.join(img_folder, img_name)

                if not os.path.exists(img_path):
                    continue

                image = cv2.imread(img_path)
                angle = float(line[3])

                if image is None:
                    continue

                image = preprocess(image)

                images.append(image)
                angles.append(angle)

                # Augmentation: flip
                images.append(np.fliplr(image))
                angles.append(-angle)

            X = np.array(images, dtype=np.float32)
            y = np.array(angles, dtype=np.float32)

            yield (X, y)  


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# ----------- MODEL (NVIDIA ARCHITECTURE) -------------
model = Sequential([
    Cropping2D(cropping=((0, 0), (0, 0)), input_shape=(66, 200, 3)),
    Lambda(lambda x: x),

    Conv2D(24, (5,5), strides=(2,2), activation="relu"),
    Conv2D(36, (5,5), strides=(2,2), activation="relu"),
    Conv2D(48, (5,5), strides=(2,2), activation="relu"),
    Conv2D(64, (3,3), activation="relu"),
    Conv2D(64, (3,3), activation="relu"),

    Flatten(),
    Dense(100),
    Dropout(0.3),
    Dense(50),
    Dense(10),
    Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(1e-4))

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_samples)//32,
    validation_data=validation_generator,
    validation_steps=len(validation_samples)//32,
    epochs=25,
    verbose=1
)

model.save("model.h5")
print("✔ Training complete")
print("✔ Saved as model.h5")

# ----------- PLOT LOSS ------------
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.savefig("loss_plot.png")
plt.show()

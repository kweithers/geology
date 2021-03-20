import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

#Read in our data

batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.2,
  subset="training",
  seed=123,
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=batch_size)

class_names = train_ds.class_names

# Visualize a few rock samples

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

from tensorflow.keras import layers

num_classes = 6

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)


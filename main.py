import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

### Read in our data

batch_size = 32
img_height = 28
img_width = 28

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

### Visualize a few rock samples to see what we're working with

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

### Setup a simple classification model 

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(len(class_names),activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=.01, verbose=99)

model.fit(train_ds,
  validation_data=(val_ds),
  epochs=10,
  callbacks=[early_stop]
)

### Remove the final dense softmax layer so we can output a feature representation
model2 = tf.keras.Model(model.input,model.layers[5].output)

#Get validation data all in one batch for plotting
all_validation = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=2999)

data = all_validation.as_numpy_iterator()
batches = [x for x in data]
image_data = [x[0] for x in batches]
classes = [x[1] for x in batches]

#Get the encoded representations
embedded_data = model2.predict(image_data[0])

#Map the integers to the proper class name
mapped_classes = list(map(lambda x: class_names[x], classes[0]))

### Dimensionality Reduction so we can plot
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=20,random_state=42)
svd.fit(embedded_data)

#Plot a few to see if the classes are separated reasonably
images_svd = svd.transform(embedded_data)
x = [x[1] for x in images_svd]
y = [x[2] for x in images_svd]

fig, ax = plt.subplots()
scatter_x = np.array(x)
scatter_y = np.array(y)
group = np.array(mapped_classes)
for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(scatter_x[i], scatter_y[i], label=g)
ax.legend()
plt.title("2D Dimensionality Reduction of Rock Features from NN")
plt.show()
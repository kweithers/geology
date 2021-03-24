import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

### Set up the data

batch_size = 32
img_height = 28
img_width = 28

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

### Visualize a few rock samples to see what we're working with

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4,4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


### Set up a simple classification model 

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

### Remove the final dense softmax layer so it can output a feature representation
feature_model = tf.keras.Model(model.input,model.layers[5].output)

#Get some validation data for plotting
plotting_data = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=2999)

data = plotting_data.as_numpy_iterator()
batches = [x for x in data]
image_data = [x[0] for x in batches]
classes = [x[1] for x in batches]

#Get the feature representations
features = feature_model.predict(image_data[0])

#Map the integers to the proper class name
mapped_classes = list(map(lambda x: class_names[x], classes[0]))

### Dimensionality Reduction for plotting
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=20,random_state=42)
svd.fit(features)

#Plot a few to see if the classes are separated reasonably
images_svd = svd.transform(features)
x = [x[0] for x in images_svd]
y = [x[1] for x in images_svd]

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

# Looks good - Now I have a 64 dimensional vector representing each image
# The distance between two vectors represents image similarity!
# I can calculate and cache these vectors for every image, then
# do a lookup for the closest k images when I am presented with a query image

one_batch = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/kevinweithers/Documents/miscProjects/Geology/geological_similarity',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=29998)

data = one_batch.as_numpy_iterator()
batches = [x for x in data]
image_data = [x[0] for x in batches]
classes = [x[1] for x in batches]

#Calculcate feature vectors for every image
feature_vectors = feature_model.predict(image_data[0])
mapped_classes = list(map(lambda x: class_names[x], classes[0]))

#Take a sample query vector
image_id = 42
query_vector = feature_vectors[image_id]
query_image = image_data[0][image_id]
query_class = mapped_classes[image_id]

#Initialize distance vector
distances = np.zeros(29998)

#Calculate distances
for i in range(len(distances)):
    distances[i] = np.linalg.norm(query_vector - feature_vectors[i])

#Exclude the query image itself by setting its distance to infinity
distances[image_id] = np.float('inf')

#Find the k closest images
k=8
matches = np.argpartition(distances, k)[:k]

# Plot the query image and its closest matches
plt.figure(figsize=(10, 10))
for i in range(9):
  if i == 0:
      ax = plt.subplot(3,3, i+1)
      plt.imshow(query_image.astype("uint8"))
      plt.title(query_class)
      plt.axis("off")
  else:
      ax = plt.subplot(3,3,i+1)
      plt.imshow(image_data[0][matches[i-1]].astype("uint8"))
      plt.title(mapped_classes[matches[i-1]])
      plt.axis("off")
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

### Read in our data

batch_size = 64
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

### Visualize a few rock samples

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

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

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=.05, verbose=99)

model.fit(train_ds,
  validation_data=(val_ds),
  epochs=10,
  callbacks=[early_stop]
)


### Create an encoder only model
model2 = tf.keras.Model(model.input,model.layers[5].output)

data = val_ds.as_numpy_iterator()
temp = [x for x in data]


training_data = [x[0] for x in temp]
classes = [x[1] for x in temp]

#Get the encoded representations
embedded_data = model2.predict(training_data[2])

mapped_classes = list(map(lambda x: class_names[x], classes[2]))

### Dimensionality Reduction with SVD

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10)
svd.fit(embedded_data)

# class_labels  = [0] * 500 + [1]*500

images_svd = svd.transform(embedded_data)
x = [x[1] for x in images_svd]
y = [x[2] for x in images_svd]
plt.scatter(x,y,c=classes[2])

import numpy as np
fig, ax = plt.subplots()
scatter_x = np.array(x)
scatter_y = np.array(y)
group = np.array(mapped_classes)
for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(scatter_x[i], scatter_y[i], label=g)
ax.legend()
plt.show()


    




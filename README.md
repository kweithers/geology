# Geological Similarity 
#### Author: Kevin Weithers
#### March 2021

## Overview 

This repository allows users to search for similar geological images given a query image. It trains a neural network to classify the training data into one of the six classes. Then, it removes the final layer of the neural network such that it creates sub-model that outputs a feature vector that represents an image. The similarity of two images is represented by the Euclidean distance between their feature vectors.

Now, for any query image, I can run it through the sub-model to create its feature vector, and calculate the distance between this new feature vector and each feature vector in the dataset. The k most similar images are the images with the k smallest distances.

### First Look at the Data

![]("samples.png")

There are clear similarities among samples of the same class, and clear differences between samples of different classes. If I train a model to classify these images into their classes, the model will learn the most salient features that describe an image. I will use these features as a measure of image similarity.

### Neural Network

I trained a simple convolutional neural network with the following architecture. 

```python
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(len(class_names),activation='softmax'),
])
```

After training for a couple epochs, the validation set accuracy was already ~95%, so I didn't spend much time tuning the parameters, trying different architectures, or doing any data augmentation. Since the images are quite small, I didn't use any MaxPool layers. 

This dataset is rather small, but if I were working with something much larger, I would try using transfer learning to avoid having to train a model completely from scratch. For example, I would take a ResNet50 trained on ImageNet and use the output of one or more of the intermediate layers as features. If that doesn't produce good enough results, I would freeze the beginning layers (that represent low level features like edges, curves, lines, etc.) and alter the final layers such that they can be trained to classify images for the given use case. Then I could remove the final/softmax layer and use the sub-model to produce a feature vector.

### Feature Extractor

Now that I have the trained model, I remove the final layer and use this sub-model that outputs a 64 dimensional feature vector. I like to visualize the results in order to confirm that the feature vector is meaningful, so I applied Singular Value Decomposition to the validation data feature vectors and then plotted the first 2 components.

![]("2dFeatures.png")

The classes are well separated, so this confirms that the feature vector gives a meaningful representation of an image. I am confident that the distance between the feature vectors of 2 images is a good measure of their similarity.

### Generate k Most Similar Images for a Query Image

I compute the feature vector for every image in the dataset. Then, when presented with a query image, I compute its feature vector. Next, I calculate the distance between this new vector and every other feature vector in the dataset. Since I only need the top k matches, I don't need to sort all of the distances, I can use a partition algorithm like np.argpartition that has better asymptotic time complexity. (Not impactful in this case since the dataset is small, but impactful if working with a huge dataset.)

Now, given any query image, it returns the query image and the k most similar images and plots them.

Here are some example outputs: The query image is in the top left, shown with the 8 most similar images (in no particular order).

![]("similar4.png")

![]("similar3.png")

![]("similar5.png")

![]("similar2.png")

![]("similar1.png")

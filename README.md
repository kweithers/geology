# Geological Similarity 
#### Author: Kevin Weithers
#### March 2021

## Overview 

This repository allows users to search for similar geological images given a query image. It trains a neural network to classify our training data into one of the six classes. Then, it removes the final layer of the neural network such that we have a sub-model that outputs a feature vector that represents an image. The similarity of two images is represented by the distance between their feature vectors.

Now, for any query image, we can run it through the model to create its feature vector, and return the k most similar images by comparing its feature vector to all of the other feature vectors in the dataset.

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

This dataset is rather small, but if I were working with something much larger, I would try using transfer learning to avoid having to train a model completely from scratch. For example, I would take a ResNet50 trained on ImageNet and use the output of one or more of the intermediate layers as features. If that doesn't produce good enough results, I would freeze the beginning layers (that represent low level features like edges, curves, lines, etc.) and alter the final layers such that they can be trained to classify images for the given use case. Then we could remove the final/softmax layer and use the sub-model to produce a feature vector.

### Feature Extractor

Now that we have our trained model, we can remove the final layer and use this submodel that outputs a 64 dimensional feature vector. I like to visualize the results in order to confirm that the feature vector is meaningful, so I applied Singular Value Decomposition to our validation data and then plotted the first 2 components.

![]("2dFeatures.png")

The classes are well separated, so this confirms that our feature vector gives a meaningful representation of an image. I am confident that the distance between the feature vectors of 2 images is a good measure of their similarity.

### Generate k Most Similar Images for a Query Image

I compute the feature vector for every image in our dataset. Then, when presented with a query image, I compute its feature vector. It calculates the distance between this new vector and every other feature vector in our dataset. Since we only need the top k matches, we don't need to sort all of the distances, we can use a partition algorithm like np.argpartition that has better asymptotic time complexity. (Not impactful in this case since our dataset is small, but impactful if working with a huge dataset.)

Now, given any query image, it returns the query image and the k most similar images and plots them.


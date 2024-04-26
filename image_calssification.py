# Image Classifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# For certificate permissions
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Just checks
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))
print('x_train_shape:', x_train.shape)
print('y_train_shape:', y_train.shape)
print('x_test_shape:', x_test.shape)
print('y_test_shape:', y_test.shape)

index = 2
#Show image as picture
img = plt.imshow(x_train[index])

#Image label
print('The image label is:', y_train[index])

#Getting the image classification
classification = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('The image class is:', classification[y_train[index][0]])

#Converting labels into a set of 10 numbers to input into neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print(y_train_one_hot)
print('The one hot label is:', y_train_one_hot[index])

#Normalising the pixels within 0 and 1
x_train = x_train / 255;
x_test = x_test /255
x_train[index]

#Creating the model's architecture
model = Sequential()

#Adding the first layer which is a convolutional layer
model.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (32,32,3)))

#Adding a pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#Adding another convolution layer
model.add(Conv2D(32, (5,5), activation = 'relu'))

#Adding another pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#Adding a flattening layer
model.add(Flatten())

#Adding a layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

#Adding a dropout layer with 50% dropout rate
model.add(Dropout(0.5))

#Adding a layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

#Adding a dropout layer with 50% dropout rate
model.add(Dropout(0.5))

#Adding a layer with 250 neurons
model.add(Dense(250, activation = 'relu'))

#Adding a layer with 10 neurons
model.add(Dense(10, activation = 'softmax'))

#Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Training the model
hist = model.fit(x_train, y_train_one_hot, batch_size = 256, epochs = 10, validation_split = 0.2)

#Evaluating model using test data set
model.evaluate(x_test, y_test_one_hot)[1]

#Visualising model's accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

#Visualising model's loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

#Testing the model with an example
from google.colab import files
uploaded = files.upload()

#Showing the image
new_image = plt.imread('cat.4016.jpg')
img = plt.imshow(new_image)

#Resizing the image
from skimage.transform import resize
resized_image = resize(new_image, (32, 32, 3))
img = plt.imshow(resized_image)

#Getting the models' predictions
predictions = model.predict(np.array([resized_image]))

#Showing the predictions
predictions

#Sorting the predictions from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp
            
#Showing the sorted labels in order
print(list_index)

#Printing the first 5 predictions
for i in range(5):
    print(classification[list_index[i]], ':', predictions[0][list_index[i]]*100, '%')
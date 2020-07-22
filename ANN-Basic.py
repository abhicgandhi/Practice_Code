
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = data.load_data()

train_images = train_images/255.0 #divide the numbers so, the rbg value ranges towards 0 and 1
test_images = test_images/255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(train_images[12])
plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"), #Rectifier Unit,makes all negative values 0
	keras.layers.Dense(10, activation="softmax")
	]) #softmax - probablitistic connection


model.compile(optimizer="adam", loss="sparse_categorical_croessentropy", metrics=["accuracy"])
model.fit(train_images,train_labels, epochs=10)

model.evaluate(test_images,test_labels)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\n Accuracy:', test_acc)

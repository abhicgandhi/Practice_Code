from keras.layers import Input, Lambda, Dense, Flatten, Conv2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_SIZE = [224,224] #Resize the images
train_path = 'Dataset/Train' #training data location
test_path = 'Dataset/Test' #test data location

vgg19 = VGG19(input_shape=IMAGE_SIZE + [3], weights = 'imagenet', include_top=False)
#Image size + 3(rgb)(depth),use imagenet weights, cut the first layer(we'll determine image size and last layer(how many number of classes(infected and not should be there)
vgg19.summary()
#transfer learning techniques will be used

for layer in vgg19.layers:
    layer.trainable = False #we're not training every layer

folders = glob('Dataset/Train/*')

x = Flatten()(vgg19.output) #flatten our output, need to flatten before adding dense layers

prediction = Dense(len(folders), activation= 'softmax')(x) #output will have 2 layers and the activation will be softmax(2 nodes)
model = Model(inputs=vgg19.input, outputs = prediction) #the input and output used to create model

model.summary() #see structure of the model

from keras.layers import MaxPooling2D
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()

#Uncomplete,
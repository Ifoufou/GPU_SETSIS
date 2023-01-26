from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

def simplecnn():
  model = Sequential()

  model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', input_shape=(28,28,1)))
  model.add(Activation("relu"))
  model.add(Conv2D(filters=32, kernel_size=(5,5),
                   padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(filters=64, kernel_size=(5,5),
                   padding='same', activation='relu'))
  model.add(Conv2D(filters=64, kernel_size=(5,5),
                   padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(120, activation='relu'))
  model.add(Dense(84, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  return model





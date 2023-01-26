from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from model import simpleCNN

model = simpleCNN.simplecnn()
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

data_label = ["t-shirt", "pantalon", "pull-over", "robe", 
              "manteau", "sandale", "chemise", "basket",
              "sac", "bottine"]

from keras.datasets import fashion_mnist
((trainX,trainY),(testX,testY)) = fashion_mnist.load_data()

if K.image_data_format() == "channels_first":
  trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
  testX = testX.reshape((testX.shape[0], 1, 28, 28))
else:
  trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
  testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainX = trainX.astype("float32")/255.0
testX = testX.astype("float32")/255.0
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

hist = model.fit(trainX, trainY,
                 validation_data=(testX, testY),
                 batch_size=16 , epochs=10)

model.evaluate(testX, testY)

from matplotlib import pyplot as plt
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Precision du modele')
plt.ylabel('Precision')
plt.xlabel('Iteration')
plt.legend(['Apprentissage', 'Test'], loc='upper left')
plt.show()

from keras.models import load_model
model.save('model.h5')

from keras.models import load_model
model = load_model('model.h5')


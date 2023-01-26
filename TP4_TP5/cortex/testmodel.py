import numpy as np
from keras.datasets import fashion_mnist
from keras import backend as K
from keras.utils import np_utils
from matplotlib import pyplot as plt

data_label = ["t-shirt", "pantalon", "pull-over", "robe",
              "manteau", "sandale", "chemise", "basket",
              "sac", "bottine"]

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

if K.image_data_format() == "channels_first":
  testX = testX.reshape((testX.shape[0], 1, 28, 28))
else:
  testX = testX.reshape((testX.shape[0], 28, 28, 1))

testX = testX.astype("float32")/255.0
testY = np_utils.to_categorical(testY, 10)

tabimages =[]
rand_indexes = np.random.choice(np.arange(0 ,len(testY)),
               size =(16,))
for i in rand_indexes:
  results = model.predict(testX[np.newaxis, i])
  prediction = results.argmax(axis=1)
  label = data_label[prediction[0]]
  if K.image_data_format() == "channels_first":
    image = (testX[i][0]*255).as_type('uint8')
  else:
    image = (testX[i]*255).as_type('uint8')

  couleurTXT = (0, 255, 0)
  if prediction[0] != np.argmax(testY[i]):
    couleurTXT = (255, 0, 0) # mauvaise prediction

  image = cv2.merge([image]*3)
  cv2.putText(image, label, (5,20), cv2.FONT_HERSHEY_SIMPLEX,
              0.75, couleurTXT, 2)
  tabimages.append(image)

plt.figure(figsize=(7,7))
for i in range (0, len(tabimages)):
  plt.subplot(4, 4, i+1)
  plt.imshow(tabimages[i])
plt.show()


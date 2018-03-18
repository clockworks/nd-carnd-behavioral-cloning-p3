import csv
import cv2
import numpy as np

lines = []
with open('./data/lap1/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
images = []
measurements = [] 
for line in lines:
  source_path = line[0]
  tokens = source_path.split('/')
  filename = tokens[-1]  
  local_path = "./data/lap1/IMG/" + filename
  image = cv2.imread(local_path)
  images.append(image)
  measurement = line[3]
  measurements.append(measurement)
    
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * 1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
    
    
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#lenet implemented

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
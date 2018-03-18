import csv
import cv2
import numpy as np

lines = []
with open('./data/sample/data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = [] 
for line in lines:
  source_path = line[0]
  tokens = source_path.split('/')
  filename = tokens[-1]
  local_path = "./data/sample/data/IMG/" + filename
  image = cv2.imread(local_path)
  images.append(image)
  measurement = line[3]
  measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')

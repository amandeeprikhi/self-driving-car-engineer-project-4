import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

lines = []
correction_factor = 0.2
# reading the driving log csv file
with open('training_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
images = []
measurements = []

# iterating through the csv file row by row
for line in lines:
    # taking the left, right and center images into consideration
  for i in range(3):
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = 'training_data/IMG/' + filename
    img = mpimg.imread(current_path)
    images.append(img)
    if(i == 0):
        #measuring the steering angle for each image 
        measure_center = float(line[3])
        measurements.append(measure_center)
    elif(i == 1):
        #adding correction factor to the left images
        measure_left = float(line[3]) + correction_factor
        measurements.append(measure_left)
    elif(i == 2):
        # subtracting correction factor from the right images
        measure_right = float(line[3]) - correction_factor
        measurements.append(measure_right)
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = 'training_data/IMG/' + filename
  img = mpimg.imread(current_path)
    # flipping the images to augment the data
  flipped_image = np.fliplr(img)
  measurement = float(line[3])
  flipped_measurement = -measurement
  images.append(flipped_image)
  measurements.append(flipped_measurement)
  
# feeding the data into a numpy array
X_train = np.array(images)
y_train = np.array(measurements)

#importing the required keras modules
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#model architecture begins
model = Sequential()
#normalizing the dataset
model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape = ( 160, 320, 3)))
#crooing the images in the dataset
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5,subsample = (2,2), activation = 'relu'))
#adding dropout in order to reduce overfitting of the data
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#using Adam optimiser
model.compile(loss='mse', optimizer='adam')
#training the model
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)
#saving the model
model.save('model.h5')
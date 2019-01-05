#### Data Import ####
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


### Generator for pre processing and using on the fly in case of large data sets
import csv
import sklearn
from sklearn.model_selection import train_test_split

## Reading csv line by line and storing
samples = []
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
## Splitting data in validation and training 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples,batch_size = 32,correct_value = 0.2):
    correction_dict = {0:0,1:-correct_value,2:correct_value}
    num_samples = len(samples)
    while 1 :
        for offset in range(0,num_samples,batch_size):
            sklearn.utils.shuffle(samples)
            batch_samples = samples[offset :offset+batch_size]
            images = []
            measurement = []
            for row in batch_samples:
                # Read all images taken from cameras on right,left and center
                for loc in [0]:
               
                    file_path = "/opt/data/{}".format(row[loc].strip())
                    img = cv2.imread(file_path)
                    images.append(img)
                    measurement.append(float(row[3]))
                    ## Data augmentation by flipping the image
                    flip_img = np.fliplr(img)
                    images.append(flip_img)
                    measurement.append(-(float(row[3])))
                    
            images = np.array(images)
            measurement = np.array(measurement)
#             print("Image Data Shape : {} \nShape of Labels : {}".format(images.shape,measurement.shape))
            yield sklearn.utils.shuffle(images,measurement)
            
train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)

### 3. Nvidia Network Implementation
from keras.layers import Input, Lambda, Cropping2D,Convolution2D,Flatten,Dense,Dropout
from keras.models import Model
## Input Layer
inp = Input(shape=(160 , 320 , 3))
## Normalization layer
norm_inp = Lambda (lambda x : (x/255.)-0.5)(inp)
## Cropping layer
crop_inp = Cropping2D(cropping=((70,25),(0,0)))(norm_inp)
## Conv layers
conv1 = Convolution2D(24,5,5,subsample=(2,2),activation="relu")(crop_inp)
conv2 = Convolution2D(36,5,5,subsample=(2,2),activation="relu")(conv1)
conv3 = Convolution2D(48,5,5,subsample=(2,2),activation="relu")(conv2)
conv4 = Convolution2D(64,3,3,activation="relu")(conv3)
conv5 = Convolution2D(64,3,3,activation="relu")(conv4)
## Flatten Layer
flat = Flatten()(conv5)
flat_drop = Dropout(0.2)(flat)
## Dense fully connected layers
x1 = Dense(100,activation="elu")(flat_drop)
x1_drop = Dropout(0.5)(x1)
x2 = Dense(50,activation="elu")(x1_drop)
x2_drop = Dropout(0.5)(x2)
x3 = Dense(10,activation="elu")(x2_drop)
## output layer
out = Dense(1,activation="elu")(x3)
## defining model
model = Model(input = inp,output=out)
## Defining loss function and optimizer
from keras import optimizers
adam = optimizers.Adam(lr=0.00001)
model.compile(loss='mse',optimizer=adam)
## Fitting the model with validation split and shuffle
# model.fit(images,measurement,validation_split=0.2,shuffle=True,nb_epoch=2)
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator,                                             validation_steps=len(validation_samples), epochs=10, verbose = 1)
## save the model
model.save("model.h5")

### Visualizing the training and validation losses across epochs
# ## saving history object to visulaize loss
# history_object = model.fit(images,measurement,validation_split=0.2,shuffle=True,nb_epoch=5,verbose=1)
# model.save("model.h5")
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("loss_graph.png")


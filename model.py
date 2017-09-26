# Routine to plot a graph showing angle distributions
import scipy.stats as stats
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
path = '../../udacity/CarND-Behavioral-Cloning-P3/data_comb/'
#Loading CSV File
lines = []
with open(path+'driving_log.csv') as csvfile:
    count=0
    count2=0
    angle=0
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        angle = float(line[3])
        if (angle > -0.001 and angle < 0.001):
            if (count == 0):
                lines.append(line+[0])
            count +=1
            if (count >= 100): count = 0
        elif (abs(angle) > 0.001 and abs(angle) < 0.1):
            if (count2 == 0):
                lines.append(line+[0])
            count2 +=1
            if (count2 >= 2): count2 = 0
        else:
            lines.append(line+[0])
            lines.append(line+[1])
            
            
    
data=np.array(lines)[:,3].astype(float)
data=sorted(data)
fit = stats.norm.pdf(data, np.mean(data), np.std(data))
plt.plot(data,fit,'-r')
plt.hist(data,100,[-0.3,0.3],normed=True)
plt.show()


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- img: Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,color = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    
    # Shear
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    
    # Apply values
    img = cv2.warpAffine(img,Rot_M,(cols,rows),borderMode=1)
    img = cv2.warpAffine(img,Trans_M,(cols,rows),borderMode=1)
    img = cv2.warpAffine(img,shear_M,(cols,rows),borderMode=1)
    return img

def crop_resize(image):
    #image = cv2.resize(image[60:140,:], (200,66))
    return image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.27
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = path+'IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    image = crop_resize(image)
                    if (batch_sample[4] == '1'):
                        image = transform_image(image,10,2,4)
                    measurement = float(batch_sample[3])
                    if i == 1:
                        measurement = measurement + correction
                    elif i == 2:
                        measurement = measurement - correction
                    images.append(image)
                    measurements.append(measurement)

            #Data Augmentation
            #Flipping the images
            #Multiplying the steering angle measurement with -1
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            #Converting the list into numpy arrays
            #This constitutes Features and Labels
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#Model Architecture starts from here
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Model

model = Sequential()

#Preprocessing the images
#Normalization and Mean Centre
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3)))

#Image cropping
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Nvidia Model starts here
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Dropout(.1))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Dropout(.2))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Dropout(.2))
model.add(Convolution2D(64,3,3,subsample=(2,2), activation="relu"))
model.add(Flatten())
model.add(Dropout(.3))
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dropout(.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2*3, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
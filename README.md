# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the files for my solution to the Behavioral Cloning Project.

In this project, I will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo) to train a neural network and then use this model to drive the car autonomously around the track.

The project include four files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* video.py (script to convert captured images from the driving into a video)
This README file describes how to output the video in the "Details About Files In This Directory" section.


The Project
---
The goals of this repository are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded from [here](https://github.com/udacity/self-driving-car-sim)


## Approach

### Base Model and Adjustments

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the nVidia model (and later in the student forum, the comma.ai model) - the diagram below is a depiction of the nVidia model architecture.

<img src="./img/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text.
Relu activation has been used as recommended. The paper doesn't mention any kind of regularization. Dropout had been used in order to mitigate overfitting.  These are the values I have used based on trial and error:

* model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
* model.add(Dropout(.1))
* model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
* model.add(Dropout(.2))
* model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
* model.add(Dropout(.2))
* model.add(Convolution2D(64,3,3,subsample=(2,2), activation="relu"))
* model.add(Flatten())
* model.add(Dropout(.3))
* model.add(Dense(100))
* model.add(Dropout(.5))
* model.add(Dense(50))
* model.add(Dropout(.5))
* model.add(Dense(10))
* model.add(Dropout(.5))
* model.add(Dense(1))

The Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). The final layer (depicted as "output" in the diagram) is a fully-connected layer with a single neuron. 

### 2. Collecting Additional Driving Data

Udacity provides a dataset that can be used alone to produce a working model. However, I decided to collect my own. 
Data has been captured driven five laps unclockwise and file laps clockwise to the track. A Car RC controller connected as a joystick has been used to have better smooth data.
No edge recovery data was recorded as I will use lateral cameras to control it. This data was merged with the udacity provided composing after filters a total of 67920 samples.
 

### 3. Loading and Preprocessing

In training mode, the simulator produces three images per frame while recording corresponding to left-, right-, and center-mounted cameras, each giving a different perspective of the track ahead. 
The simulator also produces a `csv` file which includes file paths for each of these images, along with the associated steering angle, throttle, brake, and speed for each frame. 

Most data in the capture data correponds to '0' band angle meaning that the data is biased to 0. To correct that first I am filtering the csv file to remove data very close to 0 rad (I keep 1 sample every 100), and also a filter of 1 out of 2 for the range below 0.1 rad.

<img src="./img/histogram.png?raw=true" width="400px">
	
Note that the joystick had a limitation in 15° (despite being well calibrated in windows using the full range) that corresponds to 0.26 rad. That is the reason that the is no data above that value.
 
I aslo have included an extra column to the list to inform to the generator wich data will need to be tranformed. Data above 0.1 rad will be duplicated, one corresponding to the original image and other that will suffer some transformation (rotation, shear and tranlation).

After the list of images and angles is build it will be passed to a generator that will be in charge of load it secuentially to the CNN. In this phase I include the side cameras (adjusted by +0.27 for the left frame and -0.27 for the right) and also a flipped version of all the images to have a median of 0 and better distribution.
Images produced by the simulator in training mode are 320x160 but the top 70 pixels and the bottom 25 pixels are cropped from the image in the CNN to increse speed (work done in parallel in the GPU). 
The use of a generator gives a really good performance and saves a lot of memory if we compare it to loading the complete data set into memory. A batch size of 32 performed well in my hardware (GTX 970 GPU).

### 4. Training

Training the network using Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). Two eppochs were used. 
Training and validation data were split in 80% and 20% respectively. Test data will come from the simulator and will be tested using drive.py.


Below is an example of the training data.

AÑADIR IMAGENESSSSSSSSSSSSSS

## Details About How to Use the files in this directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.



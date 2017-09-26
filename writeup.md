# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the nVidia model (and later in the student forum, the comma.ai model) - the diagram below is a depiction of the nVidia model architecture.

<img src="./img/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text.
Relu activation has been used as recommended. The final layer (depicted as "output" in the diagram) is a fully-connected layer with a single neuron.  

#### 2. Attempts to reduce overfitting in the model

The paper doesn't mention any kind of regularization. Dropout had been used in order to mitigate overfitting (model.py lines 156-169).  These are the values I have used based on trial and error:

* model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
* **model.add(Dropout(.1))**
* model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
* **model.add(Dropout(.2))**
* model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
* **model.add(Dropout(.2))**
* model.add(Convolution2D(64,3,3,subsample=(2,2), activation="relu"))
* model.add(Flatten())
* **model.add(Dropout(.3))**
* model.add(Dense(100))
* **model.add(Dropout(.5))**
* model.add(Dense(50))
* **model.add(Dropout(.5))**
* model.add(Dense(10))
* **model.add(Dropout(.5))**
* model.add(Dense(1))

The model was trained and validated on different data sets to ensure that the model was not overfitting. One using Udacity data and other using my own data. Final model was train in a data set resulted from merging both. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 172).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a center lane driving only becouse using data recovering from the left and right sides of the road were always giving worse results. Five laps in each direction were used. The final model was trained in a merge dataset with Udacity one.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the nVidia model (and later in the student forum, the comma.ai model) - the diagram below is a depiction of the nVidia model architecture.

<img src="./img/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text.
Relu activation has been used as recommended. 

#### 3. Creation of the Training Set & Training Process

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the nVidia model (and later in the student forum, the comma.ai model) - the diagram below is a depiction of the nVidia model architecture.

<img src="./img/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text.
Relu activation has been used as recommended. 

Most data in the capture data correponds to '0' band angle meaning that the data is biased to 0. To correct that first I am filtering the csv file to remove data very close to 0 rad (I keep 1 sample every 100), and also a filter of 1 out of 2 for the range below 0.1 rad.

<img src="./img/histogram.png?raw=true" width="400px">
	
Note that the joystick had a limitation in 15° (despite being well calibrated in windows using the full range) that corresponds to 0.26 rad. That is the reason that the is no data above that value.
 
I aslo have included an extra column to the list to inform to the generator wich data will need to be tranformed. Data above 0.1 rad will be duplicated, one corresponding to the original image and other that will suffer some transformation (rotation, shear and tranlation).

After the list of images and angles is build it will be passed to a generator that will be in charge of load it secuentially to the CNN. In this phase I include the side cameras (adjusted by +0.27 for the left frame and -0.27 for the right) and also a flipped version of all the images to have a median of 0 and better distribution.
Images produced by the simulator in training mode are 320x160 but the top 70 pixels and the bottom 25 pixels are cropped from the image in the CNN to increse speed (work done in parallel in the GPU). 
The use of a generator gives a really good performance and saves a lot of memory if we compare it to loading the complete data set into memory. A batch size of 32 performed well in my hardware (GTX 970 GPU).

















I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
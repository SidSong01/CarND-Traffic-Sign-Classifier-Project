## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tried out my model on images of German traffic signs that I found on the web.

The main execution is `Traffic_Sign_Classifier.ipynb`.

Also there is a comparison between with and without data augmentation.

The execution without data augmentation is described in `Traffic_Sign_Classifier_without_dataAugmentation.html`.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


# **Traffic Sign Recognition** 
---


[//]: # (Image References)

[image1]: ./examples/training_data.png
[image2]: ./examples/data_distribution.png
[image3]: ./examples/after_processing.png
[image4]: ./examples/acc.png
[image5]: ./signs/label1.jpg
[image6]: ./signs/label13.jpg
[image7]: ./signs/label14.jpg
[image8]: ./signs/label15.jpg
[image9]: ./signs/label17.jpg
[image10]: ./examples/top5softmax_for_new_imgs.png
[image11]: ./examples/test.png
[image12]: ./examples/visualize_network_state.png

### Data Set Summary & Exploration

#### 1. A basic summary of the data set.


* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Visualization of the dataset.

Visualization of the data set and the label.

![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data sets distribution.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. How I preprocessed the image data.

Here I decided to convert the images to grayscale as previous experiments show using gray images produces better results than using color images. Histogram equalization was applied to improve visibility of the signs and normalization with zero mean was used to facilitate the convergence of the optimizer during training.
Visualization after the pre-processing.

![alt text][image3]

##### Also, there is a data augmentaion, after that the number of training examples is 215000.

#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale images   							| 
| 3x3x12 kernel, stride1, convolution with ReLU activation	| outputs 30x30x12 	|
| 3x3x24 kernel, stride1, convolution with ReLU activation | outputs 28x28x4 		|
| 5x5x36 kernel, stride1, convolution with ReLU activation | outputs 10x10x36 	|
| 5x5x48 kernel, stride1, convolution with ReLU activation	| outputs 6x6x48				|
| Flatten, concatenate 14x14x24 and 3x3x48 | outputs 5136		|
|	First fully connected with 512 neurons and dropout | outputs 512 |
|	Second fully connected with 256 neurons and dropout | outputs 256 |
|	Outputs with 43 neurons, the label number |	outputs	43	|
 


#### 3. Model training.

batch_size = 32
keep_prob = 0.7
The maximum number of epochs is 30, and if up to 5 epochs there is no change in the accuracy, the training will be forced to stop. So first I process the data to 32x32x1 images and then I input the training data to the network, getting the logits. And use tf.nn.softmax_cross_entropy_with_logits to get the cross entropy. I use AdamOptimizer to train my model, with a learning rate of 0.0001. In each training epoch, I will shuffle the training data of the single epoch first, and then feed them into the model. After that I will test the accuracy with the epoch’s training data and validation data. And for the two fully connected layers, I keep a keep probability as 0.7 for drop out operation.

#### 4. Testing

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.8%
* test set accuracy of 96.69%

The data augmentation has helpped improve the acc, without it, the training results are:
* training set accuracy of 100%
* validation set accuracy of 97.6%
* test set accuracy of 95.566%

#### 5. Plot the acc.

![alt text][image4]

### Test a Model on New Images

#### 1. Testing examples

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]


#### 2. Testing results for new images.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h | 30 km/h	|
| Yield | Yield	|
| Stop Sign | Stop sign | 
| No vehicle			| No vehicle	|
| No entry | No entry |


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%. This is better compares favorably to the accuracy on the test set of 96.69%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

The top 5 softmax probabilities for each image along with the sign type of each probability are shown.

![alt text][image10]

The code for making predictions on my final model is located after the `Output Top 5 Softmax Probabilities For Each Image Found on the Web` cell of the Ipython notebook.

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

![alt text][image11]

![alt text][image12]

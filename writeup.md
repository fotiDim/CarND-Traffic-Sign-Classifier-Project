# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image4]: ./extra_traffic_signs/1.jpg "Traffic Sign 1"
[image5]: ./extra_traffic_signs/11.jpg "Traffic Sign 2"
[image6]: ./extra_traffic_signs/17.jpg "Traffic Sign 3"
[image7]: ./extra_traffic_signs/35.jpg "Traffic Sign 4"
[image8]: ./extra_traffic_signs/13.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I ploted histograms of the label distrubution for each of the training, validation and test datasets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data to make the training more efficient. I didn't do any grayscaling as the color can be usefull for recognizing traffic signs.

I didn't add any additional augmented data as I achieved quite a high accuracy by only tweaking slightly our network.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the exact LeNet-5 architecture with the only difference that I replaced the last 2 `relu` activations of the 2 fully connected layers with `dropout`.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 20 `Epochs` and a batch size of `32`. Learning rate was left at `0.001`. The optimizer was also left to the suggested one `AdamOptimizer`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.66%
* validation set accuracy of 94.06%
* test set accuracy of 91.66%

If a well known architecture was chosen:
* What architecture was chosen? As stated above I used the LeNet-5 architecture which some tiny modifications.
* Why did you believe it would be relevant to the traffic sign application? LeNet works well for image recognition of digits so it seemed like a good starting point.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? It seems that the model might be overfitting slightly but over all the performance is acceptable on the test set. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

As I live in Germany I had the chance to get on the street and capture some German traffic sign pictures on my own.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        		| 
|:-------------------------------------:|:---------------------------------:| 
| Speed limit (30km/h)                  | Correct					        |
| Right-of-way at the next intersection | Correct				    	    |
| Yield					                | Correct						    |
| No entry	      		                | Correct					 		|
| Ahead only		                    | Correct							|

The accuracy on the five German traffic signs was 100%. It should be noted that the image contrast was low, yet the recognition worked exceptionally.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the next to last cell of the Ipython notebook. The 1st result for each image had the following probabilities.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield  					    				| 
| 1     				| Ahead only 									|
| 1 					| No entry										|
| .78	      			| Speed limit (30km/h)					 		|
| 1 				    | Right-of-way at the next intersection    		|

You can find detailed charts of the 5 first probabilities for each image in the Ipython notebook.

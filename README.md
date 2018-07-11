[//]: # (Image References)
[car]: ./examples/car_not_car.png
[hog]: ./examples/HOG_example.jpg
[windows]: ./examples/sliding_windows.png
[union]: ./examples/sliding_windows_union.png
[output]: ./examples/example_output.png
[heat]: ./examples/bboxes_and_heat.png
[label]: ./examples/labels_map.png

#Vehicle Detection Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![output]

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start wivth the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

## Data 

For this project, I use the data prepared by Udacity. The data is divided into two sets corresponding for the two labels: **cars and not-cars**.

![car]

    Number of car images: 8792
    Number of non-car images: 8968

The vehicle data set contains the `GTI*` folders which are time-series data where identical cars may appear multiple times. Data is shuffled before splitting 4:1 into training and testing set.

## Feature Extraction and Training

I selected three categories of feature for this classification: spatial (the raw pixel value of an image), color in different channels and structural (histogram of oriented gradients). 

#### 1. Histogram of Oriented Gradients (HOG)

To select the parameters for extracting the HOG that best describes the shape of the object in images, I first experiment with the different color space, including `RGB`,`HSV`, `YUV`, and `YCrCb`, and judge based on the HOG extracted from the three different channels after the image is converted to the particular color space. Then I tune the orientation (number of available directions for the gradient), pixel per cell and cell per block.

![hog]

I settled with the following parameters:

    COLOR_SPACE = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    ORIENT = 12  # HOG orientations
    PIX_PER_CELL = 8 # HOG pixels per cell
    CELL_PER_BLOCK = 2 # HOG cells per block
    HOG_CHANNEL = "ALL" # Can be 0, 1, 2, or "ALL"
    SPATIAL_SIZE = (16, 16) # Spatial binning dimensions
    HIST_BINS = 32     # Number of histogram bins
    # HIST_BINS = 30   # Number of histogram bins
    SPATIAL_FEAT = False # Spatial features on or off
    HIST_FEAT = True # Histogram features on or off
    HOG_FEAT = True # HOG features on or off

These parameters are found in `svgClassifier.py`.

#### 2. Training a Classifier 

I trained a linear SVM using the parameters mentioned above. The length of the feature vector is `7152`. The data consisted of `8792` vehicle images and `8968` non-vehicle images. I trained the model with 80% of the images and test them on the remaining 20%.

Besides the feature parameters, I also tried to tune the following SVM parameters using GridSearchCV(): 

	{'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ('linear', 'rbf')}

However, this tuning process is extremely timing consuming and the result is not too explicitly different from training a regular linear SVM without tuning any other parameters. In both cases, 98% training accuracy is achieved, so I stick with the model trained without further parameter tuning. 

The training procedure is found in both `svClassifier.py` and `detect.ipynb`

### Sliding Window Search

#### 1. Implementing the Sliding Window Search.

I adapted much of the solution from the lessons on sliding window technique. There are total of 5 layers of windows each with different window size. 

![windows]

In union, here is how all the layers overlapped.

![union]

#### 2. Filtering False Positive and Resolve Overlapping windows

Since there are many points at which layers of window searching overlap each other, there will be multiple detection of the same object on the input image. To concatenate the result, and as well as to remove false positive, I created a heat map from all the detected windows.

![heat]

The false positive is usually 'cooler' than the true positive, so I set a threshold for the hot region where region lower than such threshold will be zeroed out.

![label]

In addition, some false positive will appear on the left side of image at the other side of the road going in the reverse direction. I set x-position threshold where any window left of such x-position will be discarded.

Finally, I also enforced a minimum width for the detected window, and discard any window if is narrower than the threshold.

    MIN_X_POS = 700
    MIN_BOX_WIDTH = 40

The filtering process is found in the file `heatmap.py`
---

### Video Implementation

[![Link to Youtube](https://i.ytimg.com/vi/Hov1K5ptxa8/hqdefault.jpg?sqp=-oaymwEZCPYBEIoBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLB0vnjBqv8LRDWrBUIKSirAXbQvtw)](https://youtu.be/Hov1K5ptxa8)

### Using YOLOv2
[![Link to Youtube](https://i.ytimg.com/vi/3k1MoMVDxEk/hqdefault.jpg?sqp=-oaymwEZCPYBEIoBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLCrmlEyPsLQjduIWYkho-U1rxQX8w)](https://youtu.be/3k1MoMVDxEk)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The robustness of this model is solely relying on the position of the sliding windows in relation to where the vehicles are on the road. When doing this project, there are countless times the model fails to detect car on the test images not because the SVM model is underfitting but because the windows are not landing on the exact position of the vehicle. 

It is not practical to overlap numerous layers of searching windows. Reducing the number of searching windows in exchange for efficiency is inevitable. 

The model is not doing too well on areas with shallow. It sometimes will mistaken road signs as vehicles, perhaps the rectangular shape of both objects share some resemblance. 

I think detecting objects with features based on HOG and color patterns is downplaying the complexity of the problem. Certainly, this project is meant to introduce traditional  machine learning techniques, but this highlights the critical dilemma of feature based machine learning and computer vision: engineer complicated features to render high performance but giving up generalization across all situations.

For improving the stability of the detected frames, I can keep track of the detected vehicles frame to frame and normalize the difference between the size of the detected window from frame to frame. In addition, I can also deploy some tracking techniques such as `BOOSTING, MIL, KCF, TLD, MEDIANFLOW, and GOTURN`. There are implementation fo these algorithms in cv2 such as `cv2.TrackerKCF`. 

The mixed used of detection and tracking algorithms can compensate each other and also improve the performance of the detection model, since object tracking is much less expensive computation when compared to object detection. 

Another idea of is that once object is detected, the frame of search can be reduced to fixed pixel area around the detected window in the next frame. 

Comparing the result with that using YOLOv2, the output of the CNN based model is much cleaner. Needless to justify YOLO's effectiveness in the task of object detection. The false positive on the left side of the road can be easily filtered with a single-point reference, but the model did misclassify a car to be a truck on a rare occasion throughout the video and mistaken the traffic sign for a truck. These false positives can be handled by tuning the hyperparameters of the model such the minimum confidence thresholds.

The YOLO output is ran on AWS g2.2xlarge instance. The processing bandwidth is `3.5 FPS`. 


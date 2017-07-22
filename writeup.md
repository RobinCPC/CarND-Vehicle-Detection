**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/noncar_hog.png
[image4]: ./output_images/train_result.png
[image5]: ./output_images/hog_slide1.png
[image6]: ./output_images/box_heat1.png
[image7]: ./output_images/box_heat2.png
[image8]: ./output_images/label_map.png
[image9]: ./output_images/output_boxes.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
## [Source Code](https://github.com/RobinCPC/CarND-Vehicle-Detection)

How to use:

    For traing classifier, please run :
        python main.py -t 1
    For testing pipeline, please run:
        python main.py
    For detecting vehicle in video, please run:
        python main.py -v 1
    Pleae type `python main.py --help`, for more detail.

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The codes for feature extraction of training step are contained in in lines 161 through 173 of the file called `main.py` and lines 86 through 99 of the file called `detect_util.py`.

I started by reading in all the `vehicle` and `non-vehicle` images (in lines 133 through 147 of `main.py`).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is HOG parameters and color space I used in this project:
```python
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9            # HOG orientations
    pix_per_cell = 8      # HOG pixels per cell
    cell_per_block = 2    # HOG cells per block
    hog_channel = "ALL"   # Can be 0, 1, 2, or "ALL"
```
The below figures are the output of `YCrCb` color space and HOG feature transform for `vehicle` and `non-vehicle` images.
From ch1 of HOG, we can tell that the HOG features are very different between `vehicle` and `non-vehicle`. In addition, HOG features of `vehicle` in three channel are kind of similar, but HOG features of `non-vehicle` of ch1 is different from ch2 & ch3.


|       *Car*         |       *Non-Car*     |
| :-------------:     | :-------------:     |
| ![alt text][image2] | ![alt text][image3] |

#### 2. Explain how you settled on your final choice of HOG parameters.

I have tuned the orientation from 6 to 12, and the accuracy is better with bigger orientation number. But bigger orientation will cause more time to compute HOG. Therefore I choose to use 9 orientation. In addition, I have tried different color spaces (such as `RGB`, `HLS`, and `YCrCb`), and I found that `YCrCb` color space have less false positive result when there are tree shadows inside the image.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The codes for this step (training a classifier) is contained in lines 174 through lines 202 in `main.py`. I trained a linear SVM using the HOG features, the spatial features,and  histogram of the original image, and the total number of feature vector of each image is 8460.

The features vector of linear SVM are extracted using the following parameters:
```python
    color_space = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9              # HOG orientations
    pix_per_cell = 8        # HOG pixels per cell
    cell_per_block = 2      # HOG cells per block
    hog_channel = "ALL"     # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32          # Number of histogram bins
    spatial_feat = True     # Spatial features on or off
    hist_feat = True        # Histogram features on or off
    hog_feat = True         # HOG features on or off
```

The following figure is detected result after training with whole dataset of `KITTI` and `GTI`:
![alt text][image4]

In addition, I use `pickle` library to store the parameters of classifier in line 204 through 219 in `main.py`.

### Sliding Window Search and Hog Subsampling

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The codes for this step (sliding window search) is implemented in lines 227 through 237 and lines 451 through 461 in `main.py`, and the default window size is 48X48 with 50% overlap. I have try sliding window with 3 size (48X48, 96X96, 144X144), and I found that window size `48X48` will take more time to compute HOG features and window size `144X144` may cover too large area (produce larger detected box than we need). Therefore, I use window size `96X96` with 75% overlap to search for smaller step than 50%. Smaller step can make detected boxes for the same car get closer, and let them form a larger box by `scipy.ndarray.measure.lable`.


#### 2. Describe how (and identify where in your code) you implemented a HOG subsampling search.  How did you decide what scales to search ?

Because sliding window search will need to compute HOG features in each window area, and those areas are overlapped by each other (depending on the overlap rate). Therefore, it will compute HOG features redundantly and cost extra time which is not good for realtime application. Hence, in lesson, we use another method called Hog subsampling, which compute HOG features for entire image and extra sub-area for prediction. The codes for HOG Subsampling is implemented in lines 10 through lines 76 in the file called `hog_subsample.py`, and the default window size is `64` as scale `1`.  I have manually try scale `1.0` through `2.0` with `0.1` each step, and I found that scale from 1.3 to 1.5 have better accuracy.

#### 3. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For sliding window search, I only use one scale (window size `96X96`) with 75% overlap. For HOG subsampling, I also use one scale (`1.5` in subsampling will use window size `96X96`). And The feature vectors of my linear SVM classifier is YCrCb 3-channel HOG features plus spatially binned color and histograms of color, which provided a nice result.  Here are some example images of both subsampling and Sliding window search:

![alt text][image5]

From the above figure, I also show the computing time of both method, and it shows that HOG subsampling is about 4 times faster than Sliding window search. Therefore, I will use Subsampling method for video or realtime processing.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I use `deque` (double side queue) to record the positions of positive detections in the recent N frames of the video (in lines 247 of `main.py`).  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (from lines 304 through lines 317 of `mian.py`).  However, the threshold in heatmap will also reduce area of correct detection, I set threshold to 0 for my processing. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (from lines 322 to lines 330). In addition, after `label()`, I add another filter inside `draw_labeled_bboxes()`, and this filter compute the average heat point in each label area (from lines 105 to 108 of `hog_sumsample.py`). If one of label area has few average point (less than threshold), it means that this label only detect in one or two frame (total 6 or more frame). Then, I will skip the label with less average point.  Next, I assumed the rest of each blob (with high average heat point) corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]
![alt text][image7]

From above figure, I know that the detection of cars in opposite direction only show up in 3rd & 4th frames, so they will have less heat points after combine the result of 6 frames

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image8]

From above figure, `scipy.ndimage.measurements.label()` can label all detected box in 6 frames.

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image9]

Because the label of cars in opposite direction has less average heat point, the filter skip two labels on the left. Therefore, it only draw bounding box on the right two label, which are cars we  want to detect.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In order to remove false positive (car in opposite direction or shadows on the ground), I try to increase threshold of heatmap, but it will also reduce detect area of corrected detection. Therefore, I use another filter after labeling, and remove some labels with less average heat points.

In addition, there are still some false positives (such as shadows on the ground or trees on the sideroad) show up consist in several frame. It may not easy to remove by the filter. One of way to correct this is to add more tree images (and shadow images) to the non-car group of the training set and train the classifier to make a better prediction.


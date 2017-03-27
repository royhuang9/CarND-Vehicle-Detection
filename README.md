# Vehicle Detection


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/opponent.png
[image51]: ./output_images/heat1.png
[image52]: ./output_images/heat2.png
[video1]: ./project_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This step is calling skimage.feature.hog to extract HoG features for a channel. The code for this step is following:
```python
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        # Use skimage.hog() to get both features
        # If vis is True, visualization is also returned
        return hog(img, orientations=orient, 
                    pixels_per_cell=(pix_per_cell, pix_per_cell), 
                    cells_per_block=(cell_per_block, cell_per_block), 
                    transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I tried different color spaces and compared the accuracy of SVM. The first color space is opponent color space that we can get by RGB as following

![alt text][image3]

The following table is accuracy comparing of SVM, only HoG feature of three channels of color space is considered without bin spatial and color histogram. You can seee only YCrCb can achieve a accuracy higher than 0.99

| color space 	| accuracy 	|
|------------	|----------	|
| Opponent 	| 0.9842 	|
| HSV 	| 0.9851 	|
| LUV 	| 0.9862 	|
| HLS 	| 0.9848 	|
| YUV 	| 0.9873 	|
| YCrCb 	| 0.9910 	|

After including the bin spatital and color histogram, the accuracy can be 0.9961. The colorspace for bin spatial and color histogram is not necessary the same color space to generate HoG features. Finally I choosed YCrCb color space for HoG feature and HSV color space for bin spatial and histogram features.

| HoG color space 	| Histogram color space 	| accuracy 	|
|-----------------	|-----------------------	|----------	|
| YCrCb 	| HSV 	| 0.9961 	|
| YCrCb 	| YCrCb 	| 0.9930 	|
| YCrCb 	| HLS 	| 0.9924 	|
| HSV 	| HSV 	| 0.9848 	|
	
#### 2. Explain how you settled on your final choice of HOG parameters.

After I determine the color space, the next step is to choose HoG parameters. I tried various combinations of parameters and compared the accuracy of SVM.  The results are following:

| Orient 	| pixel_per_cell 	| cell_per_block 	| SVM accuracy 	|
|--------	|----------------	|----------------	|--------------	|
| 6      	| 8*8            	| 2*2            	| 0.9916       	|
| 7      	| 8*8            	| 2*2            	| 0.9930       	|
| 9      	| 16*16          	| 2*2            	| 0.9913       	|
| 9      	| 8*8            	| 4*4            	| 0.9935       	|
| 9      	| 8*8            	| 2*2            	| 0.9961       	|
| 9      	| 8*8            	| 3*3            	| 0.9924       	|

Eventually, I choose the HoG parameters of orientations=9, pixels_per_cell=(8,8) and cells_per_block=(2,2).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as the following steps:
1) Collect all vehicle and non-vehicle file in two lists name cars and notcars.
2) Define parameters as what I choosed above. 
3) Call extract_features function two times to genearate feature vector for both cars and notcars.
4) Compute the mean and std to be used for later scaling by calling function StandardScaler().fit().
5) Call the scaler to create in step 4) to normalize the features.
6) Call train_test_split to shuffle and split the data set into training and test data.
7) Create a LinearSVC() class and call fit function to train the classifier.
8) Call predict with test data to check the accuracy.
Done

``` python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time
import pickle

#Normalize the feature value
features_scaler = StandardScaler().fit(all_features)

# Apply the scaler to features
scaled_features = features_scaler.transform(all_features)

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, all_labels, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
   
```

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I am using the Udacity HoG sub-sampling window search. In the implementation,  the searching area of the image is defined by ((ystart, ystop), (xstart, xstop)). Then the searching image is converted to the color space for HoG features and binned spatial and color histogram features. The hog features is got for the whole searching image just once, but only the hog features in search window will be inputed into classifier.

There is a trick with scale. The searching window is always 64*64, but when scale is bigger than 1, the image will be shrank and it has the same effect of enlarging the searching window.

```python
# Define a single function that can extract features using hog sub-sampling, bin spatital and color histogram and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    img_tosearch = img[ystart:ystop,xstart:xstop, :]
    
    # convert RGB to YCrCb
    ctrans_tosearch = convert_color(img_tosearch, cspace=hog_cspace)
    hist_tosearch = convert_color(img_tosearch, cspace=hist_cspace)
    
    # resize the image
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        hist_tosearch = cv2.resize(hist_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    #get three channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above for the whole searching image
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = hist_tosearch[ytop:ytop+window, xleft:xleft+window]
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features,spatial_features, 
                                                          hist_features)).reshape(1, -1))    
            
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox_list.append(((xbox_left+xstart, ytop_draw + ystart), \
                                  (xbox_left+win_draw+xstart, ytop_draw+win_draw + ystart)))
                
    return bbox_list

```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. The binned color and histograms is generated in a different color space HSV. Here are some example images:

![alt text][image51]
![alt text][image52]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

The bounding boxes, heatmap and labed result is already shown in the window search part.

``` python
def pipe_line(image):
    ystart = 400
    ystop = 600
    
    xstart = 600 
    xstop = image.shape[1]
    
    # use different scale to define the search window
    scales = [1, 1.5, 2]
    
    bboxes_all = []
    for scale in scales:
        bbox_list = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, 
                                      X_scaler, orient, pix_per_cell, cell_per_block)
        bboxes_all += bbox_list

    bboxes_nframes.append(bboxes_all)
    if len(bboxes_nframes) > nframes_th:
        bboxes_nframes.pop(0)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, sum(bboxes_nframes, []))
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 10)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image, labels)
    return draw_img
```
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Sometimes the surface of road is identified as car. One way to fix this is to use more car and not car database to train the svm classifier, or consider other method like SSD, YOLO.

## Advanced Lane Lines Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/camera_calibration.png "Camera Calibration"
[image2]: ./examples/undistorted1.png "Undistorted"
[image3]: ./examples/undistorted2.png "Undistorted"
[image4]: ./examples/yellow_mask.png "Yellow Mask"
[image5]: ./examples/white_mask.png "White Mask"
[image6]: ./examples/combined_mask.png "Combined Mask"
[image7]: ./examples/combined_region_mask.png "Combined Mask with Region Mask"
[image8]: ./examples/warped1.png "Warp Calibration 1"
[image9]: ./examples/warped2.png "Warp Calibration 2"
[image10]: ./examples/warped_mask.png "Warped Mask"
[image11]: ./examples/histogram.png "Histogram"
[image12]: ./examples/sliding_windows.png "Sliding Windows"
[image13]: ./examples/highlighted_lane.png "Highlighted Lane"
[image14]: ./examples/calibration2.png "Camera Calibration"
[video1]: ./test_video_output/project_video_output_pipeline1.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first four code cells of the IPython notebook located in "./advanced-lane-finding-sandbox1.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image14]

### Pipeline (single images)

My pipeline is in my `pipeline(image)` method.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I first tried a bunch of different tactics, but I eventually landed on a very simple method which definitely overfits on our first project example.  However, I wanted to start with something that was simple and worked.  Later, if I had time, try to come up more robust ways of detecting lanes.

First, I detected yellow lines by converting the image to a HLS image and extracting the yellow portions of the image.  I determined this range by using some guides on the internet regarding HLS.  I chose HLS because of its robustness to different lighting conditions.  After some careful tinkering, I decided on yellow ranges between (20,25,100) and (25, 225, 255) using the cv2.inRange method.

```python
yellowmask = cv2.inRange(hls, np.array([20,25,100]), np.array([25, 225, 255]))
```

![alt text][image4]

Next, I detected white lines by using the converted HSL image and extracting white portions from the image.  This was much easier to determine compared to the yellow portion since white is mostly the lightness in the HLS color space.  My HLS values chosen were between (0, 200, 0) and (255,255,255).

```python
whitemask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255,255,255]))
```

![alt text][image5]

After extracting yellow and white channels, I combined them to create a single combined binary mask.

```python
combined = np.zeros_like(whitemask)
combined[((whitemask == 255) | (yellowmask == 255))] = 255
```
 
![alt text][image6]

This binary mask was then filtered via a region mask which was carefully chosen to select the bottom portion of the image.

```python
vertices = np.array([[(0,undistorted.shape[0]),(550, 300), (730, 300), (undistorted.shape[1],undistorted.shape[0])]], dtype=np.int32)
image_region = region_of_interest(combined, vertices)
```

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the 4th code block of my jupyter notebook `advanced-lane-finding.ipynb`.  The `warp()` function takes as input an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32(
        [[200,img_size[1]],
         [575,450],
         [750,450],
         [1200,img_size[1]]])
    
    dst = np.float32(
        [[320,img_size[1]],
         [210,0],
         [1300,0],
         [970,img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720        | 
| 575, 450      | 210, 0      |
| 750, 450     | 1300, 0      |
| 1200, 720      | 970, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane pixels, I first applied undistortion and filters to extract the lane pixels from the image. Then, I used a perspective transform `warp()` to create a birds-eye view of the lane.

![alt text][image10]

From here, I then utilized the idea of histograms along the x axis to determine where in the picture a lane exists.

![alt text][image11]

Taking the histogram idea further, I implemented the sliding windows technique `find_lane_pixels()` and fitted a polynomial `fit_polynomial()` to the lines detected.  In assumed that the left lane would always be in the left side of the image and, likewise, for the right lane.  Although this isn't the most robust strategy, it worked for the project video provided.

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In order to calculate the curvature, I implemented a method called `measure_curvature()`.  This method took the left and right fitted polynomials and calculated their curvature in real world space via a pixel to meters convertion.

The offest was calculated in a method called `measure_offset()`.  Once again, I used the fitted polynomicals to get the x-coordinates for each.  After doing so, I was able to get the mid of these two points and compare this middle (lane middle) with the middle of the entire image (the car's center) giving me the offset.  `measure_offset()` also converts pixel space to real world meter space.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented a method called `highlight_lane()` in the python notebook which took the fitted polynomials, filled them in, and then used `Minv` from my `warp()` method to perform an inverse perspective transform from the warped image back to the original undistorted image.  Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

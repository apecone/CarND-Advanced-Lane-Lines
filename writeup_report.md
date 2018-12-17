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
[image7]: ./examples/warped1.png "Warp Calibration 1"
[image7]: ./examples/warped2.png "Warp Calibration 2"
[image8]: ./examples/warped_mask.png "Warped Mask"
[image9]: ./examples/histogram.png "Histogram"
[image10]: ./examples/sliding_windows.png "Sliding Windows"
[image11]: ./examples/highlighted_lane.png "Highlighted Lane"
[image11]: ./examples/calibration2.png "Camera Calibration"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first four code cells of the IPython notebook located in "./advanced-lane-finding-sandbox1.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image11]

### Pipeline (single images)

My pipeline is in my `pipeline(image)` method.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I first tried a bunch of different tactics, but I eventually landed on a very simple method which definitely overfits on our first project example.  However, I wanted to start with something that was simple and worked.  Later, if I had time, try to come up more robust ways of detecting lanes.

First, I detected yellow lines by converting the image to a HLS image and extracting the yellow portions of the image.  I determined this range by using some guides on the internet regarding HLS.  I chose HLS because of its robustness to different lighting conditions.  After some careful tinkering, I decided on yellow ranges between (20,25,100) and (25, 225, 255) using the cv2.inRange method.
`yellowmask = cv2.inRange(hls, np.array([20,25,100]), np.array([25, 225, 255]))`
![alt text][image4]

Next, I detected white lines by using the converted HSL image and extracting white portions from the image.  This was much easier to determine compared to the yellow portion since white is mostly the lightness in the HLS color space.  My HLS values chosen were between (0, 200, 0) and (255,255,255).
`whitemask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255,255,255]))`
![alt text][image5]

After extracting yellow and white channels, I combined them to create a single combined binary mask.
`combined = np.zeros_like(whitemask)
 combined[((whitemask == 255) | (yellowmask == 255))] = 255`
![alt text][image6]

This binary mask was then filtered via a region mask which was carefully chosen to select the bottom portion of the image.
![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

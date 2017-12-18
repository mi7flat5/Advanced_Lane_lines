
# Advanced Lane Finding Project

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

[image1]: ./output_images/camera_undist.jpg "Undistorted"
[image2]: ./output_images/undistorted.jpg  "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warp_perspective.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/final.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[image7]: ./output_images/combined_thresh.jpg "color select"
[image8]: ./output_images/soebel_threshY.jpg "Video"
[image9]: ./output_images/soebel_threshX.jpg "Video"
[image10]: ./output_images/mag_thresh.jpg "Video"

[image11]: ./output_images/dir_thresh.jpg "color select"
[image12]: ./output_images/hls_thresh.jpg "Video"
[image13]: ./output_images/color_select_thresh.jpg "Video"
[image14]: ./output_images/combined.jpg "Video"
[image15]: ./output_images/masked.jpg "Video"
[image16]: ./output_images/hist_warped.jpg "Video"
[image17]: ./output_images/sliding_window.jpg "Video"
[image18]: ./output_images/hist_warped.jpg "Video"
[image19]: ./output_images/hist_warped.jpg "Video"
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
## For single image results follow along in Perception_Pipeline.ipynb. The video creation code is added into Pipeline.py, with line objects to manage persistent state between frames, along with tests for erroneous radius and polynomial fit calculations. 
### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "Perception_Pipeline.ipynb" 

The idea here is to take an object for wheich we know the dimension, and find the difference betweeen what is in the image and what the dimensions should be. Most of the heavy lifitng is done by OpenCV. all that is needed is to supply an image of an obkect for which we know the dimensions, commonly a chessboard. The image is searche for corners based on object points supplied to the   cv2.findChessboardCorners function. This will return the points on the image where the corners are detected. Then the object point(3d real dimension) and image points( where the object points are in the image) are  passed into cv2.calibrateCamera which returns a transformation matrix, and distortion coefficients.  These items along with the origional image are passed into  cv2.undistort where they are applied to the image and returns an image that the distortion has been corrected in. 

Here is the result:
![alt text][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.
Here's an image that is undistorted in cv2.undistort with the distortion coefficents and transformation matrix used in camera calibration. 

![alt text][image2]

#### 2. Color and Gradient Thresholding

I used a combination of color and gradient thresholds to generate a binary image. I first created a color thresholded binary image that selects, yellows and whites to isolate the lane lines of each color., I then combined this with gradient edge detection image, and finally combined this result with saturation threshold image. 
### Soebel X and Y 
![alt text][image8]
![alt text][image9]
### Color Selected
![alt text][image13]
### Magnitude Selection
![alt text][image10]
### Direction Selection
![alt text][image11]
### Combined Color Selection and Soebel X&Y and Magnitude, Direction selection. 
![alt text][image7]
### Saturation Selection 
![alt text][image12]
### Combination of all Filters
![alt text][image14]

At this point in video rendering a mask is applied to the binary's to remove irellevent pixels
![alt text][image15]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective()`, which appears in lines 33 through 38 in the file `Pipeline.py`  The `perception()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
     [((img_size[0] / 6) - 10), img_size[1]],
     [(img_size[0] * 5 / 6) + 60, img_size[1]],
     [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])

dst = np.float32(
    [[(img_size[0] / 4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1066, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

The final image of the selction  pipeline looks like this:

![alt text][image16]
#### 4. Line object
The methods for identifying and fitting pixels to a polynomial curve can be found in the notebook under

In order to find the pixels that will be used to fit a polynomial to, an informed beginning point is necessary. To do this I created a histogram of the warped output from the selection pipeline, and took the X value's for which the most Y values were above 0. One such maximum vlaue is searche for both left and righ of the midpoint of the image. After this a sliding widown search is performed in which the image is iteratively searched withing a small margin around the X values found from the histogram. Once this has been done, indecies for good pixels are saved, and then used to fit a polynomial curve to. the output of this process looks like this: 


![alt text][image17]

#### 5. Calculating Curvature
In Pileline.py the curvature of radius is calculated Line object method call test_fit. 
The curvature is calculated so that we can determine the robustness of the fit, I calculated the curve in meters. and then applied a check to see if it is a reasonable radius. if it is, the pipeline is not reset to continually do a window search, but if it fails it will throw out the new fit and use the old one, and cause a new window search. The code for the calculation looks like this:
```python
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Calculate the radius of curve in meters
        fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
```

This code can be found in lines 257 - 259 in Pipeline.py

#### 6. The final result of single image

![alt text][image6]

---

### Pipeline (video)

#### Here is the result for the project video:

<<<<<<< HEAD
Here's the [project video result](./output_images/Final2.mp4)
 [Youtube](https://youtu.be/DOgUgcw_Q_g)
 
=======
Here's the [project video result](./output_images/Fianl1.mp4)<br>
 [Project video Youtube link](https://youtu.be/2EG7aCboAAw)
---
--- 
>>>>>>> f76f9f16c854b4d90d4c0ea427445cbdc7cb054b
#### Here is the result for the challenge video:

Here's the [challenge video result](./output_images/challenge1.mp4)<br>
 [Challenge video Youtube link](https://youtu.be/-Rnfal1ogkw)
---

### Discussion

My pipeline is currently highely dependant on the lighting conditions, I think it may genralize well in some cases but would need some other system to scale thresholds based on ambient light. My biggest challenge for the project was trying to set up thresholds that worked for each video. The second big challenge was coming up with good robustness checks, and they seem to change depending on lighting and road conditions as well. I think this could be a useful pipeline for finding the lanes in conjunction with input from other sensors like ambient light detection. It may also be advantageous to combine this with some sort of machine learning models that are trained not only on image classification but perhaps they could be trained to manage threshold settings and the settings for robustness checks. 

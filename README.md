# **Self-Driving Car Engineer** Nanodegree
## Project: Finding Lane Lines on the Road 

This project is for Udacity CarND P2: Advanced lane lines. You can run the code locally. 

## Writeup

The goals / steps of this project are the following:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
1. Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
1. Apply a perspective transform to rectify binary image ("birds-eye view").
1. Detect lane pixels and fit to find the lane boundary.
1. Determine the curvature of the lane and vehicle position with respect to center.
1. Warp the detected lane boundaries back onto the original image.
1. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle
position

## Reflection
## 1. Camera Calibration and image undistortion

I coded this part in the [jupyter notebook](P2.ipynb). The images for calibration were shot by the camera in different distence or angels of the same printed chessboard. I counted the corners in the chessboard images. The the corners were detected in cv2.findChessboardCorners, and the coordinates of the corners were stored in a list. With the corresponding point coordinates in the object points which I listed and the corners' coordinates in images, the openCV calibrateCamera function implement the calibration. Then I get the mtx and dist array which were used to undistort images.

<img src="output_images/calibration and undistort.png">

The picture show the images before and after the undistortion, and the diference between them. I can see that the farther the pixel is away from the center, the worse the distortion.

## 2. Preprocessing
This step is the core of the project. The main idea is to seprate the lane lines. I have tried to check the difference between every channel of the image in RGB, HLS and HSV color space. I found that the R,S,V channel performed well in seprating the lane lines. I run my color_channel_threshld function to convert the image into binary image. The desired binary images were bright in lane lines and dark in the background. 
Here are some of the results through my code.

<img src="output_images/color channel and thresholds.png">

The [Preprocessing.py](Preprocessing.py) contains a pipline to get warped binary image from the original image. 

## 3. Warp image
Perspective transformation were needed to calculate the distance or lane curvature because of the perspective phenomenon. The transform need 4 pixels' coordinates in the source and destination images. I chose them manually form the image with straight lane lines. Then I got the images in "birds-eye view".
Here are one example of the transform.

<img src="output_images/warped image.png">

## 4.Find and fit lane lines
I wrote the lane detecting code in the Lane class which were written in [LaneLines.py](LaneLines.py). I use the sliding windows method to detect the lane line pixels coordinates and fit them into left and right lane polynomial function. 
Because of the lightness variation and some other reason, I couldn't detect the correct lane lines. Since the lane line doesn't change too much from frame to frame. The bad lane lines detecting result should be reject. In my code, if the positions of the near ends of lane lines, or the lane area changed drasticlly, the detecting result would be rejected.
With the lane line functions, the radius of the lane line curvature could be calcalated. 

## 5. Test on iamges and videos
I run my pipline on the test images firstly. The result looks good. Here is an example.
<img src="output_images/7.jpg">

The test result on videos were saved in the outputs video folder. It performed reasonably well on the entire [project video](output_videos/project_video_out.mp4). There is no catastrophic failures. And the radius of curvature of the lane and vehicle position within the lane were calculated.

## 6. Discuss
The pipline run through well on the test images and the project video. But it didn't performed properly on the two challenges videos.So the pipline were still not robust enough to satisfy all the scenarios. That's the major shortcommings of the pipline. Besides, it can also be imporved in time efficient.

There are methods to tackle the problems. More accurate thresholds and binary combination could be found if I implement the pipline through the tricky frames of the challenage video. 



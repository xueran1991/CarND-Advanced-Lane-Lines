# **Self-Driving Car Engineer** Nanodegree
## Project: Finding Lane Lines on the Road 
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

I coded this part in the jupyter notebook. The images for calibration were shot by the camera in different distence or angels of the same printed chessboard.  I counted the corners in the chessboard images. The the corners were detected in cv2.findChessboardCorners, and the coordinates of the corners were stored in a list. With the corresponding point coordinates in the object points which I listed and the corners' coordinates in images, the openCV calibrateCamera function implement the calibration. Then I get the mtx and dist array which were used to undistort images.
![image](855388ACF529435AA0DC35CEAACF7270)
The picture show the images before and after the undistortion, and the diference between them. I can see that the farther the pixel is away from the center, the worse the distortion.

## 2. Preprocessing
This step is the core of the project. The main idea is to seprate the lane lines. I have tried to check the difference between every channel of the image in RGB, HLS and HSV color space. I found that the R,S,V channel performed well in seprating the lane lines. I run my color_channel_threshld function to convert the image into binary image. The desired binary images were bright in lane lines and dark in the background. 
Here are some of the results therough my code.

![image](A105B2DBEE3D467389DB95D0E057EF10)

I wrote the image preprocessing and lane detecting code in two python filse: Preprocessing.py and LaneLines.py. The Preprocessing.py contains a pipline to get warped binary image from the original image. 

## 3. Warp image
Perspective transformation were needed to calculate the distance or lane curvature because of the perspective phenomenon. The transform need 4 pixels' coordinates in the source and destination images. I chose them manually form the image with straight lane lines. Then I got the images in "birds-eye view".
Here are one example of the transform.

![image](9745941C0A504834A10EFF5C230A6F6F)

## 4.Find and fit lane lines
I wrote the lane detecting code in the Lane class which were written in LaneLines.py. I use the sliding windows method to detect the lane line pixels coordinates and fit them into left and right lane polynomial function. 
Because of the lightness variation and some other reason, I couldn't detect the correct lane lines. Since the lane line doesn't change too much from frame to frame. The bad lane lines detecting result should be reject. In my code, if the positions of the near ends of lane lines, or the lane area changed drasticlly, the detecting result would be rejected.
With the lane line functions, the radius of the lane line curvature could be calcalated. 

## 5. Test on iamges and videos
I run my pipline on the test images firstly. The result looks good. Here is an example.
![image](08D38BA9A635456EA232658505B2D528)

The test result on videos were saved in the outputs video folder. It performed reasonably well on the entire project video. There is no catastrophic failures. And the radius of curvature of the lane and vehicle position within the lane were calculated.


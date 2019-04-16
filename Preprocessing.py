import cv2
import numpy as np
import matplotlib.image as mpimg

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def abs_sobel_thresh(img, orient='x', threshold=(100,200)):
## img in grayscale
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    sbinary[(scaled_sobel>threshold[0]) & (scaled_sobel<threshold[1])] = 1
    #binary_output = np.copy(img) # Remove this line
    return sbinary

def mag_thresh(img, sobel_kernel=3, threshold=(0, 255)):
## img in grayscale    
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude 
    sm = np.sqrt(sx**2 + sy**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_magnitude = np.uint8(255*sm / np.max(sm))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_magnitude)
    
    # 6) Return this mask as your binary_output image
    binary_output[(scaled_magnitude>threshold[0]) & (scaled_magnitude<threshold[1])] = 1
    return binary_output
    
def dir_threshold(img, sobel_kernel=3, threshold=(0, np.pi/2)):
    ## img in grayscale 
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1
    
    # Return the binary image
    return binary_output

# def color_threshold(img, threshold=(100,200)):
#     ## img in RGB colormap
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     s = hls[:,:,2]
#     s_binary = np.zeros_like(s)
#     s_binary[(s >= threshold[0]) & (s <= threshold[1])] = 1
    
#     return s_binary

def color_channel_threshold(channel, threshold=(100,200)):
    ## img in RGB colormap

    channel_binary = np.zeros_like(channel)
    channel_binary[(channel >= threshold[0]) & (channel <= threshold[1])] = 1
    
    return channel_binary

def preprocessing_image(image, mtx, dist, src, dst):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(undist, cv2.COLOR_RGB2HSV)
    s = hls[:,:,2]
    r = undist[:,:,0]
    v = hsv[:,:,2]
    
    bottom_v_mean = int(np.mean(v[v.shape[0]//3:,:]))
    
    gradx_binary = abs_sobel_thresh(s, 'x', (25,100))
    s_binary = color_channel_threshold(s, (130, 255))
    r_binary = color_channel_threshold(r, (210, 255))
    v_binary = color_channel_threshold(v, (200, 255))
    
    #------------
    s_ok = np.sum(s_binary[360:,:]) < 30000
    grad_ok = np.sum(gradx_binary[360:,:]) < 30000
    r_ok = np.sum(r_binary[360:,:]) < 30000
    v_ok = np.sum(v_binary[360:,:]) < 30000
    
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary*s_ok==1) | (gradx_binary*grad_ok==1) | (r_binary*r_ok==1) | (v_binary*v_ok==1)] = 1
    warped_binary = warper(combined_binary, src, dst)
    
    return warped_binary
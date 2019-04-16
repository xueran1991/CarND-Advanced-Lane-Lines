from Preprocessing import warper
import numpy as np
import cv2



class Line():
    def __init__(self, debug = 1, showProcess = 1, iter_num = 10):
        
        # left / right lane detected or not
        self.detected = [False, False]                         
               
        #average x values of the fitted line over the last n iterations
        self.bestx = None                           
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None                           
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]        
        self.recent_fit = [np.array([False])]         
                
        #radius of curvature of the line in some units
        self.radius_of_curvature = None                
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0                     
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')  
        
        #x values for detected line pixels
        self.allx = None                               
        
        #y values for detected line pixels
        self.ally = None                               
        
        self.image_width = 0
        self.image_height = 0
        
        self.iter_num = iter_num
        self.frame_num = 0
        self.current_area = 0
                
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700
        
        # 中间图像
        self.processing_img = 0
        self.debug_container = []
        
    def sliding_windows(self, binary_warped, nwindows = 9, margin = 50, minpix = 50):
        ## Finding base points
        self.image_width = binary_warped.shape[1]
        self.image_height = binary_warped.shape[0]
        
        hist = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int(hist.shape[0]//2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint
        
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        self.processing_img = np.dstack((binary_warped, binary_warped, binary_warped))

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(self.processing_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(self.processing_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
#                 left_lane_inds.append(good_left_inds)
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
#                 right_lane_inds.append(good_right_inds)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            
            self.detected = False
            
            return 
            # Avoids an error if the above is not implemented fully

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        self.allx = [leftx, rightx]
        self.ally = [lefty, righty]
        
        self.processing_img[lefty, leftx, :] = [0,0,255]
        self.processing_img[righty, rightx,:] = [0,0,255]
        

        
    def fit_polynomial(self):
        # Fit a second order polynomial to each using `np.polyfit`

        try:
            left_fit = np.polyfit(self.ally[0], self.allx[0], 2)
            right_fit = np.polyfit(self.ally[1], self.allx[1], 2)
        # if failed to fit a polynomial
        except TypeError:
            
            self.detected = False
            self.current_fit = [np.array([0,0,0]), np.array([0,0,0])]
            self.frame_num += 1
            return
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.image_width-1, self.image_width)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        self.current_xfitted = [left_fitx,right_fitx]
        # refresh the detected flag and the coefficients
        self.detected = True
        # calc the difference between the current fit and last fit
        if self.frame_num > 0:
            self.diffs = [left_fit-self.current_fit[0], right_fit-self.current_fit[1]]
        self.current_fit = [left_fit, right_fit]
            
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
#         cv2.polylines(self.processing_img, np.int_([pts_left]), True, [255,0,0], thickness=3)
#         cv2.polylines(self.processing_img, np.int_([pts_right]), True, [255,0,0], thickness=3)
        self.frame_num += 1
        
        
    def polynomial(self, fit):
        y = np.linspace(0, self.image_width-1, self.image_width)
        x = fit[0]*y**2 + fit[1]*y + fit[2]
        return x, y
    
    def search_around_poly(self, binary_warped, margin = 100):

        # Grab activated pixels in new binary_warped image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Determine the pixels which are in the fit and margin area
        left_fit = self.current_fit[0]
        right_fit = self.current_fit[1]
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                          & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                           & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
                
        if np.sum(left_lane_inds==right_lane_inds) > 0:
            self.detected = False
            self.sliding_windows(binary_warped)
            self.fit_polynomial()
            return
        
        self.allx = [leftx, rightx]
        self.ally = [lefty, righty]

        # Fit new polynomials        
        self.fit_polynomial()
        if self.detected == False:
            return
        left_fitx, ploty = self.polynomial(self.current_fit[0])
        right_fitx, _ = self.polynomial(self.current_fit[1])
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        self.processing_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    def check_lane(self):
        ### Some methods to distinguish the current lane line is valid or not
        #    1. The near ends of the lane lines or the line_base_pos dont change too much from 
        #       frame to frame.
        #    2. The pixel area of the lane region dont change dramaticly from frame to frame
        #    3. The lane line shape dont change too much too between neighbooor frames
        self.current_xfitted.append(self.current_fit)
        
        abs_diffs = np.abs(self.diffs)        
        left_fitx, ploty = self.polynomial(self.current_fit[0])
        right_fitx, _ = self.polynomial(self.current_fit[1])
        
        # Calc the car position respect to the center of the lane
        left_lane_xcor = left_fitx[self.image_height]
        right_lane_xcor = right_fitx[self.image_height]        
        current_line_base_pos = ((right_lane_xcor - left_lane_xcor)/2 + left_lane_xcor - self.image_width/2) * self.xm_per_pix
        
        if self.best_fit == None:
            self.best_fit = self.current_fit
        if self.frame_num > 1:
            lane_pos_ok = np.abs(current_line_base_pos - self.line_base_pos) < 0.20
            lane_area_ok = np.abs((np.sum(right_fitx-left_fitx)-self.current_area)/self.current_area) < 0.20
            self.debug_container.append((np.abs(current_line_base_pos - self.line_base_pos), 
                 np.abs((np.sum(right_fitx-left_fitx)-self.current_area)/self.current_area)))
        else:
            lane_pos_ok = True
            lane_area_ok = True
        
        if lane_pos_ok and lane_area_ok:
            self.best_fit = self.current_fit
        
        # Calc the pixel area of the lane
        self.current_area = np.sum(right_fitx-left_fitx)
        self.line_base_pos = current_line_base_pos
        
        left_fit = self.best_fit[0]
        right_fit = self.best_fit[1]
        
        y_eval = np.max(ploty)
        
        # meters per pixel in y / x dimension
        my = self.ym_per_pix 
        mx = self.xm_per_pix 
        
        real_left_fit = np.array([mx/my**2*left_fit[0], mx/my*left_fit[1], left_fit[2]])
        real_right_fit = np.array([mx/my**2*right_fit[0], mx/my*right_fit[1], right_fit[2]])
        radius_of_left_curvature = ((1 + (2*real_left_fit[0]*y_eval* + real_left_fit[1])**2)**1.5) / np.absolute(2*real_left_fit[0])
        radius_of_right_curvature = ((1 + (2*real_right_fit[0]*y_eval* + real_right_fit[1])**2)**1.5) / np.absolute(2*real_right_fit[0])
        self.radius_of_curvature = np.mean([radius_of_left_curvature, radius_of_right_curvature])
        
    def output(self, original_image, src, dst):
        self.check_lane()
        lane_line = np.zeros([self.image_height, self.image_width, 3], dtype=np.uint8)
        lane_region = np.zeros_like(lane_line)
        lane_line[self.ally[0], self.allx[0],:] = [255, 0, 0]
        lane_line[self.ally[1], self.allx[1],:] = [255, 0, 0]  
        
        unwarped_lane_line = warper(lane_line, dst, src)
        
        x = np.linspace(0,self.image_width-1, self.image_width)
        x = np.tile(x, (self.image_height, 1))
        y = np.linspace(0,self.image_height-1,self.image_height)
        y = np.tile(y,(self.image_width,1))
        y = y.T
            
        left_fit = self.best_fit[0]
        right_fit = self.best_fit[1]
            
        lane_region[(x > (left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]))
                    & (x < (right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]))] = [0,255,0]
            
        lane = cv2.addWeighted(lane_region, 0.5, lane_line, 1, 0)
            
        unwarped_lane = warper(lane, dst, src)
            
        ## show process img
        resized_processing = cv2.resize(self.processing_img, (self.image_width //3, self.image_height //3))
            
        out_frame = cv2.addWeighted(unwarped_lane, 0.8, original_image, 1, 0)
        out_frame[:self.image_height //3, :self.image_width //3,:] = resized_processing
        # add the radius of curvature text
        cv2.putText(out_frame, 'Radius of curvature = '+str(int(self.radius_of_curvature))+'m', 
                    (450,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        # add the car position offset text       
        cv2.putText(out_frame, 'Vehicle is '+str(int(self.line_base_pos*100))+'cm right of center', 
                    (450,180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        # add the frame text       
        cv2.putText(out_frame, 'frame:'+str(self.frame_num), 
                    (450,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        return out_frame

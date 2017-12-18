import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Load distortion correction coefficents from pickle -- created in notebook
dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


img_shape3 = mpimg.imread('test_images/90.jpg').shape
img_size = (img_shape3[1], img_shape3[0])

# source and destination coridinates for warp perspective
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
imshape = img_shape3
vertices = np.array([[(100, imshape[0]),
                          (imshape[1] / 2 + 30, imshape[0] / 2 + 30),
                          (imshape[1] / 2 + 30, imshape[0] / 2 + 30),
                          (imshape[1] - 100, imshape[0])]], dtype=np.int32)
vertices2 = np.array([[(50, imshape[0]),
                          (0, 0),
                          (imshape[1]-100, 0),
                          (imshape[1]-950, imshape[0])]], dtype=np.int32)

# Inverse warp matrix, returns warp to regular perspective
Minv = cv2.getPerspectiveTransform(dst, src)

def perspective(img, src, dst, mtx, dist):
    img = np.copy(img)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

# COLOR SELECTION - Yellow and White
def c_f(img, rgb_thresh=(255, 255, 90)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def b_f(img, rgb_thresh=(150, 100, 20)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select & c_f(img)

def color_thresh(img, rgb_thresh=(215, 215, 215)):
    # Create an array of zeros same xy size as img, but single channel
    img = region_of_interest(img,vertices)
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return  color_select | b_f(img)

# Combine color selection and edge detection
def edges(im):
    # im = region_of_interest(im)
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    gradx = abs_sobel_thresh(im, orient='x', sobel_kernel=ksize, thresh=(100, 155))
    grady = abs_sobel_thresh(im, orient='y', sobel_kernel=ksize, thresh=(100,152))
    mag_binary = mag_thresh(im, sobel_kernel=ksize, mag_thresh=(120,200))
    dir_binary = dir_threshold(im, sobel_kernel=ksize, thresh=(.7, 1.3))

    combined = np.zeros_like(dir_binary).astype(np.uint8)
    thing = color_thresh(im).astype(np.uint8)

    mask = ((gradx == 1)|(grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))

    combined[mask] = 1
    combined = region_of_interest(combined,vertices).astype(np.uint8)
    return np.logical_or(combined , thing).astype(np.float32)


# Saturation selection
def hls_select(img, thresh=(90, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result

    img = region_of_interest(region_of_interest(img,vertices),vertices2)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 0]

    sthresh_min = 20
    sthresh_max = 30
    s_output = np.zeros_like(s_channel).astype(np.uint8)
    s_output[(s_channel > sthresh_min) & (s_channel <= sthresh_max)] = 1
    return s_output


# Image Masking
def region_of_interest(img,verts):
    imshape = img.shape
    vertices = verts
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

class Line():
    def __init__(self, side='left', ):
        self.side = side
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        ## NEED OUTSIDE
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = [np.array([0.0,0.0,0.0])]
        self.best_fit_avg = 0
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = [[0]]
        # distance in meters of vehicle center from the line
        if self.side == 'left':
            self.line_base_pos = 350
        else:
            self.line_base_pos = 950
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # RIGHT LINE NEEDS MIDPOINT SET
        self.midpoint = 0
        self.last_fit = None
        ## NEED OUTSIDE
        self.ploty = None
        self.avg_pool = 2
        self.maxrad =0
        self.minrad =0
        self.avgrad =0
        self.other_fit = None
        self.other_fit = None

    def test_fit(self, fit,ploty):
        y_eval = np.max(ploty)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Calculate the radius of curve in meters
        fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        self.detected = True

        # udate metrics for curve radius
        if curverad < 3000:
            self.radius_of_curvature.append(curverad)
            self.minrad = np.min(self.radius_of_curvature)
            self.maxrad = np.max(self.radius_of_curvature)
            self.radius_of_curvature.append(curverad)
            self.avgrad = np.mean(self.radius_of_curvature)

        # First few frames self.current_fit will be None
        if len(self.current_fit) ==1:
            self.best_fit = np.mean(fit)
            self.last_fit = fit

        # Tests for fits that are not similar to current best fit.
        if np.abs(np.mean(self.best_fit)- np.mean(fit))>150:
            self.detected = False
        if np.mean(fit) <0 :
            self.detected = False

        # Tests curve radius for erroneous results
        if curverad > 7000:
            self.detected = False
        if curverad < 200:
            self.detected = False

        # only update with new fit if all tests passed
        # otherwise use last good fit as update value
        if self.detected:
            self.update_fit(fit)
        else:
            self.update_fit(self.last_fit)

        # Update pixel indices
        self.ploty = ploty
        fitx = self.best_fit[0] * ploty ** 2 + self.best_fit[1] * ploty + self.best_fit[2]
        self.update_xfitted(fitx)

        return True

    def update_xfitted(self, newx):

        # keep a running average of known good indices
        self.recent_xfitted.insert(0,[newx])
        if len(self.recent_xfitted)> self.avg_pool:
            self.bestx = np.mean(self.recent_xfitted, axis=0)
            self.recent_xfitted.pop()
        else:
            self.bestx = newx



    def update_fit(self, fit):

        # Keep running average of known good fits
        self.last_fit = fit
        self.current_fit.insert(0,fit)
        if len(self.current_fit)> self.avg_pool:
            self.best_fit = np.mean(self.current_fit, axis=0)
            self.current_fit.pop()
        else:
            self.best_fit = fit
        self.best_fit = fit

    def process_line(self, bin_img):

        nonzero = bin_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100
        if self.detected:
            left_lane_inds = ((nonzerox > (self.best_fit[0] * (nonzeroy ** 2) + self.best_fit[1] * nonzeroy +
                                           self.best_fit[2] - margin)) & (nonzerox < (self.best_fit[0] * (nonzeroy ** 2) +
                                           self.best_fit[1] * nonzeroy + self.best_fit[2] + margin)))


            self.allx = nonzerox[left_lane_inds]
            self.ally = nonzeroy[left_lane_inds]
            # Test that there are enough pixels to fit polynomial
            if len(self.allx)<50:
                self.detected = False
            if len(self.ally)< 50:
                self.detected = False
            if self.detected:
                fit = np.polyfit(self.ally, self.allx, 2)
                # test fit, curve, set detected false and return if out of whack
                ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
                self.test_fit(fit, ploty)

        if not self.detected:
            histogram = np.sum(bin_img[bin_img.shape[0] // 2:, :], axis=0)
            histogram[:200] = 0
            histogram[500:900] = 0
            histogram[1100:] = 0

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            if self.side == 'right':
                self.midpoint = np.int(histogram.shape[0] / 2)
            self.line_base_pos = np.argmax(histogram[self.midpoint:]) + self.midpoint

            ##test position of line base, set appropriately if erroneous
            if self.side == 'left' and (self.line_base_pos > 500 or self.line_base_pos < 300):
                self.line_base_pos = 350
            if self.side == 'right' and (self.line_base_pos < 850):
                self.line_base_pos = 1000
            x_current = self.line_base_pos

            lane_inds = []

            minpix = 50
            margin = 100
            nwindows = 9
            # Step through the windows one by one
            window_height = np.int(bin_img.shape[0] / nwindows)

            # Search image for pixels for polynomial fit
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = bin_img.shape[0] - (window + 1) * window_height
                win_y_high = bin_img.shape[0] - window * window_height
                win_xleft_low = x_current - margin
                win_xleft_high = x_current + margin
                win_xright_low = x_current - margin
                win_xright_high = x_current + margin

                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

                # Append these indices to the lists
                lane_inds.append(good_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > minpix:
                    x_current = np.int(np.mean(nonzerox[good_inds]))

            lane_inds = np.concatenate(lane_inds)
            self.allx = nonzerox[lane_inds]
            self.ally = nonzeroy[lane_inds]
            # if len(self.allx)<10:
            #     return
            # if len(self.ally)< 10:
            #     return

            fit = np.polyfit(self.ally, self.allx, 2)
            ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
            return self.test_fit(fit, ploty)

left_line = Line('left')
right_line = Line('right')
frame_num = 0

def pipeline(img):
    global frame_num
    # Combination of gradient and color thresholding
    sxbinary = edges(img)

    # Threshold saturation channel from HLS color space
    s_binary = hls_select(img)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(sxbinary == 1)| (s_binary == 1)] = 1

    # Warp the thresholded binary image to top down view
    # then send to line for line update
    binary_warped = perspective(combined_binary, src, dst, mtx, dist)
    left_line.process_line(binary_warped)
    right_line.process_line(binary_warped)

    # Visual of histogram, for output image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # histogram[:200] = 0
    # histogram[500:800] = 0
    # histogram[1100:] = 0
    midpoint = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) +midpoint
    plt.plot(histogram)
    # Extrememly hacky, and slowest part of pipeline.
    # there is a better way, but not my focus at the momment
    plt.savefig('hist.jpg')
    plt.cla()
    histogram = mpimg.imread('hist.jpg')

    # create copy of warped image
    color_warp = np.array(cv2.merge((binary_warped * 25, binary_warped * 25, binary_warped * 255)), np.uint8)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, left_line.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, right_line.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Test that lane found, avoids pipeline crash
    if pts.all()==None:
        return img

    # Draw the lane onto the warped blank image
    # clean_color is for final video result
    # color_warp is for tool in output image that shows the lane line detection
    # with the lane drawn
    clean_color = np.zeros_like(color_warp)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 180, 0))
    cv2.fillPoly(clean_color, np.int_([pts]), (0, 180, 0))

    out = np.zeros_like(img)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(clean_color, Minv, (img.shape[1], img.shape[0]))

    # Add numerous information fields to output image
    cv2.putText(out, "Left Line: ", (20, 520),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
    cv2.putText(out, "Average Radius: " + str(left_line.avgrad), (20, 540),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)

    cv2.putText(out, "Current Radius: " + str(left_line.radius_of_curvature[-1]), (20, 560),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)

    cv2.putText(out, "Best Fit: " + str(np.mean(left_line.best_fit)), (20, 580),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
    cv2.putText(out, "Base Pos: " + str(left_line.line_base_pos), (20,600),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)


    cv2.putText(out, "Right Line: ", (20, 620),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
    cv2.putText(out, "Average Radius: " + str(right_line.avgrad), (20, 640),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)

    cv2.putText(out, "Current Radius: "  + str(right_line.radius_of_curvature[-1]), (20, 660),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
    cv2.putText(out, "Best Fit: " + str(np.mean(right_line.best_fit)), (20, 680),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
    cv2.putText(out, "Base Pos: " + str(right_line.line_base_pos), (20, 700),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)

    # Combine the result with the original imagecd /me
    cache_image = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)

    rs = cv2.resize(cache_image,(845,475))
    out[0:rs.shape[0],0:rs.shape[1]] = rs
    sx = cv2.resize(np.array(cv2.merge((sxbinary*255,sxbinary*255,sxbinary*255)),np.uint8),(435,238))
    sl = cv2.resize(np.array(cv2.merge((s_binary * 255, s_binary * 255, s_binary * 255)), np.uint8), (435, 238))
    bw = cv2.resize(color_warp,(435,238))
    hist = cv2.resize(histogram, (423, 238))

    out[0:sx.shape[0], rs.shape[1]:] = sx
    out[sx.shape[0]:sx.shape[0]*2, rs.shape[1]:] = sl
    out[2*sx.shape[0]:sx.shape[0] * 3, rs.shape[1]:] = bw
    out[2 * sx.shape[0]:sx.shape[0] * 3, int(rs.shape[1]/2):rs.shape[1]] = hist

    xm_per_pix = 3.7 / 700
    a = (midpoint -np.abs(right_base - left_base)) * xm_per_pix
    side = 'right'
    if a <0:
        a *= -1
        side = 'left'

    cv2.putText(out, "Shane Harmon, Project 4 SDCND", (60, 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 70), 1)

    cv2.putText(out, "Car: "+ str(round(a,3))+ "m "+side+" of center" , (60, 440),
                cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1)
    if left_line.detected:
        cv2.putText(out, "DETECTED", (160, 520),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0), 1)

    else:
        cv2.putText(out, "WINDOW SEARCH", (160, 520),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 0, 0), 1)
    if right_line.detected:
        cv2.putText(out, "DETECTED", (160, 620),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0), 1)
    else:
        cv2.putText(out, "WINDOW SEARCH", (160, 620),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 0, 0), 1)

    return out


output = 'output_images/chall.mp4'


clip1 = VideoFileClip("project_video.mp4")#.subclip(38,45)
white_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!
white_clip.write_videofile(output,audio=False, threads=8,verbose=True)
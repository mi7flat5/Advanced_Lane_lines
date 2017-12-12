import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


image = mpimg.imread('test_images/straight_lines1.jpg')
img_size = (image.shape[1], image.shape[0])
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
     [((img_size[0] / 6) - 10), img_size[1]],
     [(img_size[0] * 5 / 6) + 60, img_size[1]],
     [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])

# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result
# again, not exact, but close enough for our purposes
dst = np.float32(
    [[(img_size[0] / 4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])

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

def edges(im):
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    im = np.copy(im)
    gradx = abs_sobel_thresh(im, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(im, orient='y', sobel_kernel=ksize, thresh=(20, 255))
    mag_binary = mag_thresh(im, sobel_kernel=ksize, mag_thresh=(130,255))
    dir_binary = dir_threshold(im, sobel_kernel=ksize, thresh=(.8, 1.2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) | (grady == 1))|((mag_binary == 1) & (dir_binary == 1))] = 1
    #
    return combined


def hls_select(img, thresh=(90, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    img = np.copy(img)  # placeholder line
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    thresh_min = 170
    thresh_max = 255
    s_output = np.zeros_like(s_channel)
    s_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    l_channel = hsv[:, :, 1]

    thresh_min = 100
    thresh_max = 250
    l_output = np.zeros_like(l_channel)
    l_output[(l_channel > thresh_min) & (l_channel <= thresh_max)] = 1

    #return s_output
    return np.logical_and(l_output, s_output, dtype=np.float32)


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
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # RIGHT LINE NEEDS MIDPOINT SET
        self.midpoint = 0
        ## NEED OUTSIDE
        self.ploty = None
        self.avg_pool = 4

    def test_fit(self, fit,ploty):
        y_eval = np.max(ploty)




        curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])


        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        print(curverad)
        # if fail return false, set self.detected = False
        # if curverad < 10:
        #     self.detected = False
        #     return False

        self.detected = True
        self.update_fit(fit)
        # print(np.abs(np.mean(fit)))
        # print('.')
        # print(np.mean(self.best_fit))
        self.ploty = ploty
        fitx = self.best_fit[0] * ploty ** 2 + self.best_fit[1] * ploty + self.best_fit[2]
        self.update_xfitted(fitx)
        return True

    def update_xfitted(self, newx):

        self.recent_xfitted.insert(0,[newx])

        if len(self.recent_xfitted)> self.avg_pool:
            self.bestx = np.mean(self.recent_xfitted, axis=0)
            self.recent_xfitted.pop()
        else:
            self.bestx = newx


    def update_fit(self, fit):

        self.current_fit.insert(0,fit)

        if len(self.current_fit)> self.avg_pool-2:
            self.best_fit = np.mean(self.current_fit, axis=0)
            self.current_fit.pop()
        #else:
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

            fit = np.polyfit(self.ally, self.allx, 2)
            # test fit, curve, set detected false and return if out of whack
            ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
            self.test_fit(fit, ploty)

        elif not self.detected:
            histogram = np.sum(bin_img[bin_img.shape[0] // 2:, :], axis=0)
            histogram[:250] = 0
            histogram[500:900] = 0
            histogram[1100:] = 0

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            if self.side == 'right':
                self.midpoint = midpoint = np.int(histogram.shape[0] / 2)
            self.line_base_pos = np.argmax(histogram[self.midpoint:]) + self.midpoint
            x_current = self.line_base_pos


            lane_inds = []

            minpix = 50
            margin = 100
            nwindows = 9
            # Step through the windows one by one
            window_height = np.int(bin_img.shape[0] / nwindows)


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

            fit = np.polyfit(self.ally, self.allx, 2)
            ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
            return self.test_fit(fit, ploty)

left_line = Line('left')
right_line = Line('right')


def pipeline(img):
    # Gradient Threshold
    sxbinary = edges(img)

    # Threshold color channel
    s_binary = hls_select(img)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    binary_warped = perspective(combined_binary, src, dst, mtx, dist)
    left_line.process_line(binary_warped)
    right_line.process_line(binary_warped)


    color_warp = np.array(cv2.merge((binary_warped * 255, binary_warped * 255, binary_warped * 255)), np.uint8)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, left_line.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, right_line.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    cache_image = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)


    return cache_image


output = 'output_images/lane2.mp4'


clip1 = VideoFileClip("project_video.mp4")#.subclip(15,26)
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(output,audio=False, threads=8,verbose=True)
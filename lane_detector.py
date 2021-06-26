import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

from moviepy.editor import VideoFileClip
import logging
import warnings
from pathlib import Path
import tqdm
import pickle
from collections import deque

from IPython.display import HTML


def eval_poly(fit, p):
    return fit[0] * p ** 2 + fit[1] * p + fit[2]


def sample_poly(fit, y_range):
    ploty = np.linspace(0, y_range - 1, y_range)
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    return fitx, ploty


class Lane:
    def __init__(self, memory_size=3):
        # was the line detected in the last iteration?
        self.detected = False
        self.last_valid_frame_id = None
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=memory_size)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # radius of curvature of the line in some units
        self.best_radius_of_curvature = np.nan
        # radius of curvature of the line in some units
        self.current_radius_of_curvature = np.nan
        # distance in meters of vehicle center from the line
        self.best_lane_center_distance = np.nan
        # distance in meters of vehicle center from the line
        self.current_lane_center_distance = np.nan
        # distance in meters of vehicle center from the line
        self.best_lane_width = np.nan
        # distance in meters of vehicle center from the line
        self.current_lane_width = np.nan
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = deque(maxlen=memory_size)
        # y values for detected line pixels
        self.ally = deque(maxlen=memory_size)
        # distance metric between the current and the previous lane
        self.tracking_distance = np.nan
        self.tracking_state = "reset"

        self.consistent_with_other_separator = False

    def get_points(self, tracked):
        if not self.allx or not self.ally:
            return None, None

        if tracked:
            return np.hstack(self.allx), np.hstack(self.ally)
        else:
            return self.allx[-1], self.ally[-1]

    @staticmethod
    def get_base_point_for_poly(y, fit):
        return eval_poly(fit, y)

    def get_base_point(self, y):
        return Lane.get_base_point_for_poly(y=y, fit=self.best_fit)

    def get_curvature(self, tracked):
        return self.best_radius_of_curvature if tracked else self.current_radius_of_curvature

    def get_width(self, tracked):
        return self.best_lane_width if tracked else self.current_lane_width

    def get_center_dist(self, tracked):
        return self.best_lane_center_distance if tracked else self.current_lane_center_distance


class LaneDetector:
    def __init__(self):
        # Calibration values
        self.calib_mtx = None
        self.calib_dist = None

        # Number of frames to keep in memory for tracking
        self.memory_size = 4
        # Tracked left and right lanes
        self.left_lane = Lane(self.memory_size)
        self.right_lane = Lane(self.memory_size)

        # Processing mode - image or video. Debug plots are generated when image is selected
        self.mode = 'image'  # or 'video'
        # Current frame id
        self.frame_id = 0
        # File name when processing videos - used also for exporting images in case of an error
        self.video_fn = None
        # Save every nth frame of the video during generation for debug purposes
        self.save_every_nth_frame = False

        # Input image size
        self.image_size = (None, None)
        # Image size of the warped image
        self.warped_image_size = 1280, 720

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 3 / 140  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 890  # meters per pixel in x dimension

        # Threshold under which two successive lane detections are considered to be consistent
        # This is the average distance in pixels between the polynomials fitted to the two lanes
        # across the full (warped) image
        self.tracking_lane_consistency_threshold = 100
        # Maximum number of frames where the detected lane is considered inconsistent before resetting
        # the tracker and starting from scratch
        self.tracking_max_inconsistent_frames = 8

    def reset_tracker(self):
        """ Reset tracker state - call it e.g. when jumping between unrelated images """
        self.left_lane = Lane(self.memory_size)
        self.right_lane = Lane(self.memory_size)

    def set_mode(self, mode):
        assert mode in ['image', 'video']
        self.mode = mode

    def calibrate_camera(self, input_images, grid_size=(6, 9)):
        """ Calibrate the camera using a set of checkerboard images """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[1], 0:grid_size[0]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        image_statistics = {'success': 0,
                            'failure': 0}
        # Step through the list and search for chessboard corners
        for fname in tqdm.tqdm_notebook(input_images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (grid_size[1], grid_size[0]), None)

            # If found, add object points, image points
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (grid_size[1], grid_size[0]), corners, ret)
                p = Path(fname)
                output_image_fn = str(p.with_name('calibrated_{}{}'.format(p.stem, p.suffix)))
                cv2.imwrite(output_image_fn, img)
                image_statistics['success'] += 1
            else:
                image_statistics['failure'] += 1

        print("Calibrating using {} images, could not find corners on {} images".format(image_statistics['success'],
                                                                                        image_statistics['failure']))

        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        self.calib_mtx = mtx
        self.calib_dist = dist

        return ret, mtx, dist

    def save_calibration(self, filename='calibration.pkl'):
        calib = {'mtx': self.calib_mtx, 'dist': self.calib_dist}
        with open(filename, 'wb') as f:
            pickle.dump(calib, f)

    def load_calibration(self, filename='calibration.pkl'):
        with open(filename, 'rb') as f:
            calib = pickle.load(f)
            self.calib_mtx = calib['mtx']
            self.calib_dist = calib['dist']

    def show_calibration_results(self, image_fns, grid_size=(6, 9)):
        """ Show a debug plot of the checkerboard calibration images. """
        n_images = len(image_fns)
        font_size = 16
        f, axs = plt.subplots(n_images, 3, figsize=(16, n_images * 4))
        f.tight_layout()
        for i, image_fn in enumerate(image_fns):
            image = mpimg.imread(image_fn)
            undist = cv2.undistort(image, self.calib_mtx, self.calib_dist, None, self.calib_mtx)

            axs[i][0].imshow(image)
            axs[i][0].set_title(image_fn, fontsize=font_size)
            axs[i][1].imshow(undist)
            axs[i][1].set_title("Undistorted", fontsize=font_size)

            # Find the chessboard corners
            gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (grid_size[1], grid_size[0]), None)

            if ret is True:
                x, y = 1200, 720
                selected_corners_src = corners[[0, grid_size[1] - 1, -grid_size[1], -1]]
                selected_corners_dst = np.float32([[100, 100], [x - 100, 100], [100, y - 100], [x - 100, y - 100]])
                M = cv2.getPerspectiveTransform(selected_corners_src, selected_corners_dst)
                warped = cv2.warpPerspective(undist, M, (x, y))
                axs[i][2].imshow(warped)
                axs[i][2].set_title("Warped", fontsize=font_size)

    def _warp(self, image, top_width, top_offset):
        """
        Apply perspective transform on an image to gain a birds eye view (BEV) representation.

        The transformation is characterized by four corners of a rectangle. Instead of specifying all
        corners individually, the base (closed side) of the rectangle is hard-coded along with the height,
        and only the upper side is changed.
        :param image: Input image (original perspective)
        :param top_width: Width of the further (top) edge of the trapezoid in the original perspective, in pixels
        :param top_offset: Offset from the center of the further (top) edge of the trapezoid in the original perspective
        :return:
            - The warped image
            - Perspective transformation matrix
            - Inverse of the transformation matrix
            - Source image points (trapezoid in the original perspective image)
            - Destinatinon image points (rectangle in the warped image)
        """
        self.warped_image_size = image.shape
        x, y = (image.shape[1], image.shape[0])
        # x, y = 1280, 1280
        dst_x_margin, dst_y_margin = 250, 0
        src_image_points = np.array([[220, 720],
                                     [x // 2 + top_offset - top_width / 2, 470],
                                     [x // 2 + top_offset + top_width / 2, 470],
                                     [1110, 720]], np.float32)
        dst_image_points = np.array([[dst_x_margin, y - dst_y_margin], [dst_x_margin, dst_y_margin],
                                     [x - dst_x_margin, dst_y_margin], [x - dst_x_margin, y - dst_y_margin]],
                                    np.float32)
        M = cv2.getPerspectiveTransform(src_image_points, dst_image_points)
        warped = cv2.warpPerspective(image, M, (x, y))

        M_inv = np.linalg.pinv(M)
        #M_inv = cv2.getPerspectiveTransform(dst_image_points, src_image_points)

        return warped, M, M_inv, src_image_points, dst_image_points

    @staticmethod
    def _threshold(img, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2), debug=True):
        """
        Apply thresholding on the image.

        Color thresholding is used to find the yellow and the bright white areas in the image.
        Gradient thresholding is used to select all gradients in the image that match the given
        magnitude and directrion parameters.
        The results of these two are combined together using an and.

        :param img: Input image (currently BEV)
        :param sobel_kernel: Kernel size of the Sobel operator used for gradient calculation
        :param mag_thresh: (min, max) of the accepted magnitude values
        :param dir_thresh: (min, max) of the accepted direction values
        :param debug: If True then the intermediate debug images are returned
        :return: A binary output image, plus 3 intermediate images if debug is True
        """
        # Make sure the dir thresh value is in a valid range, because the Jupyter widget had to be set
        # to a maximum value that is slightly above the largest valid value to still be usable
        dir_thresh = (dir_thresh[0], min(dir_thresh[1], np.pi / 2))

        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # Select the white-ish and yellow-ish parts of the image individually and combine (and) them
        # Inspired by this SO answer: https://stackoverflow.com/a/55827176

        # White-ish areas in image
        # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
        # L value must be relatively high (we want high brightness), e.g. within [0.8 ... 1.0] (OpenCV: [0 ... 255])
        # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
        white_lower = np.array([np.round(0 / 2), np.round(0.8 * 255), np.round(0.00 * 255)])
        white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
        white_mask = cv2.inRange(hls, white_lower, white_upper)

        # Yellow-ish areas in image
        # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
        # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
        # S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
        yellow_lower = np.array([np.round(40 / 2), np.round(0.00 * 255), np.round(0.10 * 255)])
        yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

        # Calculate combined mask, and masked image
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        color_thresh = cv2.bitwise_and(hls, hls, mask=mask)
        binary_color_thresh = color_thresh.sum(axis=-1) > 0

        # Use gradient filtering on a regular grayscale image
        # Calculate the magnitude and direction and apply the thresholds on both
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        absgradx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        absgrady = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        gradmag = np.sqrt(absgradx ** 2 + absgrady ** 2)
        graddir = np.arctan2(absgrady, absgradx)
        gradmag = np.uint8(255 * gradmag / np.max(gradmag))

        gradmag_filt = ((mag_thresh[0] <= gradmag) & (gradmag <= mag_thresh[1])) * 1
        graddir_filt = ((dir_thresh[0] <= graddir) & (graddir <= dir_thresh[1])) * 1

        # 5) Create a binary mask where direction thresholds are met
        binary_output = ((gradmag_filt & graddir_filt) | binary_color_thresh).astype(np.uint8)
        #binary_output = binary_color_thresh.astype(np.uint8)

        # Return this mask as your binary_output image
        if debug:
            return binary_output, gradmag_filt, graddir_filt, color_thresh
        else:
            return binary_output, None, None, None

    @staticmethod
    def _fit_poly(img_shape, leftx, lefty, rightx, righty):
        left_fit, right_fit = None, None
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        try:
            ### Fit a second order polynomial to each with np.polyfit() ###
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ### Calc both polynomials using ploty, left_fit and right_fit ###
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left_fit` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    @staticmethod
    def _get_points(fit_x, fit_y):
        return np.column_stack([fit_x, fit_y]).reshape((-1, 1, 2)).astype(np.int32)

    @staticmethod
    def _visualize_poly(out_img, left_pts, right_pts):
        if out_img is not None:
            out_img = cv2.polylines(out_img, [left_pts], isClosed=False, color=(255, 100, 0), thickness=10)
            out_img = cv2.polylines(out_img, [right_pts], isClosed=False, color=(0, 100, 255), thickness=10)

    @staticmethod
    def _find_lane_pixels_by_windows(binary_warped, out_img):
        """
        Find pixels belonging to the left and right lanes using a moving windows algorithm.

        :param binary_warped: Binary input image
        :param out_img: An image like binary_warped where debug plots are drawn, or None.
        :return: x and y coordinates of the identified left and right lanes, plus the modified out_img
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 30

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            if out_img is not None:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = int(nonzerox[good_left_inds].mean())
            if len(good_right_inds) > minpix:
                rightx_current = int(nonzerox[good_right_inds].mean())

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    @staticmethod
    def _find_lane_pixels_around_poly(binary_warped, out_img, left_poly, right_poly):
        """
        Find pixels belonging to the left and right lanes using a polynomial as a guide (which should
        be the last detected lanes).

        :param binary_warped: Binary input image
        :param out_img: An image like binary_warped where debug plots are drawn, or None.
        :param left_poly: Polynomial coefficients of the left lane (in the old numpy poly format)
        :param right_poly: Polynomial coefficients of the right lane (in the old numpy poly format)
        :return: x and y coordinates of the identified left and right lanes, plus the modified out_img
        """
        # Width of the margin around the previous polynomial to search
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (left_poly[0] * (nonzeroy ** 2) + left_poly[1] * nonzeroy +
                                       left_poly[2] - margin)) & (nonzerox < (left_poly[0] * (nonzeroy ** 2) +
                                                                              left_poly[1] * nonzeroy + left_poly[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_poly[0] * (nonzeroy ** 2) + right_poly[1] * nonzeroy +
                                        right_poly[2] - margin)) & (nonzerox < (right_poly[0] * (nonzeroy ** 2) +
                                                                                right_poly[1] * nonzeroy + right_poly[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        if out_img is not None:
            window_img = np.zeros_like(out_img)
            left_fitx, ploty = sample_poly(left_poly, binary_warped.shape[0])
            right_fitx, ploty = sample_poly(right_poly, binary_warped.shape[0])

            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                            ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                             ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        else:
            out_img = None

        return leftx, lefty, rightx, righty, out_img

    @staticmethod
    def _find_lane_separators(binary_warped, out_img, left_poly=None, right_poly=None):
        """
        Find lane pixels in the BEV image, fit a polynomial and sample it.

        :param binary_warped: Binary input image
        :param out_img: Output image based on the input image, with lane pixels highlighted and
            other debug information drawn for the lane finding.
        :param left_poly: Polynomial coefficients for the left lane, used as a prior for lane finding.
            If set then lane pixels are detected around this poly, if None then the sliding windows
            algorithm is used.
        :param right_poly: Polynomial coefficients for the right lane, used as a prior for lane finding.
            If set then lane pixels are detected around this poly, if None then the sliding windows
            algorithm is used.
        :return:
        """
        # Find our lane pixels first
        if left_poly is not None and right_poly is not None:
            leftx, lefty, rightx, righty, out_img = LaneDetector._find_lane_pixels_around_poly(
                binary_warped, out_img, left_poly, right_poly)
        else:
            leftx, lefty, rightx, righty, out_img = LaneDetector._find_lane_pixels_by_windows(
                binary_warped, out_img)

        # print("L/r points {} / {}".format(len(leftx), len(rightx)))
        left_poly, right_poly, left_fitx, right_fitx, ploty = LaneDetector._fit_poly(
            binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Colors in the left and right lane regions
        if out_img is not None:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        left_pts = LaneDetector._get_points(left_fitx, ploty)
        right_pts = LaneDetector._get_points(right_fitx, ploty)

        LaneDetector._visualize_poly(out_img, left_pts, right_pts)

        return left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx, righty, out_img

    def _measure_curvature(self):
        """ Calculates the curvature of polynomial functions in meters. """
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        # y_eval = np.max(ploty)
        y_eval = self.warped_image_size[1]

        def curv(a, b, y):
            return ((1 + (2 * a * y + b) ** 2) ** (3 / 2)) / np.abs(2 * a)

        if self.left_lane.current_fit is not None:
            self.left_lane.current_radius_of_curvature = curv(
                self.left_lane.current_fit[0], self.left_lane.current_fit[1], y_eval * self.ym_per_pix)
        if self.left_lane.best_fit is not None:
            self.left_lane.best_radius_of_curvature = curv(
                self.left_lane.best_fit[0], self.left_lane.best_fit[1], y_eval * self.ym_per_pix)

        if self.right_lane.current_fit is not None:
            self.right_lane.current_radius_of_curvature = curv(
                self.right_lane.current_fit[0], self.right_lane.current_fit[1], y_eval * self.ym_per_pix)
        if self.right_lane.best_fit is not None:
            self.right_lane.best_radius_of_curvature = curv(
                self.right_lane.best_fit[0], self.right_lane.best_fit[1], y_eval * self.ym_per_pix)

    def _get_lane_width_and_center_distance(self, left_fit, right_fit):
        """ Get the lane width and distance from the lane center for a left and right lane separator. """

        if left_fit is None or right_fit is None:
            return np.nan, np.nan
        left_center = Lane.get_base_point_for_poly(y=self.warped_image_size[1], fit=left_fit)
        right_center = Lane.get_base_point_for_poly(y=self.warped_image_size[1], fit=right_fit)
        lane_width_px = right_center - left_center
        lane_width_m = lane_width_px * self.xm_per_pix
        lane_center_px = left_center + lane_width_px / 2
        distance_m = (self.warped_image_size[0] - lane_center_px) * self.xm_per_pix
        return lane_width_m, distance_m

    def _measure_lane_center_distance(self):
        """ Calculate and save lane center distance based on the tracked lane separators. """

        width, center_distance = self._get_lane_width_and_center_distance(
            left_fit=self.left_lane.best_fit, right_fit=self.right_lane.best_fit)
        self.left_lane.best_lane_center_distance = center_distance
        self.right_lane.best_lane_center_distance = center_distance
        self.left_lane.best_lane_width = width
        self.right_lane.best_lane_width = width

    def _draw_text(self, img, text, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (200, 100, 100)
        thickness = 2
        return cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    def _get_final_image(self, undist, M_inv, warped_shape, warped_pts, tracked):
        """
        Get the final output image by drawing the detected lane polygon as an overlay and
        adding textual information.

        :param undist: Undistorted (input) image
        :param M_inv: Inverse perspective transformation (from warped to undistorted)
        :param warped_shape: Shape of the warped image
        :param warped_pts: Points of the lane to draw (in warped image space)
        :param tracked: If True then the tracked (best) values are drawn on the image,
            otherwise the current (last) detection values are added.
        :return: Modified undistorted image with the detected lane
        """
        x, y = (warped_shape[1], warped_shape[0])
        u_x, u_y, u_c = (undist.shape[1], undist.shape[0], undist.shape[2])
        warped_pts = warped_pts.reshape(-1, 2)

        # Originally I wanted to apply the perspective transform on the points directly
        # from warped image space to the original (undistorted) image space, and drawing the polygon
        # on the undistorted image, as this is more efficient
        # than drawing in the warped space and transforming the whole image. However in this case the
        # resulting polygon was always slightly off (larger), which I verified by comparing the two methods.
        # I am not sure what caused this, but I had to switch to using warpPerspective instead of perspectiveTransform
        # to correct it.
        if False:
            undist_pts = cv2.perspectiveTransform(warped_pts.reshape(1, -1, 2).astype(np.float32), M_inv)
            lane_layer = np.zeros((u_y, u_x, u_c), dtype=np.uint8)
            lane_layer = cv2.polylines(lane_layer_original, [undist_pts.astype(np.int32).reshape(-1, 2)], isClosed=True,
                                      color=(200, 0, 0, 90), thickness=20)

        lane_layer = np.zeros((y, x, u_c), dtype=np.uint8)
        lane_layer = cv2.polylines(lane_layer, [warped_pts.astype(np.int32).reshape(-1, 2)], isClosed=True,
                      color=(0, 200, 0, 30), thickness=20)
        lane_layer = cv2.warpPerspective(lane_layer, M_inv, (u_x, u_y))
        out_img = cv2.addWeighted(undist, 1.0, lane_layer, 0.3, 0.0)

        out_img = self._draw_text(out_img, "Frame: {}".format(self.frame_id), (50, 50))
        out_img = self._draw_text(out_img, "Curvature l/r: {:6.0f} m / {:6.0f} m".format(
            self.left_lane.get_curvature(tracked), self.right_lane.get_curvature(tracked)), (50, 100))
        out_img = self._draw_text(out_img, "Lane center dist: {:.02f} m".format(
            self.left_lane.get_center_dist(tracked)), (50, 150))
        out_img = self._draw_text(out_img, "Lane width: {:.02f} m".format(
            self.left_lane.get_width(tracked)), (50, 200))
        out_img = self._draw_text(out_img, "Current lane separators consistent: {}".format(
            self.left_lane.consistent_with_other_separator), (50, 250))

        out_img = self._draw_text(out_img, "Lane tracking dist l/r: {:.0f} / {:.0f}".format(
            self.left_lane.tracking_distance, self.right_lane.tracking_distance), (550, 50))
        out_img = self._draw_text(out_img, "Lane tracking state l/r: {:13s} / {:13s}".format(
            self.left_lane.tracking_state, self.right_lane.tracking_state), (550, 100))

        return out_img

    def _is_lane_separator_temporally_consistent(self, poly, poly_base, range_, threshold):
        """
        Check if the current and the previous best polynomial for the given separator are
            within a threshold.

        This is used to check if the current detection is consistent with the previous one.

        The currently used metric is the average distance (in pixels) between the points
            of the two polynoms across the range. This is now calculated directly, though
            a similar value could be calculated using Legendre polynomials, as described
            here: http://asymptoticlabs.com/blog/posts/a-simple-similarity-function-for-polynomial-curves.html

        :param poly: Polynomial of the current fit.
        :param poly_base: Polynomial of the previous best fit.
        :param range_: Range of values where the consistency is checked - in practice
            this is the height of the (warped) image.
        :param threshold: Threshold value, see the docstring for the description of the metric.
        :return: True/False and the calculated distance metric
        """
        if poly is None:
            return False, -1
        if poly_base is None:
            return True, -1

        fitx, y = sample_poly(poly, range_)
        fitx_base, y = sample_poly(poly_base, range_)
        result = np.sum(np.abs(fitx - fitx_base)) / np.max(y)
        return result < threshold, result

    def _is_lane_valid(self, left_poly, right_poly):
        """
        Get if the two detected separators can describe a valid lane.
        Currently only the width of the lane is checked at the bottom of the image,
        though this could be extended with additional checks, e.g. that they are not
        crossing each other.
        """
        width, center_distance = self._get_lane_width_and_center_distance(left_poly, right_poly)
        self.left_lane.current_lane_width = width
        self.right_lane.current_lane_width = width
        self.left_lane.current_lane_center_distance = center_distance
        self.right_lane.current_lane_center_distance = center_distance
        return 3.0 < width < 4.8

    def _track_lane_separators(self, out_img, warped_img_shape, left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx,
                               righty):
        """
        Perform lane tracking across frames.

        - Check that each separator is consistent with the previous best fit for the same separator. If not then
            discard it and use the previous best
        - Check that the two last separators
        """
        # Check if the left and right separators are temporally consistent (close to their counterparts on the
        # previous frames)
        left_valid, left_tracking_distance = self._is_lane_separator_temporally_consistent(
            poly=left_poly, poly_base=self.left_lane.best_fit, range_=warped_img_shape[0],
            threshold=self.tracking_lane_consistency_threshold)
        right_valid, right_tracking_distance = self._is_lane_separator_temporally_consistent(
            poly=right_poly, poly_base=self.right_lane.best_fit, range_=warped_img_shape[0],
            threshold=self.tracking_lane_consistency_threshold)

        self.left_lane.tracking_distance = left_tracking_distance
        self.right_lane.tracking_distance = right_tracking_distance

        # Check that the resulting lane built from the current left and right poly result in a valid lane.
        # If either left_poly or right_poly is invalid in the current frame then their previous best
        # fit is used as a substitute.
        left_poly_for_lane_validity_check = left_poly if left_valid else self.left_lane.best_fit
        right_poly_for_lane_validity_check = right_poly if right_valid else self.right_lane.best_fit
        is_lane_valid = self._is_lane_valid(left_poly_for_lane_validity_check, right_poly_for_lane_validity_check)
        self.left_lane.consistent_with_other_separator = is_lane_valid
        self.right_lane.consistent_with_other_separator = is_lane_valid
        if not is_lane_valid:
            # Reject both separators if their combination does not look like a valid lane
            left_valid = False
            right_valid = False

        # Add the current left and right separators to the track if they are accepted, or signal that they
        # were invalid.
        if left_valid:
            self.left_lane.current_fit = left_poly
            self.left_lane.detected = self.left_lane.current_fit is not None
            self.left_lane.recent_xfitted.append(left_pts)
            self.left_lane.allx.append(leftx)
            self.left_lane.ally.append(lefty)
            self.left_lane.last_valid_frame_id = self.frame_id
            self.left_lane.consistent_with_other_separator = True
            if self.left_lane.tracking_state == "reset":
                self.left_lane.tracking_state = "valid (reset)"
            else:
                self.left_lane.tracking_state = "valid"
        else:
            self.left_lane.current_fit = None
            self.left_lane.detected = False
            self.left_lane.tracking_state = "invalid"

        if right_valid:
            self.right_lane.current_fit = right_poly
            self.right_lane.detected = self.right_lane.current_fit is not None
            self.right_lane.recent_xfitted.append(right_pts)
            self.right_lane.allx.append(rightx)
            self.right_lane.ally.append(righty)
            self.right_lane.last_valid_frame_id = self.frame_id
            self.right_lane.consistent_with_other_separator = True
            if self.right_lane.tracking_state == "reset":
                self.right_lane.tracking_state = "valid (reset)"
            else:
                self.right_lane.tracking_state = "valid"
        else:
            self.right_lane.current_fit = None
            self.right_lane.detected = False
            self.right_lane.tracking_state = "invalid"

        # Calculate the new best fit polynomial after the updates
        left_combined_points_x, left_combined_points_y = self.left_lane.get_points(tracked=True)
        right_combined_points_x, right_combined_points_y = self.right_lane.get_points(tracked=True)
        left_fitnew, right_fitnew, left_fitx, right_fitx, ploty = self._fit_poly(
            warped_img_shape,
            left_combined_points_x, left_combined_points_y,
            right_combined_points_x, right_combined_points_y)
        self.left_lane.best_fit = left_fitnew
        self.left_lane.bestx = left_fitx
        self.right_lane.best_fit = right_fitnew
        self.right_lane.bestx = right_fitx

        # Visualize and return results
        if out_img is not None:
            out_img[left_combined_points_y, left_combined_points_x] = [255, 0, 0]
            out_img[right_combined_points_y, right_combined_points_x] = [0, 0, 255]

        left_pts = np.column_stack([self.left_lane.bestx, ploty]).reshape((-1, 1, 2)).astype(np.int32)
        right_pts = np.column_stack([self.right_lane.bestx, ploty]).reshape((-1, 1, 2)).astype(np.int32)
        if out_img is not None:
            out_img = cv2.polylines(out_img, [left_pts], isClosed=False, color=(255, 100, 0), thickness=10)
            out_img = cv2.polylines(out_img, [right_pts], isClosed=False, color=(0, 100, 255), thickness=10)
        return out_img, left_pts, right_pts

    def _is_tracking_reset_needed(self):
        if self.left_lane.last_valid_frame_id is None or self.right_lane.last_valid_frame_id is None:
            return False
        left_reset_needed = self.frame_id - self.left_lane.last_valid_frame_id > self.tracking_max_inconsistent_frames
        right_reset_needed = self.frame_id - self.right_lane.last_valid_frame_id > self.tracking_max_inconsistent_frames
        return left_reset_needed or right_reset_needed

    def process_image(self, image, tracking_enabled=True, sobel_kernel=7,
                      mag_thresh=(20, 255), dir_thresh=(0.0, 0.6), top_width=154, top_offset=6):
        if isinstance(image, str):
            # Read in an image if an image file name was provided
            image = mpimg.imread("test_images/{}".format(image))

        # Undistort the original image
        undist = cv2.undistort(image, self.calib_mtx, self.calib_dist, None, self.calib_mtx)

        # Warp into birds eye view
        warped, M, M_inv, src_image_points, dst_image_points = self._warp(undist, top_width, top_offset)

        # Display rectangle used for warping on the original image for debug purposes
        if self.mode == 'image':
            undist_dbg = cv2.polylines(undist.copy(), [src_image_points.astype(np.int32).reshape((-1, 1, 2))],
                                       isClosed=False, color=(255, 0, 0), thickness=5)

        # Select possible lane points using thresholding - gradient magnitude, direction, and color
        binary, gradmag_filt, graddir_filt, color_thresh = LaneDetector._threshold(
            warped, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, dir_thresh=dir_thresh, debug=self.mode == 'image')

        if self.mode == 'image':
            bev_debug_output_windows = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB)
            bev_debug_output_poly = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB)
            tracking_output = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB)
        else:
            bev_debug_output_windows = None
            bev_debug_output_poly = None
            tracking_output = None

        # Reset the trackers if lane tracking failed for too long in the previous frames.
        # This is done before calling the lane finding functions (hence it is not part of the track_lanes function),
        # as it modifies the prior used by the lane pixel detector
        if self._is_tracking_reset_needed():
            self.reset_tracker()

        # Detect points that belong to the lanes.
        # If previous lane detections are available then start from those
        polybased_lanefinder_result = None
        if tracking_enabled and self.left_lane.detected and self.right_lane.detected:
            left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx, righty, polybased_lanefinder_result = LaneDetector._find_lane_separators(
                binary, bev_debug_output_poly, self.left_lane.current_fit, self.right_lane.current_fit)
        # If previous points are not available, or the polyline-based lane finding failed, then detect lanes from
        # scratch using a sliding windows algorithm
        else:
            left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx, righty, bev_debug_output_windows = LaneDetector._find_lane_separators(
                binary, bev_debug_output_windows)

        # Perform lane tracking. Reject current lane if it deviates from previous ones too much
        tracking_output, tracked_left_pts, tracked_right_pts = self._track_lane_separators(tracking_output, warped.shape,
                                                                                           left_poly, right_poly, left_pts,
                                                                                           right_pts, leftx, lefty, rightx, righty)

        # Measure curvature, lane center distance and other lane parameters
        self._measure_curvature()
        self._measure_lane_center_distance()

        # Create the final image by showing an overlay of the detected lanes.
        # When in image mode then do this separately for the tracked and untracked lanes as well, otherwise
        # only calculate the tracked version
        if self.mode == 'image':
            warped_lane_pts = np.hstack([left_pts, right_pts])
            final_img_singleframe = self._get_final_image(undist, M_inv, warped.shape, warped_lane_pts, tracked=False)
        warped_combined_lane_pts = np.hstack([tracked_left_pts, tracked_right_pts])
        final_img_tracked = self._get_final_image(undist, M_inv, warped.shape, warped_combined_lane_pts, tracked=True)

        self.frame_id += 1

        # Plot the result
        if self.mode == 'image':
            font_size = 30
            f, axs = plt.subplots(6, 2, figsize=(18, 38))
            f.tight_layout()
            axs[0][0].imshow(image)
            axs[0][0].set_title('Original', fontsize=font_size)
            axs[0][1].imshow(undist_dbg)
            axs[0][1].set_title('Undistorted', fontsize=font_size)

            axs[1][0].imshow(warped)
            axs[1][0].set_title('Birds eye view (BEV)', fontsize=font_size)

            axs[1][1].imshow(color_thresh)
            axs[1][1].set_title('Color thresholding', fontsize=font_size)

            axs[2][0].imshow(gradmag_filt, cmap='gray')
            axs[2][0].set_title('Grad. magnitude thresh.', fontsize=font_size)
            axs[2][1].imshow(graddir_filt, cmap='gray')
            axs[2][1].set_title('Grad. direction thresh.', fontsize=font_size)

            axs[3][0].imshow(binary, cmap='gray')
            axs[3][0].set_title('Final thresholded image', fontsize=font_size)

            axs[3][1].imshow(bev_debug_output_windows)
            axs[3][1].set_title('Sliding windows lane finding (if used)', fontsize=font_size)
            if polybased_lanefinder_result is not None:
                axs[4][0].imshow(polybased_lanefinder_result)
            axs[4][0].set_title('Polygon-based lane finding (if used)', fontsize=font_size)

            axs[4][1].imshow(tracking_output)
            axs[4][1].set_title('Tracked lane finding (current + previous points)', fontsize=font_size)

            axs[5][0].imshow(final_img_singleframe, cmap='gray')
            axs[5][0].set_title('Final image (current frame only)', fontsize=font_size)
            axs[5][1].imshow(final_img_tracked, cmap='gray')
            axs[5][1].set_title('Final image (tracked)', fontsize=font_size)

            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        if self.mode == 'video':
            return final_img_tracked

    def process_multiple_images(self, image_fns):
        """
        Process multiple images and display the results for them - great for debugging tracking.

        :param image_fns: Image file names to process
        :return:
        """
        self.frame_id = 0
        self.reset_tracker()
        self.set_mode('video')
        n_images = len(image_fns)
        font_size = 16
        f, axs = plt.subplots(n_images, 1, figsize=(8, n_images * 5))
        f.tight_layout()
        for i, image_fn in enumerate(image_fns):
            result = self.process_image(image=image_fn)
            axs[i].imshow(result)
            axs[i].set_title(image_fn, fontsize=font_size)

    def _process_video_frame(self, image):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = self.process_image(image)
        except Exception as e:
            logging.error(
                "Error happened while processing frame {}, saving it for debugging. Error: {}".format(self.frame_id, e))
            # save erroneous frames from the video for easier debugging
            mpimg.imsave("test_images/{}_error_frame_{}.jpg".format(self.video_fn, self.frame_id), image)
            result = image
            raise e

        if self.save_every_nth_frame and self.frame_id % self.save_every_nth_frame == 0:
            mpimg.imsave("test_images/{}_frame_{}.jpg".format(self.video_fn, self.frame_id), image)

        return result

    def display_video_in_notebook(self, fn):
        return HTML("""
        <video width="960" height="540" controls>
          <source src="{0}">
        </video>
        """.format(fn))

    def process_video(self, input_fn, subclip_range=None, display=False, as_gif=False):
        """
        Process a video, detect lanes in each frame and output a new video.

        Outputs a video by the same name in 'output_videos'.

        :param input_fn: Input video file name
        :param subclip_range: Tuple of (start, end) seconds to process. Note: 0 is always assigned as the
            first frame_id in the output, even if start > 0.
        """
        self.frame_id = 0
        self.reset_tracker()
        self.video_fn = input_fn
        input_path = "{}".format(input_fn)
        output_path = "output_videos/{}".format(input_fn)
        if not os.path.exists('output_videos'):
            os.mkdir('output_videos')
        input_clip = VideoFileClip(input_path)
        if subclip_range is not None:
            input_clip = input_clip.subclip(subclip_range[0], subclip_range[1])
        self.set_mode('video')
        output_clip = input_clip.fl_image(self._process_video_frame)
        if as_gif:
            output_path = output_path.replace(".mp4", ".gif")
            output_clip.write_videofile(output_path, audio=False, codec='gif', fps=10)
        else:
            output_clip.write_videofile(output_path, audio=False)

        if display:
            return self.display_video_in_notebook(output_path)

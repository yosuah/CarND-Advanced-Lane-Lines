#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.colors as colors
from numpy.polynomial import Polynomial
import logging
import functools
import warnings
import glob
from pathlib import Path
import tqdm
import pickle
from collections import deque


class Lane:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=3)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_fit_x_at_bottom = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = deque(maxlen=3)
        # y values for detected line pixels
        self.ally = deque(maxlen=3)
        # distance metric between the current and the previous lane
        self.tracking_distance = None

    def get_all_points(self):
        return np.hstack(self.allx), np.hstack(self.ally)

    def get_best_fit_x_at_bottom(self):
        return


class LaneDetector:
    def __init__(self):
        self.calib_mtx = None
        self.calib_dist = None
        self.left_lane = Lane()
        self.right_lane = Lane()
        self.frame_id = 0
        self.mode = 'image'  # or 'video'
        self.video_fn = None
        self.save_every_nth_frame = False
        self.image_size = (None, None)
        self.warped_image_size = 1280, 720

    def set_mode(self, mode):
        assert mode in ['image', 'video']
        self.mode = mode

    def calibrate_camera(self, input_images, grid_size=(6, 8)):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[1], 0:grid_size[0]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        image_statistics = {'success': 0,
                            'failure': 0}
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(tqdm.tqdm_notebook(input_images)):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (grid_size[1], grid_size[0]), None)

            # If found, add object points, image points
            if ret == True:
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

    @staticmethod
    def threshold(img, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2), debug=True):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        s = hls[:, :, 2]
        # binary_color_thresh = ((s_thresh[0] < s) & (s <= s_thresh[1])).astype(np.uint8)
        # binary_color_thresh = cv2.dilate(binary_color_thresh, np.ones((5, 5), np.uint8), iterations=1)
        # return binary_output

        dir_thresh = (dir_thresh[0], min(dir_thresh[1], np.pi / 2))
        # print("sobel_kernel={}, mag_thresh={}, dir_thresh={}".format(sobel_kernel, mag_thresh, dir_thresh))

        # Apply the following steps to img
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = s
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        absgradx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        absgrady = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        gradmag = np.sqrt(absgradx ** 2 + absgrady ** 2)
        graddir = np.arctan2(absgrady, absgradx)

        gradmag = np.uint8(255 * gradmag / np.max(gradmag))

        gradmag_filt = ((mag_thresh[0] <= gradmag) & (gradmag <= mag_thresh[1])) * 1
        graddir_filt = (dir_thresh[0] <= graddir) & (graddir <= dir_thresh[1])

        # 5) Create a binary mask where direction thresholds are met
        binary_output = (gradmag_filt & graddir_filt).astype(np.uint8)

        # 6) Return this mask as your binary_output image
        if debug:
            return binary_output, gradmag_filt, graddir_filt
        else:
            return binary_output, None, None

    @staticmethod
    def eval_poly(fit, p):
        return fit[0] * p ** 2 + fit[1] * p + fit[2]

    @staticmethod
    def sample_poly(fit, y_range):
        ploty = np.linspace(0, y_range - 1, y_range)
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        return fitx, ploty

    @staticmethod
    def fit_poly(img_shape, leftx, lefty, rightx, righty):
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
    def get_points(fit_x, fit_y):
        return np.column_stack([fit_x, fit_y]).reshape((-1, 1, 2)).astype(np.int32)

    @staticmethod
    def visualize_poly(out_img, left_pts, right_pts):
        if out_img is not None:
            out_img = cv2.polylines(out_img, [left_pts], isClosed=False, color=(255, 100, 0), thickness=10)
            out_img = cv2.polylines(out_img, [right_pts], isClosed=False, color=(0, 100, 255), thickness=10)

    @staticmethod
    def find_lane_pixels_by_windows(binary_warped, out_img):
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
    def find_lane_pixels_around_poly(binary_warped, out_img, left_fit, right_fit):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
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
            left_fitx, ploty = LaneDetector.sample_poly(left_fit, binary_warped.shape[0])
            right_fitx, ploty = LaneDetector.sample_poly(right_fit, binary_warped.shape[0])

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
    def find_lane_in_bev(binary_warped, out_img, left_fit=None, right_fit=None):
        # Find our lane pixels first
        if left_fit is not None and right_fit is not None:
            leftx, lefty, rightx, righty, out_img = LaneDetector.find_lane_pixels_around_poly(
                binary_warped, out_img, left_fit, right_fit)
        else:
            leftx, lefty, rightx, righty, out_img = LaneDetector.find_lane_pixels_by_windows(
                binary_warped, out_img)

        print("L/r points {} / {}".format(len(leftx), len(rightx)))
        left_fit, right_fit, left_fitx, right_fitx, ploty = LaneDetector.fit_poly(
            binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Colors in the left and right lane regions
        if out_img is not None:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        left_pts = LaneDetector.get_points(left_fitx, ploty)
        right_pts = LaneDetector.get_points(right_fitx, ploty)

        LaneDetector.visualize_poly(out_img, left_pts, right_pts)

        return left_fit, right_fit, left_pts, right_pts, leftx, lefty, rightx, righty, out_img

    def warp(self, image, top_width, top_offset):
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

        return warped, M, M_inv, src_image_points, dst_image_points

    def measure_curvature(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        # y_eval = np.max(ploty)
        y_eval = self.warped_image_size[1]

        def curv(a, b, y):
            return (1 + (2 * a * y + b) ** 2) ** (3 / 2) / (2 * a)

        self.left_lane.radius_of_curvature = curv(self.left_lane.best_fit[0], self.left_lane.best_fit[1],
                                                  y_eval * ym_per_pix)  ## Implement the calculation of the left line here
        self.right_lane.radius_of_curvature = curv(self.right_lane.best_fit[0], self.right_lane.best_fit[1],
                                                   y_eval * ym_per_pix)  ## Implement the calculation of the right line here

    def measure_lane_center_distance(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        # y_eval = np.max(ploty)
        y_eval = self.warped_image_size[1]

        center = self.left_lane

    def get_final_image(self, undist, M_inv, warped_shape, warped_pts):
        x, y = (warped_shape[1], warped_shape[0])
        warped_pts = warped_pts.reshape(-1, 2)

        # undist_pts = cv2.warpPerspective(warped_pts.astype(np.float32), M_inv, (len(warped_pts), 2))
        undist_pts = cv2.perspectiveTransform(warped_pts.reshape(1, -1, 2).astype(np.float32), M_inv)
        out_img = undist.copy()
        out_img = cv2.polylines(out_img, [undist_pts.astype(np.int32).reshape(-1, 2)], isClosed=True,
                                color=(0, 200, 0, 30), thickness=20)
        out_img = cv2.addWeighted(undist, 0.7, out_img, 0.3, 0.0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (200, 100, 100)
        thickness = 2
        out_img = cv2.putText(out_img, "Frame: {}".format(self.frame_id), (50, 50), font, fontScale, color, thickness,
                              cv2.LINE_AA)
        out_img = cv2.putText(out_img, "Curvature l/r: {:.0f} / {:.0f}".format(
            self.left_lane.radius_of_curvature, self.right_lane.radius_of_curvature), (50, 100), font, fontScale, color,
                              thickness, cv2.LINE_AA)
        out_img = cv2.putText(out_img, "Lane center dist: {}".format(self.frame_id), (50, 150), font, fontScale, color,
                              thickness, cv2.LINE_AA)
        out_img = cv2.putText(out_img, "Lane tracking dist l/r: {:.0f} / {:.0f}".format(
            self.left_lane.tracking_distance, self.right_lane.tracking_distance), (50, 200), font, fontScale, color,
                              thickness, cv2.LINE_AA)

        return out_img

    @staticmethod
    def is_lane_consistent(poly, poly_base, range_, threshold=100):
        if poly is None:
            return False, -1
        if poly_base is None:
            return True, -1

        fitx, y = LaneDetector.sample_poly(poly, range_)
        fitx_base, y = LaneDetector.sample_poly(poly_base, range_)
        result = np.sum(np.abs(fitx - fitx_base)) / np.max(y)
        return result < threshold, result

    def track_lanes(self, out_img, warped_img_shape, left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx,
                    righty):
        left_consistent, left_tracking_distance = self.is_lane_consistent(left_poly, self.left_lane.best_fit,
                                                                          warped_img_shape[0])
        right_consistent, right_tracking_distance = self.is_lane_consistent(right_poly, self.right_lane.best_fit,
                                                                            warped_img_shape[0])

        self.left_lane.tracking_distance = left_tracking_distance
        self.right_lane.tracking_distance = right_tracking_distance

        if left_consistent:
            self.left_lane.current_fit = left_poly
            self.left_lane.detected = self.left_lane.current_fit is not None
            self.left_lane.recent_xfitted.append(left_pts)
            self.left_lane.allx.append(leftx)
            self.left_lane.ally.append(lefty)
        else:
            self.left_lane.current_fit = None
            self.left_lane.detected = False

        if right_consistent:
            self.right_lane.current_fit = right_poly
            self.right_lane.detected = self.right_lane.current_fit is not None
            self.right_lane.recent_xfitted.append(right_pts)
            self.right_lane.allx.append(rightx)
            self.right_lane.ally.append(righty)
        else:
            self.right_lane.current_fit = None
            self.right_lane.detected = False

        left_combined_points_x, left_combined_points_y = self.left_lane.get_all_points()
        right_combined_points_x, right_combined_points_y = self.right_lane.get_all_points()
        # Fit new polynomials
        left_fitnew, right_fitnew, left_fitx, right_fitx, ploty = self.fit_poly(
            warped_img_shape,
            left_combined_points_x, left_combined_points_y,
            right_combined_points_x, right_combined_points_y)
        self.left_lane.best_fit = left_fitnew
        self.left_lane.bestx = left_fitx
        self.right_lane.best_fit = right_fitnew
        self.right_lane.bestx = right_fitx

        if out_img is not None:
            out_img[left_combined_points_y, left_combined_points_x] = [255, 0, 0]
            out_img[right_combined_points_y, right_combined_points_x] = [0, 0, 255]

        left_pts = np.column_stack([self.left_lane.bestx, ploty]).reshape((-1, 1, 2)).astype(np.int32)
        right_pts = np.column_stack([self.right_lane.bestx, ploty]).reshape((-1, 1, 2)).astype(np.int32)
        if out_img is not None:
            out_img = cv2.polylines(out_img, [left_pts], isClosed=False, color=(255, 100, 0), thickness=10)
            out_img = cv2.polylines(out_img, [right_pts], isClosed=False, color=(0, 100, 255), thickness=10)
        return out_img, left_pts, right_pts

    def detect_lanes(self, image, sobel_kernel=7,
                     mag_thresh=(10, 255), dir_thresh=(0.7, 1.2),
                     top_width=154, top_offset=6):
        if isinstance(image, str):
            # Read in an image if an image file name was provided
            image = mpimg.imread("test_images/{}".format(image))

        # Undistort
        undist = cv2.undistort(image, self.calib_mtx, self.calib_dist, None, self.calib_mtx)

        # Threshold
        binary, gradmag_filt, graddir_filt = LaneDetector.threshold(undist, sobel_kernel=sobel_kernel,
                                                                    mag_thresh=mag_thresh, dir_thresh=dir_thresh,
                                                                    debug=self.mode == 'image')

        # binary_color = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB)
        # binary_color[:, :, 1][binary_color.any(axis=2)] = np.outer(np.arange(720) // (720/255), np.ones(1280))[binary_color.any(axis=2)]

        warped, M, M_inv, src_image_points, dst_image_points = self.warp(binary, top_width, top_offset)

        undist_dbg = cv2.polylines(undist.copy(), [src_image_points.astype(np.int32).reshape((-1, 1, 2))],
                                   isClosed=False, color=(255, 0, 0), thickness=5)

        if self.mode == 'image':
            bev_debug_output = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2RGB)
            bev_debug_output2 = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2RGB)
            tracking_output = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2RGB)
        else:
            bev_debug_output = None
            bev_debug_output2 = None
            tracking_output = None

        # If previous lane detections are available then start from those
        polybased_lanefinder_result = None
        if self.left_lane.detected and self.right_lane.detected:
            left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx, righty, polybased_lanefinder_result = LaneDetector.find_lane_in_bev(
                warped, bev_debug_output2, self.left_lane.current_fit, self.right_lane.current_fit)
        # Otherwise detect lanes from scratch
        else:
            left_poly, right_poly, left_pts, right_pts, leftx, lefty, rightx, righty, bev_debug_output = LaneDetector.find_lane_in_bev(
                warped, bev_debug_output)

        tracking_output, tracked_left_pts, tracked_right_pts = self.track_lanes(tracking_output, warped.shape,
                                                                                left_poly, right_poly, left_pts,
                                                                                right_pts, leftx, lefty, rightx, righty)

        self.measure_curvature()

        warped_lane_pts = np.hstack([left_pts, right_pts])
        final_img_singleframe = self.get_final_image(undist, M_inv, warped.shape, warped_lane_pts)

        warped_combined_lane_pts = np.hstack([tracked_left_pts, tracked_right_pts])
        final_img_tracked = self.get_final_image(undist, M_inv, warped.shape, warped_combined_lane_pts)

        if self.mode == 'image':
            # Plot the result
            f, axs = plt.subplots(6, 2, figsize=(24, 38))
            f.tight_layout()
            axs[0][0].imshow(image)
            axs[0][0].set_title('Original Image', fontsize=50)
            axs[0][1].imshow(undist_dbg)
            axs[0][1].set_title('Undistort', fontsize=50)

            axs[1][0].imshow(gradmag_filt, cmap='gray')
            axs[1][0].set_title('Mag filt', fontsize=50)
            axs[1][1].imshow(graddir_filt, cmap='gray')
            axs[1][1].set_title('Dir filt', fontsize=50)

            axs[2][0].imshow(binary, cmap='gray')
            axs[2][0].set_title('Thresholded image', fontsize=50)
            axs[2][1].imshow(warped, cmap='gray')
            axs[2][1].set_title('BEV', fontsize=50)

            axs[3][0].imshow(bev_debug_output)
            axs[3][0].set_title('BEV with debug', fontsize=50)
            if polybased_lanefinder_result is not None:
                axs[3][1].imshow(polybased_lanefinder_result)
            axs[3][1].set_title('Polybased BEV with debug', fontsize=50)

            axs[4][0].imshow(tracking_output)
            axs[4][0].set_title('Tracked BEV with debug', fontsize=50)

            axs[5][0].imshow(final_img_singleframe, cmap='gray')
            axs[5][0].set_title('Final image singleframe', fontsize=50)
            axs[5][1].imshow(final_img_tracked, cmap='gray')
            axs[5][1].set_title('Final image tracked', fontsize=50)

            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        self.frame_id += 1

        if self.mode == 'video':
            return final_img_tracked

    def process_video_frame(self, image):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = self.detect_lanes(image)
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

    def process_video(self, input_fn, subclip_range=None):
        self.frame_id = 0
        self.video_fn = input_fn
        input_path = "{}".format(input_fn)
        output_path = "output_videos/{}".format(input_fn)
        if not os.path.exists('output_videos'):
            os.mkdir('output_videos')
        input_clip = VideoFileClip(input_path)
        if subclip_range is not None:
            input_clip = input_clip.subclip(subclip_range[0], subclip_range[1])
        self.set_mode('video')
        output_clip = input_clip.fl_image(self.process_video_frame)
        output_clip.write_videofile(output_path, audio=False)


if __name__ == "__main__":
    detector = LaneDetector()
    # calibration_images = glob.glob('camera_cal/calibration*.jpg')
    # ret, mtx, dist = detector.calibrate_camera(calibration_images, grid_size=(6, 9))
    # detector.save_calibration()
    detector.load_calibration()
    detector.set_mode('image')

    detector.detect_lanes('project_video.mp4_frame_3.jpg')

    # detector.save_every_nth_frame = 1
    # detector.process_video('project_video.mp4')

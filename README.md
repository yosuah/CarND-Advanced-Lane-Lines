# Advanced Lane Finding Project (using traditional CV methods)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Result

I created a traditional CV-based solution that works reasonably well on the test videos, 
but obviously could not scale to more complex scenarios. 

- Description of the pipeline (the write-up) can be found in [AdvancedLaneLines.ipynb](AdvancedLaneLines.ipynb).
    - Open [AdvancedLaneLines.html](AdvancedLaneLines.html) or [AdvancedLaneLines.md](AdvancedLaneLines.md) 
      if you can not render the notebook.
- The actual implementation is in [lane_detector.py](lane_detector.py)
- Sample debug output images are in [output_images](output_images)

Sample output gif:
![Sample output from project video](output_videos/project_video.gif)

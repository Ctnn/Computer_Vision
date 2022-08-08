import cv2
from matplotlib import pyplot as plt
import numpy as np


def imshow(text="Text",image=None,size=14):
    w,h = image.shape[:2]
    aspect_ratio=w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(text)
    plt.show()


# Load video stream, short clip
# cap = cv2.VideoCapture('walking_short_clip.mp4')

# Load video stream, long clip
cap = cv2.VideoCapture('../videos/walking.avi')

# Get the height and width of the frame (required to be an interger)
width = int(cap.get(3))
height = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('optical_flow_walking.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

# Set parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Set parameters for lucas kanade optical flow
lucas_kanade_params = dict(winSize=(15, 15),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
# Used to create our trails for object movement in the image
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Find inital corner locations
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(prev_frame)

while (1):
    ret, frame = cap.read()

    if ret == True:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                               frame_gray,
                                                               prev_corners,
                                                               None,
                                                               **lucas_kanade_params)

        # Select and store good points
        good_new = new_corners[status == 1]
        good_old = prev_corners[status == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        # Save Video
        out.write(img)
        # Show Optical Flow
        # imshow('Optical Flow - Lucas-Kanade',img)

        # Now update the previous frame and previous points
        prev_gray = frame_gray.copy()
        prev_corners = good_new.reshape(-1, 1, 2)

    else:
        break

cap.release()
out.release()

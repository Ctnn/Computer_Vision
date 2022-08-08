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


cap=cv2.VideoCapture('../videos/walking.mp4')

# Get the height and width of the frame (required to be an interger)
w = int(cap.get(3))
h = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('walking_output_GM.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()

# Loop once video is successfully loaded
while True:

    ret, frame = cap.read()

    if ret:
        # Apply background subtractor to get our foreground mask
        foreground_mask = foreground_background.apply(frame)
        out.write(foreground_mask)
        imshow("Foreground Mask", foreground_mask)
    else:
        break

cap.release()
out.release()
import cv2
from matplotlib import pyplot as plt
import numpy as np


def imshow(text="Text", image=None, size=14):
    w, h = image.shape[:2]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)
    plt.show()


# Create our video capturing object
cap = cv2.VideoCapture('../videos/walking.mp4')

# Get the height and width of the frame (required to be an interfer)
w = int(cap.get(3))
h = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'walking_output.avi' file.
out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

body_detector = cv2.CascadeClassifier('../Haarcascades/haarcascade_fullbody.xml')

# Loop once video is successfully loaded
while (True):

    ret, frame = cap.read()
    # If success
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our body classifier
        bodies = body_detector.detectMultiScale(gray, 1.2, 3)

        # Extract bounding boxes for any bodies identified
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Write the frame into the file 'output.avi'
        out.write(frame)
    else:
        break

cap.release()
out.release()

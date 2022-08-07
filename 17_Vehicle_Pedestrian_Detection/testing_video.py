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


# Create our video capturing object
cap = cv2.VideoCapture('../videos/walking.mp4')

# Read first frame
ret, frame = cap.read()

# Ret is True if successfully read
if ret:

    # Load our body classifier
    body_classifier = cv2.CascadeClassifier('../Haarcascades/haarcascade_fullbody.xml')
    # Grayscale our image for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imshow("empty", gray)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Release our video capture
cap.release()
imshow("Pedestrian Detector", frame)

import cv2
import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import pyplot as plt

def imshow(title="Image", image=None, size=16):
    w, h = image.shape[:2]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

#Our Referance Image

template = cv2.imread('../publicSource/images/4star.jpg',0) #!!!! BECAREFUL THIS VALUE
imshow("Template Image",template)

#Load the target image with the shapes we're trying to match !!

target=cv2.imread('../publicSource/images/shapestomatch.jpg')
imshow("Target Image",target)
target_gray=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)



# Threshold both images first before using cv2.findCounters

ret, thresh1 = cv2.threshold(template, 127, 255, 0)

ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)


#Find contours in template

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

print(hierarchy)
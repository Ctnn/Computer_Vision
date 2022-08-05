import cv2
from matplotlib import pyplot as plt


def imshow(title="Image", image=None, size=16):
    w, h = image.shape[:2]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


image = cv2.imread('../publicSource/images/bunchofshapes.jpg')
imshow("Input Image", image)

# First You Need Gray Image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny Edge

edged = cv2.Canny(gray, 50, 200)
imshow('Canny Edges', edged)

#Find Contours and print how many were found

contours,hierarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("Contours of Numbers =",len(contours))

# Draw contours Image

cv2.drawContours(image,contours,-1,(0,255,0),3)
imshow("All Contours",image)

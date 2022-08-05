import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow(title="Image", image=None, size=16):
    w, h = image.shape[:2]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def get_contour_areas(contours):
    """returns the areas of all contours as list kind of a converter function"""

    all_areas=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas






gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 50, 200)
imshow('Canny Edges', edged)

contours,hierarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("Contours of Numbers =",len(contours))

cv2.drawContours(image,contours,-1,(0,255,0),3)
imshow("All Contours",image)


# Let's print the areas of the contours before sorting
print("Contor Areas before sorting...")
print(get_contour_areas(contours))

# Sort contours large to small by area
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

print("Contor Areas after sorting...")
print(get_contour_areas(sorted_contours))


# Iterate over our contours and draw one at a time
for (i,c) in enumerate(sorted_contours):
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.drawContours(image, [c], -1, (255,0,0), 3)

imshow('Contours by area', image)




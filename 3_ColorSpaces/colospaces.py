import cv2
import numpy as np
from matplotlib import pyplot as plt


def imshow(title="image", image=None, size=10):
    w,h=image.shape[0], image.shape[1]
    aspect_ratio=w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Use cv2.split to get each color space separately
image= cv2.imread('../publicSource/images/castara.jpeg')
B, G, R = cv2.split(image)

print(B.shape)
print(G.shape)
print(R.shape)

zeros = np.zeros(image.shape[:2], dtype = "uint8")

imshow("Red", cv2.merge([zeros, zeros, R]))
imshow("Green", cv2.merge([zeros, G, zeros]))
imshow("Blue", cv2.merge([B, zeros, zeros]))

# Let's re-make the original image,
merged = cv2.merge([B, G, R])
imshow("Merged", merged)

# Let's amplify the blue color
merged = cv2.merge([B+200, G, R])
imshow("Blue Boost", merged)

# Reload our image
image = cv2.imread('./images/castara.jpeg')

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
imshow('HSV', hsv_image)

# This looks fine beacuse RGB based

plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
plt.show()

# Switching back to viewing the RGB representation
imshow("Hue", hsv_image[:, :, 0])
imshow("Saturation", hsv_image[:, :, 1])
imshow("Value", hsv_image[:, :, 2])
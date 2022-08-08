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


def mse(image1, image2):
    # Images must be of the same dimension
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])

    return error

def compare(image1, image2):
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  print('MSE = {:.2f}'.format(mse(image1, image2)))

fireworks1 = cv2.imread('../publicSource/images/fireworks.jpeg')
fireworks2 = cv2.imread('../publicSource/images/fireworks2.jpeg')

M = np.ones(fireworks1.shape, dtype = "uint8") * 100
fireworks1b = cv2.add(fireworks1, M)

imshow("fireworks 1", fireworks1)
imshow("Increasing Brightness", fireworks1b)
imshow("fireworks 2", fireworks2)

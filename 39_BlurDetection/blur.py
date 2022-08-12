import cv2
from matplotlib import pyplot as plt

# Define our imshow function
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Load our input image
image = cv2.imread('./images/liberty.jpeg')
imshow("Original Image", image)

blur_1 = cv2.GaussianBlur(image, (5,5), 0)
imshow('Blurred Image 1', blur_1)

blur_2 = cv2.GaussianBlur(image, (9,9), 0)
imshow('Blurred Image 2', blur_2)

blur_3 = cv2.GaussianBlur(image, (13,13), 0)
imshow('Blurred Image 3', blur_3)




def getBlurScore(image):
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return cv2.Laplacian(image, cv2.CV_64F).var()



print("Blur Score = {}".format(getBlurScore(image)))
print("Blur Score = {}".format(getBlurScore(blur_1)))
print("Blur Score = {}".format(getBlurScore(blur_2)))
print("Blur Score = {}".format(getBlurScore(blur_3)))
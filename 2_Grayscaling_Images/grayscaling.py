import cv2
from matplotlib import pyplot as plt

# Define our imshow function

def imshow(title="Image",image=None,size=10):
    width,height=image.shape[0],image.shape[1]
    aspect_ratio=width/height #En/Boy Oranı
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image= cv2.imread('../publicSource/images/castara.jpeg')

# We use cvtColor, to convert to grayscale
# It takes 2 arguments, the first being the input image
# The second being the color space conversion code
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imshow("Converted to Grayscale", gray_image)

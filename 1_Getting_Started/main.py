# Load an image using 'imread' specifying the path to image
import cv2

image = cv2.imread('./1_Getting_Started/source/demoImage.png')


#Show The Image with matplotlib

from matplotlib import pyplot as plt

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

plt.show()

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
plt.show()



#Let's create a simple function to make displaying our images simpler and easier

def imshow(title="", image= None):
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


imshow("This is a Image",image)

# Simply use 'imwrite' specificing the file name and the image to be saved
cv2.imwrite('output.jpg',image)


import numpy as np

print('Height of Image: {} pixels'.format(int(image.shape[0])))
print('Width of Image: {} pixels'.format(int(image.shape[1])))
print('Depth of Image: {} colors components'.format(int(image.shape[2])))
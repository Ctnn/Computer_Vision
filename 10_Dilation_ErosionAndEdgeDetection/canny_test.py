import cv2
import numpy as np
from matplotlib import pyplot as plt

camera=cv2.VideoCapture(0)

while True:
    ret,image=camera.read()
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_blue=np.array([100,60,60])
    upper_blue=np.array([140,255,255])

mask= cv2.inRange(hsv,lower_blue,upper_blue)
lastImage=cv2.bitwise_and(image,image,mask=mask)
cv2.imshow("Original",image)
cv2.imshow("Mask Image",mask)
cv2.imshow("Last Image",lastImage)

if cv2.waitKey(25) & 0xFF ==ord('q'):
    breakpoint()

camera.release()
cv2.destroyAllWindows()

edges=cv2.Canny(image,100,200)
cv2.imshow("Canny",edges)






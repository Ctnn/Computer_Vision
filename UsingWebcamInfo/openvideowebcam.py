import cv2
import numpy as np

# Initialize webcam, cap is the object provided by VideoCapture
# '0' its own embedded camera
# '1' its usb connection
cap = cv2.VideoCapture(0)

while True:
    # It contains a boolean indicating if it was sucessful (ret)
    # It also contains the images collected from the webcam (frame)
    ret, frame = cap.read()

    cv2.imshow('Our Webcam Video', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('../text_detection/testImage.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


'''
# Detecting text in an image
hImg, wImg = img.shape[:2]
boxes = pytesseract.image_to_data(img)
for x,b in enumerate(boxes.splitlines()):
  if x!=0:
        b = b.split(' ')  # split the string into list
        print(b)
        if len(b)==12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x,y), (w+x ,h+y), (0, 255, 0), 3)
            cv2.putText(img,b[11],(x, hImg-y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2)
'''# Detecting text in an image
hImg, wImg = img.shape[:2]
cong=r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_data(img,config=cong)
for x,b in enumerate(boxes.splitlines()):
  if x!=0:
        b = b.split()  # split the string into list
        print(b)
        if len(b)==12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x,y), (w+x ,h+y), (0, 255, 0), 3)
            cv2.putText(img,b[11],(x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(50, 50,255),2)


cv2.imshow('image', img)
cv2.waitKey(0)

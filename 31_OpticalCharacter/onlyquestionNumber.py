import pytesseract
from pytesseract import Output
import cv2
from matplotlib import pyplot as plt
import re
import PIL.Image
"""
0 Yalnızca yönlendirme ve yazı algılama (OSD).
1 OSD ile otomatik sayfa segmentasyonu.
2 OSD veya OKT olmadan otomatik sayfa segmentasyonu.
3 OSD olmadan tam otomatik sayfa segmentasyonu (öntanımlı)
4 Değişken boyutlarda metinlerden oluştan tek bir sütun varsay.
5 Dikey olarak hizalanmış bir metin bloğu varsay.
6 Tek bir metin bloğu varsay.
7 Resmi tek bir metin satırı varmış gibi ele al.
8 Resmi tek bir sözcük varmış gibi ele al.
9 Resmi bir daire içinde tek bir sözcük varmış gibi ele al.
10 Resmi tek karakter varmış gibi ele al.
11 Aralıklı metin. Belirli bir sıra olmaksızın mümkün olduğunca çok metin bul.
12 OSD'li ayrık metin.
13 Ham satır. Tesseract'a özgü hack'leri atlayarak resmi tek bir metin satırı olarak ele al.
"""



img= cv2.imread('../31_OpticalCharacter/data/page1.png')

email_pattern = '\S+.\S+'


img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

custom_config = r'--oem 3 --psm 6 outputbase digits'


d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
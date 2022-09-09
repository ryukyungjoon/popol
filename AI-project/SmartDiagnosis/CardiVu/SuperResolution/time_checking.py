import numpy as np
import time
from PIL import Image
import cv2
from ISR.models import RRDN, RDN

path = r'E:\ryu_pythonProject\2. Cardivu-A\202106/'
file = "김환진_음주전_2021-05-17-181830-0000_L.avi"
img = cv2.imread(r'E:\ryu_pythonProject\7. Super Resolution\data/test_img3.png')
vidcap = cv2.VideoCapture(path + file)
frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)
# rdn = RRDN(weights='gans')
rdn = RDN(weights='psnr-small')
# rdn = RDN(weights='psnr-large')
# rdn = RDN(weights='noise-cancel')
i = 0
start = time.time()
while i < 100:
    ret, frame = vidcap.read()
    if not ret:
        break
    rdn.predict(frame)
    cv2.waitKey(100)
    print(i)
    i += 1
    exit(0)

end = time.time()

print(f' time:{end-start}초')

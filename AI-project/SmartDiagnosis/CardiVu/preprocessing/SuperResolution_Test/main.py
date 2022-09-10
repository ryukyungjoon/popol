import numpy as np
import time
from PIL import Image
import cv2
from ISR.models import RRDN, RDN

path = r'E:\ryu_pythonProject\2. Cardivu-A\202106/'
file = "김환진_음주전_2021-05-17-181830-0000_L.avi"
img = cv2.imread(r'E:\ryu_pythonProject\7. Super Resolution\data/test_img3.png')
vidcap = cv2.VideoCapture(path + file)
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
# rdn = RRDN(weights='gans')
# rdn = RDN(weights='psnr-small')
# rdn = RDN(weights='psnr-large')
rdn = RDN(weights='noise-cancel')
while True:
    ret, frame = vidcap.read()
    if not ret:
        break
    resize_frame = cv2.resize(frame, (180, 180))

    lr_img = np.array(resize_frame)
    print(np.shape(resize_frame))
    start = time.time()
    sr_img = rdn.predict(frame)
    end = time.time()
    Image.fromarray(sr_img)
    print('time : ', end - start)
    # cv2.imshow('origin_image', frame)
    # cv2.imshow('low_resolution', lr_img)
    # cv2.imshow('high_resolution', sr_img)


exit(0)
lr_img = np.array(img)

rdn = RRDN(weights='gans')
sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img)

lr_img22 = cv2.resize(lr_img, (91, 91))
cv2.imshow('low22_resolution', lr_img22)
cv2.imshow('low_resolution', lr_img)
cv2.imshow('high_resolution', sr_img)
cv2.imwrite(r'E:\ryu_pythonProject\7. Super Resolution\data/[SR_RRDN(gans)]test_img3.png', sr_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
from ISR import train
# train.Trainer(generator=)

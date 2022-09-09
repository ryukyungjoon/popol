"""

IRIS 90x90 Video to 180x180 [Video & Image]
"""

import numpy as np
import cv2
import os
from ISR.models import RDN

rdn = RDN(weights='psnr-small')

img = cv2.imread(r"D:\ryu_pythonProject\7. Super Resolution\data/meerkat.png")

sr_img = rdn.predict(img)

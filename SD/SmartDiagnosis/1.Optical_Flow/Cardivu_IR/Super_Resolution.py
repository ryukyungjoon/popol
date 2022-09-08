import cv2
import time

import main
import param

class SuperResolution:
    def __init__(self, resize_X, resize_Y):
        self.resize = (resize_X, resize_Y)

    def pred(self, low_img_L, low_img_R):
        high_L = main.rdn.predict(low_img_L)
        high_R = main.rdn.predict(low_img_R)

        reSize_sr_op_L = cv2.resize(high_L, self.resize, cv2.INTER_LINEAR)
        reSize_sr_op_R = cv2.resize(high_R, self.resize, cv2.INTER_LINEAR)

        param.sr_L.append(reSize_sr_op_L)
        param.sr_R.append(reSize_sr_op_R)

        return reSize_sr_op_L, reSize_sr_op_R

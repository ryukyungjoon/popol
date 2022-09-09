import sys
import numpy as np
from ISR.models import RDN

from PySide2.QtWidgets import QApplication

import app
import param

rdn = RDN(weights='psnr-small')

## parameter_init
param.video_save = True
param.resize_X, param.resize_Y = 180, 180
param.save_path = r"E:\ryu_pythonProject\0. Data\HRV\IR\Save_Results/"
mask_area = ['mask1', 'mask2', 'mask1-2', 'mask2-2']
param.mask_loc = mask_area[2]

if __name__ == '__main__':
    param.ap = QApplication(sys.argv)
    mainWindow = app.MainWindow()
    mainWindow.show()
    sys.exit(param.ap.exec_())

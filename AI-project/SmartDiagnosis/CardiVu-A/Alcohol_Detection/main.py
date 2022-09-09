import sys
import numpy as np
from ISR.models import RDN

from PySide2.QtWidgets import QApplication

import app
import param

# Super Resolution init
rdn = RDN(weights='psnr-small')

## parameter_init
param.result_save_path = r"D:\ryu_pythonProject\0. Data\CardiVu-A\Eye_blinking_Task\[scenario_1]\res_save\0916/"
param.pupil_size_L = []
param.pupil_size_R = []
param.out_frame_L = []
param.out_frame_R = []
param.posX1, param.posY1, param.posX2, param.posY2 = 0, 0, 0, 0
param.g_m_w = 140
param.vis_frame = True

if __name__ == '__main__':
    param.application = QApplication(sys.argv)
    mainWindow = app.MainWindow()
    mainWindow.show()
    sys.exit(param.application.exec_())

from pupil_tracking import pupil_tracking as pp

import os
import numpy as np
import cv2

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QMainWindow, QFileDialog, QPushButton, QDialog

import param


def mouse_callback(event, flags):
    # 마우스 휠을 움직인 경우, 영역 크기 설정
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            param.g_m_w = param.g_m_w + 2
            print("마우스 휠을 위로 움직:", param.g_m_w)
        elif flags < 0:
            param.g_m_w = param.g_m_w - 2
            print("마우스 휠을 아래로 움직:", param.g_m_w)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)

        # 윈도우 설정
        self.setGeometry(100, 100, 500, 200)  # x, y, w, h
        self.setWindowTitle('CardiVu_Video Analysis')

        # QButton 위젯 생성
        self.button_folder = QPushButton('Folder Open', self)
        self.button_folder.clicked.connect(self.foler_open)
        self.button_folder.setGeometry(30, 30, 100, 30)

        # QButton 위젯 생성
        self.button_file = QPushButton('File Open', self)
        self.button_file.clicked.connect(self.file_open)
        self.button_file.setGeometry(140, 30, 100, 30)

        # QDialog 설정
        self.dialog = QDialog()

    def foler_open(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        file_list = [_ for _ in os.listdir(path) if _.endswith(r".mp4")]

        ## file path & name
        fname = []
        for i in range(0, len(file_list)):
            file_path = path + "/" + file_list[i]
            fname.append(file_path)
        pupil = pp()
        self.timer.timeout.connect(pupil.update_video(fname))
        self.timer.start(33)

    # 버튼 이벤트 함수
    def file_open(self):
        pupil = pp()
        fname = QFileDialog.getOpenFileName(self, "Video File Open", "..\\",
                                            "mp4 Files (*.mp4);;avi Files (*.avi);;wmv Files (*.wmv);;All Files (*.*)")

        fname = np.delete(fname, 1)

        self.timer.timeout.connect(pupil.update_video(fname))
        self.timer.start(33)

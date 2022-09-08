import datetime
import os
import sys

import cv2
import numpy as np
import pandas as pd

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QMainWindow, QFileDialog, QPushButton, QDialog, QApplication


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = None

        # 윈도우 설정
        self.setGeometry(100, 100, 500, 200)  # x, y, w, h
        self.setWindowTitle('Video Sync')

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
        global fname

        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        file_list = [_ for _ in os.listdir(path) if _.endswith(r".mp4")]

        print(file_list)
        fname = []

        for i in range(0, len(file_list)):
            file_path = path + "/" + file_list[i]
            fname.append(file_path)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.VideoSync(fname))
        self.timer.start(33)

    # 버튼 이벤트 함수
    def file_open(self):
        global fname

        fname = QFileDialog.getOpenFileName(self, "Video File Open", "..\\",
                                            "mp4 Files (*.mp4);;avi Files (*.avi);;wmv Files (*.wmv);;All Files (*.*)")

        fname = np.delete(fname, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.VideoSync(fname))
        self.timer.start(33)

    def VideoSync(self, fname):
        global fpBPM5, fpBPM15, fpBPM30
        global fpHRV60, fpHRV120, fpHRV180

        # Read PPG
        # f = open(r"D:\02_Research\SD 연구\(202107) HRV\Raw Data/Timesync_RGB_info.txt", 'r')
        # time = []
        # for line in f.readlines():
        #     _time = str(line.replace('\n', ''))[5:7]
        #     time.append(int(_time))
        # f.close()

        for i in range(0, len(fname)):
            num_f = 0
            print("[", i, "]", "fname: ", fname[i])

            ###### 출력 폴더 생성하여 출력하기
            fnamefname = [j for j, value in enumerate(fname[i]) if value == "/"]
            output = "/output"
            path_1 = fname[i][:fnamefname[-1]]
            path_2 = fname[i][fnamefname[-1]:]
            path_2 = path_2[1:]
            output_path = path_1 + output

            # 디렉토리에 output 폴더 존재하는지 확인
            if os.path.exists(output_path):
                print("기존 폴더가 존재합니다")
            else:  # 출력 폴더 생성
                os.mkdir(output_path)
            output_path = path_1 + output + "/[Timesync]" + path_2

            cap = cv2.VideoCapture(fname[i])
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            print('fps:', fps)
            print('frame_width:', frame_width)
            print('frame_height:', frame_height)
            print('Total Frame:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            frame_cnt = 0

            print(output_path)

            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                                  fps,  # fps
                                  (frame_width, frame_height))

            startFrame = 0
            endFrame = 73 * fps
            print('startFrame:', startFrame, '\tendFrame:', endFrame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
            while True:
                ret, frame = cap.read()

                if not ret:
                    out.release()
                    cap.release()
                    break

                # Timesync 후 260초 동안의 영상 저장
                if frame_cnt > (endFrame - startFrame):
                    out.release()
                    cap.release()
                    break

                frame_cnt += 1
                print(frame_cnt)

                cv2.imshow("Video", frame)
                out.write(frame)
                if cv2.waitKey(1) and 0xFF == ord('q'):
                    print("종료")
                    sys.exit(app.exec_())

        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

from pupil_center import pupil_center
from optical_flow import Optical_Flow as Optflow
import param
import main
import saver

import os
import cv2
import sys
import time

arr_g_m_w = [106, 102, 94, 106, 98, 104, 104, 94, 96, 106]
arr_threshold = [63, 70, 45, 45, 45, 61, 59, 68, 61, 42]
arr_a1 = [327, 306, 376, 303, 305, 328, 302, 330, 350, 344]
arr_a2 = [286, 302, 414, 433, 438, 374, 268, 360, 410, 396]
arr_a3 = [1086, 1026, 1106, 1058, 1030, 1142, 1060, 1090, 1038, 1126]
arr_a4 = [482, 446, 632, 548, 570, 512, 420, 478, 536, 544]


class pupil_tracking:
    def __init__(self):
        self.g_mx, self.g_my = 0, 0
        self.g_mx2, self.g_my2 = 0, 0

    def update_video(self, fname):
        for i in range(6, len(fname)):
            num_video = i
            print("[", i, "]", "fname: ", fname[i])

            ## 구분자 "/"를 기준으로 나눔
            file_info = fname[i].split('/')
            file_name = file_info[-1]
            param.file_name = file_name[:-4]

            # 디렉토리에 output 폴더 존재하는지 확인
            if os.path.exists(param.save_path):
                print("기존 폴더가 존재합니다")
            else:  # 출력 폴더 생성
                os.mkdir(param.save_path)

            cap = cv2.VideoCapture(fname[i])
            fps = cap.get(cv2.CAP_PROP_FPS)

            fppupilL = param.save_path + param.file_name + "_PupilL.txt"
            fppupilR = param.save_path + param.file_name + "_PupilR.txt"
            irisL = param.save_path + param.file_name + "_L.mp4"
            irisR = param.save_path + param.file_name + "_R.mp4"

            param.g_m_w = arr_g_m_w[i]
            threshold = 255 - arr_threshold[i]
            a1 = arr_a1[i]
            a2 = arr_a2[i]
            a3 = arr_a3[i]
            a4 = arr_a4[i]

            if param.video_save:
                param.sr_L = []
                param.sr_R = []
                param.out_video_L = cv2.VideoWriter(irisL,
                                                    cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                                                    fps,  # fps
                                                    (180, 180))

                param.out_video_R = cv2.VideoWriter(irisR,
                                                    cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                                                    fps,  # fps
                                                    (180, 180))
            param.MAG_L = []
            param.ANG_L = []
            param.MOT_L = []

            param.MAG_R = []
            param.ANG_R = []
            param.MOT_R = []

            num_frame = 0
            while True:
                start = time.time()
                ret, image = cap.read()
                print(num_frame)

                if not ret:
                    print("프레임이 없습니다.")
                    s = saver.Save()
                    s.save_npz(LR="L")
                    s.save_npz(LR="R")
                    s.save_video()
                    break

                param.pupil_on = False
                image = image[a2:a4, a1:a3]

                if not param.pupil_on:
                    pupil_center(image, threshold)
                    try:
                        self.g_mx = int(param.pupilR_X + param.pupilR_W / 2)
                        self.g_my = int(param.pupilR_Y + param.pupilR_W / 2)
                        self.g_mx2 = int(param.pupilL_X + param.pupilL_W / 2)
                        self.g_my2 = int(param.pupilL_Y + param.pupilL_W / 2)

                    except AttributeError as e:
                        print(f'{e}')
                        continue

                self.update_image(image)
                self.processImage(image, num_frame)
                end = time.time()
                print(f'{end - start}')
                if not param.pupil_on:
                    with open(fppupilL, 'a') as f:
                        f.write("frame: " + str(num_video) + "\t" + str(0) + "\t"
                                + str(0) + "\t" + str(0) + "\n")

                    with open(fppupilR, 'a') as f:
                        f.write("frame: " + str(num_video) + "\t" + str(0) + "\t"
                                + str(0) + "\t" + str(0) + "\n")
                else:
                    with open(fppupilL, 'a') as f:
                        f.write("frame: " + str(num_video) + "\t" + str(param.pupilL_X + param.pupilL_W / 2) + "\t"
                                + str(param.pupilL_Y + param.pupilL_W / 2) + "\t" + str(param.pupilL_W / 2) + "\n")

                    with open(fppupilR, 'a') as f:
                        f.write("frame: " + str(num_video) + "\t" + str(param.pupilR_X + param.pupilR_W / 2) + "\t"
                                + str(param.pupilR_Y + param.pupilR_W / 2) + "\t" + str(param.pupilR_W / 2) + "\n")
                f.close()

                num_frame += 1

                if cv2.waitKey(1) == 27:
                    print("ESC: 프로그램을 종료합니다.")
                    sys.exit(param.ap.exec_())

        sys.exit(param.ap.exec_())

    def update_image(self, img):
        paint_img = img.copy()

        paint_img = cv2.circle(paint_img, (self.g_mx, self.g_my), 2, (0, 255, 0), 2)
        paint_img = cv2.rectangle(paint_img, (self.g_mx - int(param.g_m_w / 2), self.g_my - int(param.g_m_w / 2)),
                                  (self.g_mx + int(param.g_m_w / 2), self.g_my + int(param.g_m_w / 2)), (0, 255, 0), 2)

        paint_img = cv2.circle(paint_img, (self.g_mx2, self.g_my2), 2, (0, 255, 0), 2)
        paint_img = cv2.rectangle(paint_img, (self.g_mx2 - int(param.g_m_w / 2), self.g_my2 - int(param.g_m_w / 2)),
                                  (self.g_mx2 + int(param.g_m_w / 2), self.g_my2 + int(param.g_m_w / 2)), (0, 255, 0),
                                  2)

        if param.pupil_on:
            cv2.circle(paint_img, (int(param.pupilR_X + param.pupilR_W / 2), int(param.pupilR_Y + param.pupilR_W / 2)),
                       2, (200, 200, 200), -1,
                       8, 0)
            cv2.circle(paint_img, (int(param.pupilR_X + param.pupilR_W / 2), int(param.pupilR_Y + param.pupilR_W / 2)),
                       int(param.pupilR_W / 2),
                       (200, 200, 200), 2, 8, 0)

            cv2.circle(paint_img, (int(param.pupilL_X + param.pupilL_W / 2), int(param.pupilL_Y + param.pupilL_W / 2)),
                       2, (200, 200, 200), -1,
                       8, 0)
            cv2.circle(paint_img, (int(param.pupilL_X + param.pupilL_W / 2), int(param.pupilL_Y + param.pupilL_W / 2)),
                       int(param.pupilL_W / 2),
                       (200, 200, 200), 2, 8, 0)

        # cv2.imshow("img", img)
        # cv2.imshow("paint_img", paint_img)

    def processImage(self, img, num_frame):
        # print(img)
        op_R = img[(self.g_my - int(param.g_m_w / 2)):(self.g_my + int(param.g_m_w / 2)),
               (self.g_mx - int(param.g_m_w / 2)):(self.g_mx + int(param.g_m_w / 2))]
        op_L = img[(self.g_my2 - int(param.g_m_w / 2)):(self.g_my2 + int(param.g_m_w / 2)),
               (self.g_mx2 - int(param.g_m_w / 2)):(self.g_mx2 + int(param.g_m_w / 2))]
        op_L = cv2.flip(op_L, 1)

        # Optical Flow
        Optflow(op_L, op_R, num_frame)

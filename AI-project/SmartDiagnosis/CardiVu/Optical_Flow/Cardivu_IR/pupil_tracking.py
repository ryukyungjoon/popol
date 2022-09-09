from pupil_center import pupil_center
from optical_flow import Optical_Flow as Optflow
import param
import main
import saver

import os
import cv2
import sys
import time

arr_g_m_w = [106, 102, 94, 106, 98, 104, 104, 94, 96, 106, 100, 92, 106, 92, 106, 108, 108, 104, 112, 116, 112, 112, 102, 102, 92, 100, 96, 92, 104, 96, 92, 94, 100, 94, 96, 104, 100, 100, 104, 106, 100, 94, 104, 98, 106, 96, 104, 98, 94, 100, 100, 96, 98, 106, 96, 94, 104, 96, 98, 98, 98, 100, 90, 88, 96, 92, 92, 100, 92, 94, 100, 94, 94, 102, 92, 106, 100, 100, 94, 100, 96, 98, 98, 88, 88, 88, 94, 90, 106]
arr_threshold = [63, 70, 45, 45, 45, 61, 59, 68, 61, 42, 57, 54, 55, 36, 58, 51, 51, 52, 40, 53, 66, 63, 65, 65, 65, 48, 58, 59, 53, 64, 79, 75, 40, 64, 40, 57, 64, 55, 66, 68, 67, 72, 51, 56, 50, 62, 62, 53, 59, 48, 71, 54, 70, 60, 65, 70, 57, 58, 66, 57, 55, 40, 58, 76, 44, 66, 57, 49, 80, 71, 70, 51, 51, 45, 48, 37, 59, 64, 62, 46, 54, 58, 51, 76, 68, 65, 44, 52, 34]
arr_a1 = [312, 291, 361, 288, 290, 313, 287, 315, 335, 329, 325, 353, 261, 353, 285, 207, 305, 345, 341, 349, 269, 347, 315, 339, 323, 361, 367, 363, 337, 279, 379, 357, 399, 365, 405, 335, 321, 359, 315, 293, 273, 361, 335, 365, 299, 327, 295, 311, 323, 353, 369, 387, 317, 307, 303, 373, 335, 355, 355, 373, 273, 335, 399, 321, 371, 409, 385, 379, 423, 405, 329, 413, 375, 345, 373, 365, 345, 313, 385, 369, 345, 387, 383, 389, 349, 399, 333, 295, 365]
arr_a2 = [271, 287, 399, 418, 423, 359, 253, 345, 395, 381, 401, 417, 265, 423, 315, 201, 323, 313, 285, 279, 225, 269, 199, 205, 269, 293, 337, 307, 223, 273, 297, 321, 291, 313, 293, 197, 245, 341, 305, 219, 269, 333, 311, 315, 155, 281, 185, 279, 305, 265, 267, 269, 285, 165, 219, 285, 151, 317, 305, 297, 225, 259, 301, 281, 247, 305, 335, 279, 301, 281, 133, 313, 257, 179, 351, 269, 163, 255, 269, 265, 343, 305, 269, 309, 365, 291, 365, 233, 269]
arr_a3 = [1101, 1041, 1121, 1073, 1045, 1157, 1075, 1105, 1053, 1141, 1089, 1105, 1057, 1109, 1065, 1067, 1149, 1117, 1205, 1255, 1177, 1185, 1089, 1107, 1117, 1153, 1105, 1065, 1133, 1065, 1047, 1087, 1221, 1077, 1205, 1139, 1169, 1183, 1145, 1079, 1093, 1047, 1131, 1115, 1099, 1025, 1121, 1069, 1077, 1141, 1087, 1131, 1051, 1095, 1079, 1051, 1123, 1103, 1041, 1111, 1063, 1113, 1027, 1015, 1133, 1091, 1119, 1169, 1073, 1111, 1091, 1063, 1079, 1089, 1083, 1155, 1095, 1103, 1089, 1153, 1107, 1057, 1151, 1069, 1069, 1061, 1039, 1041, 1171]
arr_a4 = [497, 461, 647, 563, 585, 527, 435, 493, 551, 559, 615, 631, 451, 631, 463, 425, 515, 495, 473, 473, 419, 465, 395, 373, 407, 469, 505, 465, 383, 387, 437, 499, 495, 487, 495, 345, 375, 545, 153, 369, 399, 503, 465, 495, 295, 415, 331, 453, 447, 445, 423, 449, 463, 317, 451, 437, 331, 485, 423, 477, 357, 413, 427, 439, 415, 435, 505, 457, 463, 467, 315, 459, 455, 357, 503, 441, 325, 365, 439, 437, 479, 443, 433, 473, 487, 479, 499, 355, 445]

class pupil_tracking:
    def __init__(self):
        self.g_mx, self.g_my = 0, 0
        self.g_mx2, self.g_my2 = 0, 0

    def update_video(self, fname):
        for i in range(0, len(fname)):
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
            # param.MOT_L = []

            param.MAG_R = []
            param.ANG_R = []
            # param.MOT_R = []

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

import os
import cv2
import sys

import pandas as pd
from ISR.models import RDN
import mediapipe as mp

from FInd_pupil import Find_pupil_center, pupilSize
from processing import img_processing, data_processing
import param
import saver

pc = img_processing()
dp = data_processing()
save = saver.Save()

rdn = RDN(weights='psnr-small')

r1 = [380, 350, 280]
r2 = [720, 720, 720]
c1 = [250, 200, 200]
c2 = [1100, 1150, 1200]

th = [223, 223, 225]

class videoUpdate:
    def __init__(self, cap_mode=None):
        """

        :param cap_mode: "real_time" or "video_upload"
        """
        self.cap_mode = cap_mode
        self.threshold = 230

    def video_update(self, fname):
        if self.cap_mode == "real_time":
            print("Recording")

            ## 구분자 "/"를 기준으로 나눔
            file_info = fname[i].split('/')
            file_name = file_info[-1]
            param.file_name = file_name[:-4]

            # 디렉토리에 output 폴더 존재하는지 확인
            if os.path.exists(param.result_save_path):
                print("기존 폴더가 존재합니다")
            else:  # 출력 폴더 생성
                os.mkdir(param.result_save_path)

            cap = cv2.VideoCapture(0)

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = int(cap.get(cv2.CAP_PROP_FPS))

            save.video_open(param.result_save_path + "fanem", param.result_save_path, fps=fps)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f'프레임이 없습니다.')
                    break
                try:
                    self.eye_tracking(frame)

                except AttributeError as e:
                    sys.stdout.write(f'{e}\n')
                    continue

        elif self.cap_mode == "video_upload":
            for i in range(2, len(fname)):
                print("[", i, "]", "fname: ", fname[i])

                ## 구분자 "/"를 기준으로 나눔
                file_info = fname[i].split('/')
                file_name = file_info[-1]
                param.file_name = file_name[:-4]

                param.pupil_size_L = []
                param.pupil_size_R = []
                param.out_frame_L = []
                param.out_frame_R = []

                # 디렉토리에 output 폴더 존재하는지 확인
                if os.path.exists(param.result_save_path):
                    print("기존 폴더가 존재합니다")
                else:  # 출력 폴더 생성
                    os.mkdir(param.result_save_path)

                cap = cv2.VideoCapture(fname[i])

                # while True:
                #     _, f0 = cap.read()
                #     self.threshold = pc.calcAdaptiveThreshold(f0)
                #     if self.threshold is not None:
                #         break

                param.isGet = True
                frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                iris_L = param.result_save_path + param.file_name + "_iris_L.mp4"
                iris_R = param.result_save_path + param.file_name + "_iris_R.mp4"

                save.video_open(iris_L, iris_R, fps=fps)
                frame_num = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print(f'프레임이 없습니다.')
                        ## List to DataFrame
                        pupil_size_L = pd.DataFrame(param.pupil_size_L)
                        pupil_size_R = pd.DataFrame(param.pupil_size_R)

                        # ## DataFrame to csv File
                        pupil_size_L.to_csv(param.result_save_path + param.file_name + "_L.csv")
                        pupil_size_R.to_csv(param.result_save_path + param.file_name + "_R.csv")

                        ## mp4 File
                        save.save_video(param.out_frame_L, param.out_frame_R)

                        ## Data Preprocessing
                        l_retrievers, l_shrink = data_processing().Feature_Extraction(pupil_size_L)
                        r_retrievers, r_shrink = data_processing().Feature_Extraction(pupil_size_R)
                        print(f'video file{param.file_name} 좌안 수축시간{l_shrink} 우안 수축시간{r_shrink}')
                        print(f'video file{param.file_name} 좌안 회복시간{l_retrievers} 우안 회복시간{r_retrievers}')

                        # save.save_file(l_tc, l_tr, LR='L')
                        # save.save_file(r_tc, r_tr, LR='R')
                        break
                    try:
                        # print(frame_num)
                        frame = frame[r1[i]:r2[i], c1[i]:c2[i]]
                        cv2.imshow('frame', frame)
                        self.eye_tracking(frame, frame_num)
                    except AttributeError as e:
                        sys.stdout.write(f'{e}\n')
                        continue
                    frame_num += 1
            else:
                print("capture mode is None")
            if cv2.waitKey(1) == 27:
                print("ESC: 프로그램을 종료합니다.")
                sys.exit(param.application.exec_())

        sys.exit(param.application.exec_())

    def eye_range_detection(self, frame):
        ## Face to glasses-line
        ## google media pipe, return value is eyebrow left&right eye brow's center:(x1, y1) end_point(x2, y2)
        ############
        height = 300
        ############

        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Initial Coordinates
            if param.isGet:
                param.posX1 = int(results.multi_face_landmarks[0].landmark[46].x * frame.shape[1])
                param.posY1 = int(results.multi_face_landmarks[0].landmark[46].y * frame.shape[0])
                param.posX2 = int(results.multi_face_landmarks[0].landmark[276].x * frame.shape[1])
                param.posY2 = int(results.multi_face_landmarks[0].landmark[276].y * frame.shape[0])
                param.isGet = False
            ## 눈썹 좌표 받아오기
            eye_img = frame[:height, param.posX1:param.posX2, :]
            if param.vis_frame:
                cv2.imshow('eye_img', eye_img)
                cv2.waitKey(1)

        return eye_img

    def split_eyes(self, img):
        op_R = img[(param.g_my - int(param.g_m_w / 2)):(param.g_my + int(param.g_m_w / 2)),
               (param.g_mx - int(param.g_m_w / 2)):(param.g_mx + int(param.g_m_w / 2))]
        op_L = img[(param.g_my2 - int(param.g_m_w / 2)):(param.g_my2 + int(param.g_m_w / 2)),
               (param.g_mx2 - int(param.g_m_w / 2)):(param.g_mx2 + int(param.g_m_w / 2))]
        op_L = cv2.flip(op_L, 1)
        if param.vis_frame:
            cv2.imshow('op_R', op_R)
            cv2.imshow('op_L', op_L)
            cv2.waitKey(1)
        return op_L, op_R

    def SuperResolution(self, left, right):
        sr_size = 180
        sr_l = rdn.predict(left)
        sr_r = rdn.predict(right)

        sr_l = cv2.resize(sr_l, (sr_size, sr_size))
        sr_r = cv2.resize(sr_r, (sr_size, sr_size))

        if param.vis_frame:
            cv2.imshow('sr_l', sr_l)
            cv2.imshow('sr_r', sr_r)
        return sr_l, sr_r

    def eye_tracking(self, frame, frame_num=0):
        # frame = self.eye_range_detection(frame)
        Find_pupil_center(frame, threshold=self.threshold)
        op_l, op_r = self.split_eyes(frame)
        sr_l, sr_r = self.SuperResolution(op_l, op_r)
        self.getpupilSize(sr_l, LR="L", frame_num=frame_num)
        self.getpupilSize(sr_r, LR="R", frame_num=frame_num)

    def getpupilSize(self, frame, LR=None, frame_num=0):
        mask_frame, lx_idx, ly_idx = pc.draw_circle_mask(frame, mask=True, mask_loc='mask1', LR=LR)
        if param.vis_frame:
            if LR == 'L':
                cv2.imshow('mask_img_l', mask_frame)
            elif LR == 'R':
                cv2.imshow('mask_img_r', mask_frame)
        pupilSize().pupilSize(mask_frame, threshold=self.threshold)

        if LR == "L":
            param.out_frame_L.append(frame)
            param.pupil_size_L.append(param.radian)
            # print(param.pupil_size_L)
        elif LR == "R":
            param.out_frame_R.append(frame)
            param.pupil_size_R.append(param.radian)
            # print(param.pupil_size_R)
        else:
            print(f'eye location is {LR}')

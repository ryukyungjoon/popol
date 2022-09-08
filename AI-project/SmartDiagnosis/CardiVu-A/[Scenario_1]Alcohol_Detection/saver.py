import numpy as np
import param
import cv2

class Save:
    def __init__(self):
        pass

    def video_open(self, iris_L_path, iris_r_path, fps=100):
        param.out_video_L = cv2.VideoWriter(iris_L_path,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps,  # fps
                                            (180, 180))

        param.out_video_R = cv2.VideoWriter(iris_r_path,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps,  # fps
                                            (180, 180))

    def save_video(self, sr_L, sr_R):
        for n in range(len(sr_L)):
            param.out_video_L.write(sr_L[n])
            param.out_video_R.write(sr_R[n])

    def save_file(self, tc, tr, LR=None):
        if LR == 'L':
            with open(param.result_save_path + param.file_name + "_L.txt", mode='w') as f:
                f.write(tc)
                f.write(tr)

        elif LR == 'R':
            with open(param.result_save_path + param.file_name + "_R.txt", mode='w') as f:
                f.write(tc)
                f.write(tr)

        else:
            print('Select eye position!')

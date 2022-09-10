import time

from Super_Resolution import SuperResolution
import param
import saver

import cv2
import numpy as np
import pandas as pd

from collections import deque

opt = cv2.optflow.createOptFlow_DeepFlow()


class Optical_Flow:
    def __init__(self, op_L, op_R, num_frame):

        self.mask_x_idx, self.mask_y_idx = None, None
        self.OF_x_idx, self.OF_y_idx = None, None
        self.num_frame = num_frame
        # self.img_buffer = deque(np.zeros(shape=(param.resize_X, param.resize_Y)), maxlen=2)
        SR = SuperResolution(param.resize_X, param.resize_Y)
        sr_op_L, sr_op_R = SR.pred(op_L, op_R)

        if num_frame == 0:
            param.prev_L, param.prev_R = None, None
            param.prev_L = cv2.cvtColor(sr_op_L, cv2.COLOR_BGR2GRAY)
            param.prev_R = cv2.cvtColor(sr_op_R, cv2.COLOR_BGR2GRAY)
        else:
            self.OF_Extraction(sr_op_L, sr_op_R)

    def OF_Extraction(self, sr_op_L, sr_op_R):
        nexts_L = sr_op_L
        nexts_R = sr_op_R

        nexts_L = cv2.cvtColor(nexts_L, cv2.COLOR_BGR2GRAY)
        nexts_R = cv2.cvtColor(nexts_R, cv2.COLOR_BGR2GRAY)

        ## 왼쪽 눈
        prev_L_buf = param.prev_L.copy()
        nexts_L_buf = nexts_L.copy()
        L_img1, L_img2 = self.circle_mask(prev_L_buf, nexts_L_buf, mask=True, mask_loc=param.mask_loc)
        self.flow_tracking(L_img1, L_img2, LR='L')

        ## 오른쪽 눈
        prev_R_buf = param.prev_R.copy()
        nexts_R_buf = nexts_R.copy()
        R_img1, R_img2 = self.circle_mask(prev_R_buf, nexts_R_buf, mask=True, mask_loc=param.mask_loc)
        self.flow_tracking(R_img1, R_img2, LR='R')

    def create_circular_mask(self, h, w, center=None, radius=None, location=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))

        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        if location == "inner":
            mask = dist_from_center >= radius  ## Boolean 값
            return mask

        elif location == "outer":
            mask = dist_from_center <= radius  ## Boolean 값
            return mask

    def circle_mask(self, img1, img2, mask=True, mask_loc=None):
        if mask:
            inner_radius, outer_radius = None, None
            ## mask1 : 29, 56
            if mask_loc == 'mask1':
                inner_radius = int(img1.shape[0] / 2 * 0.29)
                outer_radius = int(img1.shape[0] / 2 * 0.56)

            ## mask2 : 56, 90
            elif mask_loc == 'mask2':
                inner_radius = int(img1.shape[0] / 2 * 0.56)
                outer_radius = int(img1.shape[0] / 2 * 0.90)

            ## mask2 : 38, 56
            elif mask_loc == 'mask1-2':
                inner_radius = int(img1.shape[0] / 2 * 0.38)
                outer_radius = int(img1.shape[0] / 2 * 0.56)

            ## mask2 : 63, 87
            elif mask_loc == 'mask2-2':
                inner_radius = int(img1.shape[0] / 2 * 0.63)
                outer_radius = int(img1.shape[0] / 2 * 0.87)

            else:
                print(f'mask location :{mask_loc}')
                exit(0)

            mask_inner = self.create_circular_mask(img1.shape[0], img1.shape[1], [img1.shape[0] / 2, img1.shape[1] / 2],
                                                   inner_radius,
                                                   location='inner')

            mask_outer = self.create_circular_mask(img1.shape[0], img1.shape[1], [img1.shape[0] / 2, img1.shape[1] / 2],
                                                   outer_radius,
                                                   location='outer')
            img1[~mask_inner] = 0
            img1[~mask_outer] = 0
            img2[~mask_inner] = 0
            img2[~mask_outer] = 0

            mask_x_idx, mask_y_idx = np.where((~mask_inner == True) | (~mask_outer == True))
            OF_x_idx, OF_y_idx = np.where((~mask_inner == False) & (~mask_outer == False))

            self.mask_x_idx, self.mask_y_idx = mask_x_idx, mask_y_idx
            self.OF_x_idx, self.OF_y_idx = OF_x_idx, OF_y_idx

            return img1, img2

    def flow_tracking(self, img1, img2, LR=None):
        flo = opt.calc(img1, img2, None)
        mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
        inf_x, inf_y = np.where(np.isinf(mag))
        nan_x, nan_y = np.where(np.isnan(ang))
        mag[inf_x, inf_y] = 0
        mag[nan_x, nan_y] = 0
        mag, ang = mag, ang * 180 / np.pi
        ang = np.round(ang)

        # 순간변화량 추가
        # momentum = self.momentum(mag)
        mag = np.reshape(mag, (param.resize_X, param.resize_Y))
        ang = np.reshape(ang, (param.resize_X, param.resize_Y))
        # momentum = np.reshape(momentum, (1, param.resize_X, param.resize_Y))
        # OF = np.append(mag, ang, axis=0)
        # OF = np.append(OF, momentum, axis=0)
        self.save_mag(mag, ang, LR)

    # def momentum(self, mag):
    #     self.img_buffer.append(mag)
    #     copy_buffer = self.img_buffer.copy()
    #     first = copy_buffer.pop()
    #     second = copy_buffer.pop()
    #     momentum = abs(second - first)
    #     return momentum

    def save_mag(self, mag, ang, LR):

        OF_mag = np.round(mag, 2)
        OF_ang = np.round(ang)
        # OF_mot = np.round(OF[2], 2)

        OF_mag = np.array(OF_mag, dtype=np.float16)
        OF_ang = np.array(OF_ang, dtype=np.float16)
        # OF_mot = np.array(OF_mot, dtype=np.float16)

        OF_mag_dic = dict(np.ndenumerate(OF_mag))
        OF_ang_dic = dict(np.ndenumerate(OF_ang))
        # OF_mot_dic = dict(np.ndenumerate(OF_mot))

        for i in range(len(self.mask_x_idx)):
            del (OF_mag_dic[(self.mask_x_idx[i], self.mask_y_idx[i])])
            del (OF_ang_dic[(self.mask_x_idx[i], self.mask_y_idx[i])])
            # del (OF_mot_dic[(self.mask_x_idx[i], self.mask_y_idx[i])])

        OF_mag = np.array(list(OF_mag_dic.values()))
        OF_ang = np.array(list(OF_ang_dic.values()))
        # OF_mot = np.array(list(OF_mot_dic.values()))

        save = saver.Save()
        save.frame_append(OF_mag, OF_ang, LR)

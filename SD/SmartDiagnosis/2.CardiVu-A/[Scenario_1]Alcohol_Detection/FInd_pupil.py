import cv2
import numpy as np

import param


class Find_pupil_center:
    def __init__(self, img=None, threshold=None):
        self.threshold = threshold

        self.Find_pupil_Center(img)
        param.g_mx = int(param.pupilR_X + param.pupilR_W / 2)
        param.g_my = int(param.pupilR_Y + param.pupilR_W / 2)
        param.g_mx2 = int(param.pupilL_X + param.pupilL_W / 2)
        param.g_my2 = int(param.pupilL_Y + param.pupilL_W / 2)
        # print(param.g_mx, param.g_my, param.g_mx2, param.g_my2)
        if param.vis_frame:
            self.update_image(img)

    def Find_pupil_Center(self, img):
        h, w, c = img.shape

        size = 150

        grayMat = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
        binaryMat = cv2.threshold(grayMat, self.threshold, 255, cv2.THRESH_BINARY)[1]

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 가로 3, 세로 3
        dilate1 = cv2.dilate(binaryMat, se)  # 3x3 팽창 의미는 한픽셀정도만 팽창
        erode1 = cv2.erode(dilate1, se)

        contours, hierarchy = cv2.findContours(erode1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # OpenCV=v4.5.3
        # _, contours, hierarchy = cv2.findContours(erode1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # OpenCV=v3.4.2
        dc_img = cv2.drawContours(erode1, contours, -1, (255, 255, 255))
        if param.vis_frame:
            cv2.imshow('dc_img', dc_img)
            cv2.waitKey(1)

        for i in contours:
            area = cv2.contourArea(i)
            rect = cv2.boundingRect(i)
            x, y, width, height = rect

            radius = width / 2

            area_condition = (size <= area <= (size * 20))
            symmetry_condition = (abs(1 - float(height) / float(width)) <= 0.2)
            fill_condition = (abs(1 - (area / (np.pi * (radius * radius)))) <= 0.6)

            if area_condition and symmetry_condition and fill_condition:
                param.pupil_on = True

                if x < int(w / 2):
                    param.pupilR_X, param.pupilR_Y, param.pupilR_W, param.pupilR_H = x, y, width, height
                    # print("Pupil O / Rx = " + str(param.pupilR_X + param.pupilR_W / 2) + ", Ry = " + str(
                    #     param.pupilR_Y + param.pupilR_W / 2) + ", Rr = " + str(param.pupilR_W / 2))

                else:
                    param.pupilL_X, param.pupilL_Y, param.pupilL_W, param.pupilL_H = x, y, width, height
                    # print("Pupil O / Lx = " + str(param.pupilL_X + param.pupilL_W / 2) + ", Ly = " + str(
                    #     param.pupilL_Y + param.pupilL_W / 2) + ", Lr = " + str(param.pupilL_W / 2))

    def update_image(self, img):
        paint_img = img.copy()

        paint_img = cv2.circle(paint_img, (param.g_mx, param.g_my), 2, (0, 255, 0), 2)
        paint_img = cv2.rectangle(paint_img, (param.g_mx - int(param.g_m_w / 2), param.g_my - int(param.g_m_w / 2)),
                                  (param.g_mx + int(param.g_m_w / 2), param.g_my + int(param.g_m_w / 2)), (0, 255, 0), 2)

        paint_img = cv2.circle(paint_img, (param.g_mx2, param.g_my2), 2, (0, 255, 0), 2)
        paint_img = cv2.rectangle(paint_img, (param.g_mx2 - int(param.g_m_w / 2), param.g_my2 - int(param.g_m_w / 2)),
                                  (param.g_mx2 + int(param.g_m_w / 2), param.g_my2 + int(param.g_m_w / 2)), (0, 255, 0),
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

        cv2.imshow("img", img)
        cv2.imshow("paint_img", paint_img)
        cv2.waitKey(1)


class pupilSize:
    def __init__(self):
        pass

    def pupilSize(self, img, threshold):
        param.radian = None

        size = param.g_m_w
        grayMat = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
        binaryMat = cv2.threshold(grayMat, threshold, 255, cv2.THRESH_BINARY)[1]

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 가로 3, 세로 3
        dilate1 = cv2.dilate(binaryMat, se)  # 3x3 팽창 의미는 한픽셀정도만 팽창
        erode1 = cv2.erode(dilate1, se)

        contours, hierarchy = cv2.findContours(erode1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # OpenCV=v4.5.3
        # _, contours, hierarchy = cv2.findContours(erode1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # OpenCV=v3.4.2
        dc_img = cv2.drawContours(erode1, contours, -1, (255, 255, 255))
        if param.vis_frame:
            cv2.imshow('mask_dc_img', dc_img)
            cv2.waitKey(1)

        for i in contours:
            area = cv2.contourArea(i)
            rect = cv2.boundingRect(i)
            x, y, width, height = rect
            radius = width / 2

            area_condition = (size <= area <= (size * 20))
            symmetry_condition = (abs(1 - float(height) / float(width)) <= 0.2)
            fill_condition = (abs(1 - (area / (np.pi * (radius * radius)))) <= 0.6)

            if area_condition and symmetry_condition and fill_condition:
                pupil_X, pupil_Y, pupil_W, pupil_H = x, y, width, height
                param.radian = pupil_W
                # print("Pupil O / x = " + str(pupil_X + pupil_W / 2) + ", y = " + str(
                #     pupil_Y + pupil_W / 2) + ", r = " + str(pupil_W))

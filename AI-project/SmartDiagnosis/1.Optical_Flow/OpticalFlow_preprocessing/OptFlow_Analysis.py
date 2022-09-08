from collections import Counter
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import cv2
import flow_vis as fv

## 53 & 148
## 87 & 233

class optical_flow:
    def __init__(self,
                 img1_src='',
                 img2_src='',
                 compare=None,
                 edge=None,
                 blur=None,
                 flow=None,
                 rgb_vis=None,
                 save=None,
                 flow_clean=None,
                 mask=None,
                 mask_loc=None):

        self.compare = compare
        self.edge = edge
        self.blur = blur
        self.flow = flow
        self.rgb_vis = rgb_vis
        self.save = save
        self.flow_clean = flow_clean
        self.mask = mask
        self.img1_src = img1_src
        self.img2_src = img2_src
        self.mask_loc = mask_loc

        self.opt = None
        self.prev = None
        self.nexts = None
        self.flo = None
        self.i = 0
        self.mag_mean = np.array([])

        if compare == "image":
                self.prev, self.nexts = self.image_load()
                self.prev, self.nexts = self.circle_mask(self.prev, self.nexts)
                self.prev, self.nexts = self.edge_detecting(self.prev, self.nexts)
                self.prev, self.nexts = self.bluring(self.prev, self.nexts)
                self.flow_tracking(self.prev, self.nexts)

        elif compare == "video":
            cap = cv2.VideoCapture(vid_path + vid_name + ".avi")
            cap2 = cv2.VideoCapture(vid_path + vid_name + ".avi")
            r, self.prev = cap.read()
            self.prev = cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)
            while True:
                ret, self.nexts = cap2.read()
                if not ret:
                    self.mag_mean = pd.DataFrame(self.mag_mean, columns=None)
                    # self.mag_mean.to_csv(mag_save + vid_name + ".csv")
                    print('Magnitude Done...')
                    exit(0)
                nexts = cv2.cvtColor(self.nexts, cv2.COLOR_BGR2GRAY)
                prev, nexts = self.circle_mask(self.prev, nexts)
                prev, nexts = self.bluring(prev, nexts)
                prev, nexts = self.edge_detecting(prev, nexts)
                self.flow_tracking(prev, nexts)
                cv2.waitKey(1)
            cap2.release()

    def image_load(self):
        prev = cv2.imread(self.img1_src)
        nexts = cv2.imread(self.img2_src)

        prev = cv2.resize(prev, (200, 200))
        nexts = cv2.resize(nexts, (200, 200))

        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        nexts = cv2.cvtColor(nexts, cv2.COLOR_BGR2GRAY)
        return prev, nexts

    def create_circular_mask(self, h, w, center=None, radius=None, location=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        if location == "inner":
            mask = dist_from_center >= radius       ## Boolean 값
            return mask

        elif location == "outer":
            mask = dist_from_center <= radius       ## Boolean 값
            return mask

    def circle_mask(self, img1, img2):
        if self.mask:
            inner_radius, outer_radius = None, None
            ## mask1 : 29, 56
            if self.mask_loc == 'mask1':
                inner_radius = int(img1.shape[0] / 2 * 0.29)
                outer_radius = int(img1.shape[0] / 2 * 0.56)

            ## mask2 : 56, 90
            elif self.mask_loc == 'mask2':
                inner_radius = int(img1.shape[0] / 2 * 0.56)
                outer_radius = int(img1.shape[0] / 2 * 0.90)

            mask_inner = self.create_circular_mask(img1.shape[0], img1.shape[1], [img1.shape[0]/2, img1.shape[1]/2],
                                                   inner_radius,
                                                   location='inner')
            mask_outer = self.create_circular_mask(img1.shape[0], img1.shape[1], [img1.shape[0]/2, img1.shape[1]/2],
                                                   outer_radius,
                                                   location='outer')
            img1[~mask_inner] = 0
            img1[~mask_outer] = 0
            img2[~mask_inner] = 0
            img2[~mask_outer] = 0

            cv2.imshow('masked_img1', img1)
            cv2.imshow('masked_img2', img2)
            return img1, img2

    def edge_detecting(self, img1, img2):
        if self.edge == "Canny":
            img1 = cv2.Canny(img1, 0, 30)
            img2 = cv2.Canny(img2, 0, 30)
            # cv2.imshow(self.edge, img1)
            # cv2.imshow(self.edge + "1", img2)
            return img1, img2

        elif self.edge == "Laplacian":
            img1 = cv2.Laplacian(img1, -1, scale=5)
            img2 = cv2.Laplacian(img2, -1, scale=5)
            # cv2.imshow(self.edge, img1)
            # cv2.imshow(self.edge + "1", img2)
            # cv2.waitKey(0)
            return img1, img2

        elif self.edge == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img1 = clahe.apply(img1)
            img2 = clahe.apply(img2)
            cv2.imshow(self.edge, img1)
            cv2.imshow(self.edge + "1", img2)
            return img1, img2
        return img1, img2

    def bluring(self, img1, img2):
        threshold = 3
        if self.blur == "median":
            img1 = cv2.medianBlur(img1, threshold)
            img2 = cv2.medianBlur(img2, threshold)
            # cv2.imshow('img1_blur', img1)
            # cv2.imshow('img2_blur', img2)
        else:
            print(f'blur type is {self.blur}')
        return img1, img2

    def flow_tracking(self, img1, img2):
        """
        createOptFlow_SparseToDense()
        createOptFlow_Farneback()
        createOptFlow_PCAFlow()
        createOptFlow_DeepFlow()
        createOptFlow_DIS()
        createVariationalFlowRefinement()
        createOptFlow_SimpleFlow()
        """

        if self.flow == "Farneback":
            self.opt = cv2.optflow.createOptFlow_Farneback()
            self.flo = self.opt.calc(img1, img2, None)
            # self.flo = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif self.flow == "SparseToDense":
            self.opt = cv2.optflow.createOptFlow_SparseToDense()
            self.flo = self.opt.calc(img1, img2, None)
            # self.flo = cv2.optflow.calcOpticalFlowSparseToDense(img1, img2)
        elif self.flow == "PCAFlow":
            self.opt = cv2.optflow.createOptFlow_PCAFlow()
            self.flo = self.opt.calc(img1, img2, None)
        elif self.flow == "DeepFlow":
            self.opt = cv2.optflow.createOptFlow_DeepFlow()
            self.flo = self.opt.calc(img1, img2, None)
        elif self.flow == "DIS":
            self.opt = cv2.optflow.createOptFlow_DIS()
            self.flo = self.opt.calc(img1, img2, None)
        else:
            print(f'flow type is {self.flow}')
            return -1

        if self.rgb_vis:
            self.flow_vis(self.flo)

        if self.flow_clean:
            mag, ang = cv2.cartToPolar(self.flo[..., 0], self.flo[..., 1])
            inf_x, inf_y = np.where(np.isinf(mag))
            nan_x, nan_y = np.where(np.isnan(ang))
            mag[inf_x, inf_y] = 0
            mag[nan_x, nan_y] = 0
            mag, ang = mag, ang * 180 / np.pi

            self.flo[..., 0], self.flo[..., 1] = cv2.polarToCart(mag, ang)

            ## clean flow visualizing
            self.flow_vis(self.flo)

        if self.save:
            mag, ang = cv2.cartToPolar(self.flo[..., 0], self.flo[..., 1])
            self.save_flow(mag, ang)

    def flow_vis(self, flow, flow_save=None):
        flow_c = fv.flow_to_color(flow, convert_to_bgr=False)
        flow_c = cv2.resize(flow_c, (200, 200))
        cv2.imshow(self.flow, flow_c)

    def save_flow(self, mag, ang):
        mag = np.reshape(mag, (40000,))  # 전체 히스토그램 y값의 합 = 32,400(pixel)
        ang = np.reshape(ang, (40000,))
        mag_mean = np.mean(mag)
        self.mag_mean = np.append(self.mag_mean, mag_mean)
        mag = pd.DataFrame(mag, columns=None)
        mag.T.to_csv(mag_save + "frame_"+str(self.i).rjust(4, '0')+".csv", index=False, header=False)
        self.i += 1


if __name__ == '__main__':
    """
    compare : ['image', 'video']
    edge : ['Canny', 'Laplacian', 'CLAHE', 'None']
    blur : ['median']
    flow : ['Farneback', 'SparseToDense', 'PCAFlow', 'DeepFlow']
    rgb_vis : boolean ['True', 'False']
    plot : boolean ['True', 'False']
    flow_clean : boolean ['True', 'False']
    mask : boolean ['True', 'False']
    mask_loc : ['mask1', 'mask2']
    """

    compare = 'video'
    flow = 'DeepFlow'
    edge = 'None'
    blur = None
    rgb_vis = True
    save = True
    flow_clean = True
    mask = True
    mask_loc = 'mask2'

    fps = 100

    path = r"E:\ryu_pythonProject\9. FlowNet\light reflection test IRIS\SR\image\00/"

    vid_path = r"E:\ryu_pythonProject\9. FlowNet\light reflection test IRIS\SR/"
    vid_name = "(SR)20210803-171130_L"

    flo_vis_path = r"E:\ryu_pythonProject\9. FlowNet\image_data\flow_vis/"
    mag_save = r"E:\ryu_pythonProject\9. FlowNet\light reflection test IRIS\SR\magnitude" + "/" + flow + "-" + edge + "-" + mask_loc + "/00/"
    save_vid = r"E:\ryu_pythonProject\9. FlowNet\light reflection test IRIS\SR\OF_vid" + "/" + flow + "-" + edge + "-" + mask_loc + "_"
    print(flow + "-" + edge + "-" + mask_loc+"/00")
    # 87, 233
    optical_flow(img1_src=path + "(sr_img)0_00000.jpg",
                 img2_src=path + "test.jpg",
                 compare=compare,
                 edge=edge,
                 blur=blur,
                 flow=flow,
                 rgb_vis=rgb_vis,
                 save=save,
                 flow_clean=flow_clean,
                 mask=mask,
                 mask_loc=mask_loc
                 )



import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

import param


class img_processing:
    def __init__(self):
        pass

    def calcAdaptiveThreshold(self, img):
        # Preprocessing(grayscale, resize, gaussianBlur)
        try:
            img_gray = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
        except:
            img_gray = ~img
        if img.shape[0] != img.shape[1]:
            img_gray = cv2.resize(img_gray, [180, 180])
        img_gaus = cv2.GaussianBlur(img_gray, [3, 3], 0)

        # Init Try
        threshold = 50
        _, thr = cv2.threshold(img_gaus, threshold, 255, cv2.THRESH_BINARY_INV)
        circles = cv2.HoughCircles(thr, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=10, minRadius=10, maxRadius=0)
        if type(circles) == type(None) or img_gaus.mean() > 190:
            print("No Eye Detected!")
            return None

        while circles.shape[1] > 2:
            print(threshold)
            threshold += 1
            _, thr = cv2.threshold(img_gaus, threshold, 255, cv2.THRESH_BINARY_INV)
            circles = cv2.HoughCircles(thr, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=10, minRadius=10, maxRadius=0)

        return threshold

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

    def draw_circle_mask(self, img1, mask=True, mask_loc=None, LR=None):
        if mask:
            inner_radius, outer_radius = None, None
            ## mask1 : 29, 56
            if mask_loc == 'mask1':
                inner_radius = int(img1.shape[0] / 2 * 0.29)
                outer_radius = int(img1.shape[0] / 2 * 0.48)

            ## mask2 : 56, 90
            elif mask_loc == 'mask2':
                inner_radius = int(img1.shape[0] / 2 * 0.56)
                outer_radius = int(img1.shape[0] / 2 * 0.90)

            ## mask2 : 38, 56
            elif mask_loc == 'mask1-2':
                inner_radius = int(img1.shape[0] / 2 * 0.38)
                outer_radius = int(img1.shape[0] / 2 * 0.38)

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

            # img1[~mask_inner] = 0
            img1[~mask_outer] = 0

            x_idx, y_idx = np.where(~mask_outer == False)

            return img1, x_idx, y_idx

    # def cond_illuminance(self):
    #
    #
    # def cond_eyeSize(self):


class data_processing:
    def __init__(self):
        pass

    def prep(self, data):
        data = np.array(data)
        np.argwhere(data)
        # 프레임 사이의 nan값 처리
        max_nan_count = 60

        for i in range(len(data)):
            count = 0
            if np.isnan(data[i]):
                while np.isnan(data[i + count]):
                    count += 1
                    if i + count >= len(data):
                        break
                if count < max_nan_count:
                    data[i:i + count] = data[i - 1]
        plt.plot(data)
        plt.show()
        return data

    def peak(self, data):
        peak = []
        for i in range(1, len(data)):
            if np.isnan(data[i - 1]) and ~np.isnan(data[i]):
                p = data[i:i + 20].argmax()
                peak.append(i + p)
        return peak

    def valley(self, data, peaks):
        # Nan
        data = np.nan_to_num(data)

        # valley
        valley = []
        for p in peaks:
            v = data[p:p + 200].argmin()
            valley.append(p + v)
        return valley

    def primer(self, data, peaks):
        # Nan
        data = np.nan_to_num(data)

        # primer
        primer = []
        for i in range(len(data[:peaks[0]])):
            if data[i:i + 100].sum() == 0:
                primer.append(i - 1)
                break

        for i in range(len(peaks) - 1):
            data_p2p = data[peaks[i]:peaks[i + 1]]
            for k in range(len(data_p2p)):
                if data_p2p[k:k + 100].sum() == 0:
                    primer.append(peaks[i] + k - 1)
                    break
        print(len(peaks))
        return primer

    def retriever(self, data, peaks, primer, valley):
        # Nan
        data = np.nan_to_num(data)

        # retrievers
        retrievers = []
        rt_values = data[primer] * 0.85
        for i in range(len(peaks) - 1):
            data_p2p = data[valley[i + 1]:peaks[i + 1]]
            for k in range(len(data_p2p)):
                if data_p2p[k] == int(rt_values[i]) and peaks[i] + k > valley[i]:
                    retrievers.append(peaks[i] + k)
                    break

        data_last = data[peaks[-1]:]
        for i in range(len(data_last)):
            if data_last[i] == int(rt_values[-1]) and peaks[-1] + i > valley[-1]:
                retrievers.append(peaks[-1] + i)
                break
        return retrievers

    def Feature_Extraction(self, data):
        """

        :param data:  입력 데이터
       :return: retrievers(회복), shrink(수축)
        """
        data = self.prep(data)
        peaks = self.peak(data)
        valley = self.valley(data, peaks)
        primer = self.primer(data, peaks)
        retrievers = self.retriever(data, peaks, primer, valley)

        print(valley)
        print(peaks)
        shrink = [valley[i] - peaks[i] for i in range(len(peaks))]
        print(shrink)
        return retrievers, shrink

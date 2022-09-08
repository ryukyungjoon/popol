import os
import math
import pandas as pd
import numpy as np
import time

import scipy.signal as signal
from datetime import datetime
from collections import Counter

from sklearn.cluster import KMeans

GPS_DATA_PATH = r"D:\dataset\dataset\sensing\gps/"
PHQ9_DATA_PATH = r"D:\dataset\dataset\survey/"
RES_SAVE_PATH = r"D:\dataset\rkj_RESULT/"


class loadData:
    def __init__(self):
        pass

    def gps(self, file):
        data = pd.read_csv(GPS_DATA_PATH + file, index_col=False)
        return data

    def lat_long(self, file):
        gps_data = self.gps(file)
        latitude, longitude = gps_data[['time', 'latitude']], gps_data[['time', 'longitude']]
        # latitude['time'] = pd.to_datetime(latitude['time'], unit="s")
        # longitude['time'] = pd.to_datetime(longitude['time'], unit="s")
        return latitude, longitude

    def phq_9(self):
        phq9 = pd.read_csv(PHQ9_DATA_PATH + "PHQ-9.csv", index_col=False)
        return phq9

    def getStationary_Data(self, file):
        df = self.gps(file)
        s_data = df[df.travelstate == 'stationary']
        return s_data

    def location(self, file):
        cur = self.gps(file)
        locMat = pd.concat([cur.latitude, cur.longitude], axis=1)
        timeMat = []
        time_list = list(cur.time)
        for row in time_list:
            t = datetime.fromtimestamp(row / 1000)
            timeMat.append(time.mktime(t.timetuple()))
        return np.array(locMat), np.array(timeMat)


class lifelogging(loadData):
    def __init__(self):
        super().__init__()

    def phq9_score(self):
        phq9 = self.phq_9()
        answer = ['Not at all', 'Several days', 'More than half the days', 'Nearly every day']
        for score, ans in enumerate(answer):
            phq9 = phq9.replace(ans, score)
        pre, post = phq9[phq9.type == 'pre'], phq9[phq9.type == 'post']
        header = pre.columns
        pre, post = pre[header[0:-1]], post[header[0:-1]]
        pre['PHQ-9 Score'], post['PHQ-9 Score'] = np.sum(pre[header[2:-1]], axis=1), np.sum(post[header[2:-1]], axis=1)
        pre.to_csv(RES_SAVE_PATH + "PHQ-9(pre).csv", index=False)
        post.to_csv(RES_SAVE_PATH + "PHQ-9(post).csv", index=False)

    def Location_Variance(self, file):
        data = self.gps(file)
        latitude, longitude = data['latitude'], data['longitude']
        location_variance = math.log(pow(np.var(latitude), 2) + pow(np.var(longitude), 2))
        return location_variance

    def Energy(self, y):
        y = y.iloc[:, 1]
        x = np.linspace(0, len(y), len(y)) + 1e-5
        f = np.linspace(-len(y) / 2, len(y) / 2, len(y)) + 1e-5
        pgram = signal.lombscargle(x, y, f)  # psd
        result = sum(pgram) / (len(y) + 1e-5)  # energy
        return result.real

    def Circadian_Movement(self, file):
        latitude, longitude = self.lat_long(file)
        E_lat = self.Energy(latitude)
        E_long = self.Energy(longitude)
        CM = np.log(E_lat + E_long)
        return CM

    def deg2rad(self, deg):
        return float(deg) * math.pi / 180.0

    def rad2deg(self, rad):
        return float(rad) * 180.0 / math.pi

    def clean_acos(self, acos_angle):
        return min(1, max(acos_angle, -1))

    def distance(self, vecA, vecB):
        theta = abs(vecA[1]) - abs(vecB[1])
        dist = math.sin(self.deg2rad(vecA[0])) * math.sin(self.deg2rad(vecB[0])) + math.cos(
            self.deg2rad(vecA[0])) * math.cos(self.deg2rad(vecB[0])) * math.cos(self.deg2rad(theta))
        dist = math.acos(self.clean_acos(dist))
        dist = self.rad2deg(dist)

        dist = dist * 60 * 1.1515
        dist = dist * 1.609  # mile to km
        # dist = dist * 1000.0  # km to m
        return dist

    def slldist(self, data, centroids):
        max_dist = 0
        for i in range(0, len(data)):
            for j in range(0, len(centroids)):
                dist = self.distance(data[i], centroids[j])
                if dist > max_dist: max_dist = dist
        return dist

    def classify0(self, locMat):
        k = 1
        k_means = None

        while True:
            k_means = KMeans(n_clusters=k, random_state=0)
            k_means.fit(locMat)
            distance = k_means.transform(locMat)
            print("distance", distance)
            print("distance_length", len(distance))

            dist = self.slldist(locMat, k_means.cluster_centers_)

            if dist <= 0.5:
                break
            elif k >= 50:
                break
            else:
                k += 1

        return k, k_means.cluster_centers_, k_means.labels_

    def Total_distance(self, file):
        latitude, longitude = self.lat_long(file)
        Total_distance = 0
        for i in range(len(latitude) - 1):
            Total_distance += math.sqrt(pow(latitude.iloc[i + 1, 1] - latitude.iloc[i, 1], 2) + pow(
                longitude.iloc[i + 1, 1] - longitude.iloc[i, 1], 2))
        return Total_distance

    def Entropy(self):
        N_cluster = self.n_clusters()
        Entropy = n
        return Entropy

    def norm_entropy(self):
        norm_Entropy = self.Entropy() / np.log(self.n_clusters())

    # def Home_stay(self):

    def transition_time(self, file):
        ## moving label
        total = self.gps(file)
        # print(len(total))
        transit = total[total.provider == 'gps']
        # print(len(transit))
        transit_time_percentile = len(transit)/len(total) * 100
        return transit_time_percentile

    def mining(self, locMat, timeMat):
        miningMat = []
        for i in range(locMat.shape[0] - 1):
            # calculate distance
            dist = self.distance(locMat[i][:], locMat[i + 1][:])
            # calculate state
            timeDiff = (timeMat[i + 1] - timeMat[i])
            movement = 0
            if timeDiff != 0:
                movement = dist / (timeDiff / 1000 / 60)
            state = 0
            if movement > 1.0:  # faster than 1 km/h
                state = 1
            miningMat.append([state, dist, timeDiff])
        return np.array(miningMat)

    # def trainsition_time(self, miningMat):
    #     total_time = 0
    #     for i in range(miningMat.shape[0]):
    #         if miningMat[i, 0] == 1:  # if transition state
    #             total_time += miningMat[i, 2]
    #     return total_time

    def survey_main(self):
        self.phq9_score()

    def activity_main(self, file):
        loc_var, cm, td, k, t_time = 0, 0, 0, 0, 0
        locMat, timeMat = self.location(file)
        miningMat = self.mining(locMat, timeMat)

        # t_time = self.trainsition_time(miningMat)
        # loc_var = self.Location_Variance(file)
        # cm = self.Circadian_Movement(file)
        # cm = self.circadian_movement(locMat)
        # td = self.Total_distance(file)
        k, cluster_centers, labels = self.classify0(locMat)
        # t_time = self.transition_time(file)
        # print(k)
        # exit(0)
        return loc_var, cm, td, k, t_time


if __name__ == '__main__':
    uid = []
    Location_Variance = []
    Circadian_Movement = []
    Total_Distance = []
    Number_of_Cluster = []
    Transition_Time = []

    calc = lifelogging()

    file_list = os.listdir(GPS_DATA_PATH)
    calc.survey_main()
    for file in file_list:
        uid.append(file[4:7])
        lv, cm, td, n_cluster, t_time = calc.activity_main(file)
        print(t_time)
        # print("td", td)
        Location_Variance.append(lv)
        Circadian_Movement.append(cm)
        Total_Distance.append(td)
        Number_of_Cluster.append(n_cluster)
        Transition_Time.append(t_time)

    uid = pd.DataFrame(uid)

    # lv = pd.DataFrame(Location_Variance)
    # lv = pd.concat([uid, lv], axis=1)
    # lv.to_csv(RES_SAVE_PATH + "Location Variance.csv")

    # cm = pd.DataFrame(Circadian_Movement)
    # cm = pd.concat([uid, cm], axis=1)
    # cm.to_csv(RES_SAVE_PATH + "Circadian_Movement.csv")

    # td = pd.DataFrame(Total_Distance)
    # td = pd.concat([uid, td], axis=1)
    # td.to_csv(RES_SAVE_PATH + "Total_Distance.csv")

    # nc = pd.DataFrame(Number_of_Cluster)
    # nc = pd.concat([uid, nc], axis=1)
    # nc.to_csv(RES_SAVE_PATH + "Number_of_Cluster.csv")

    tt = pd.DataFrame(Transition_Time)
    tt = pd.concat([uid, tt], axis=1)
    tt.to_csv(RES_SAVE_PATH + "Transition_Time.csv")

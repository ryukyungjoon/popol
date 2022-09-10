import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import pickle
from typing import List, Union

import numpy as np
import os
import random

from collections import Counter
from matplotlib import pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import joblib
import pickle as pkl

from keras.layers import Dense, Input, LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import tensorflow as tf
tf.random.set_random_seed(2)
initializer = tf.keras.initializers.RandomUniform(seed=1)

""" 30초 간격으로 홍채 변화 값을 신호처리한 OF(x)5856 심장박동 값 bpm(y)을 데이터로 잡는다."""

# Oversampling strategy
sp_strategy_1 = {
    "H": 27000,
    "L": 27000
}

# Undersampling strategy
sp_strategy_2 = {
    "M": 7000,
}

high_strategy = {
    "H1": 10000,
    "H2": 10000
}

low_strategy = {
    "L1": 10000,
    "L2": 10000
}

mid_strategy = {
    "M1": 10000,
    "M2": 10000,
    "M4": 10000
}

class preprocessing:
    def __init__(self, sampling=None, sampling_method=None, save=None, model_save=None):
        self.sampling_method = sampling_method
        self.sampling = sampling
        self.save = save
        self.sp_model = None
        self.model = None
        self.epoch = 200
        self.optimizer = Adam(learning_rate=0.00002)
        self.model_save = model_save

        self.tr_path = "./data_split/OF5856/mid_rf3/train/train/fs_"
        self.val_path = "./data_split/OF5856/mid_rf3/train/valid/noise_remove/"
        self.te_path = "./data_split/OF5856/mid_rf3/test/noise_remove/"

        self.model_path = "./model_save/OF5856/mid_rf3/"
        self.res_path = "./result/data_split_result/OF5856/mid_rf3/noise_remove/"

        # Oversampling
        if self.sampling_method == 'over':
            self.sp_model = SMOTE(sampling_strategy=sp_strategy_1, random_state=42)
        elif self.sampling_method == 'under':
            self.sp_model = RandomUnderSampler(sampling_strategy=sp_strategy_2, random_state=42)
        else:
            pass

    def data_normalization(self, train_x, test_x, valid_x, train_y, x_col, file_name):
        std = StandardScaler()
        std.fit(train_x, train_y)
        train_x = std.transform(train_x)
        test_x = std.transform(test_x)
        valid_x = std.transform(valid_x)

        ''' Save Scaler Model'''
        if self.model_save:
            scaler_file = self.model_path + file_name + '_scaler.pkl'
            joblib.dump(std, scaler_file)

        train_x, test_x, valid_x = pd.DataFrame(train_x, columns=x_col), pd.DataFrame(test_x,
                                                                                      columns=x_col), pd.DataFrame(
            valid_x, columns=x_col)
        print(valid_x)

        return train_x, valid_x, test_x

    def feature_selection(self, train, test, vaild, file_name, save=None):
        data_full = train.iloc[:, 72:96]
        data_GAvg = train.iloc[:, 3936:3960]
        data_34GAvg = train.iloc[:, 4896:4920]
        label = train.iloc[:, 5856]

        TRAIN = pd.concat([data_full, data_GAvg, data_34GAvg, label], axis=1, join='outer')

        data_full = test.iloc[:, 72:96]
        data_GAvg = test.iloc[:, 3936:3960]
        data_34GAvg = test.iloc[:, 4896:4920]
        label = test.iloc[:, 5856]

        TEST = pd.concat([data_full, data_GAvg, data_34GAvg, label], axis=1, join='outer')

        data_full = vaild.iloc[:, 72:96]
        data_GAvg = vaild.iloc[:, 3936:3960]
        data_34GAvg = vaild.iloc[:, 4896:4920]
        label = vaild.iloc[:, 5856]

        VAL = pd.concat([data_full, data_GAvg, data_34GAvg, label], axis=1, join='outer')

        if save:
            TRAIN.to_csv(self.tr_path+"fs_" + file_name + ".csv", index=False)
            TEST.to_csv(self.te_path+"fs_" + file_name + ".csv", index=False)
            VAL.to_csv(self.val_path+"fs_" + file_name + ".csv", index=False)

        return TRAIN, TEST, VAL

    def train_test_load(self, train_file_name, test_file_name):
        print('train:{} test:{}'.format(train_file_name, test_file_name))
        TRAIN = pd.read_csv(self.tr_path + train_file_name + ".csv", sep=',', index_col=False)
        TEST = pd.read_csv(self.te_path + test_file_name + ".csv", sep=',', index_col=False)
        VALID = pd.read_csv(self.val_path + test_file_name + ".csv", index_col=False)
        print(TRAIN)
        print(TEST)
        print(VALID)

        # TRAIN = TRAIN.drop('HML', axis=1)
        # TRAIN.drop('Unnamed: 0', axis=1, inplace=True)
        # TEST.drop('Unnamed: 0', axis=1, inplace=True)
        # VALID.drop('Unnamed: 0', axis=1, inplace=True)

        # TRAIN, TEST, VALID = self.feature_selection(TRAIN, TEST, VALID, test_file_name, save=True)

        return TRAIN, TEST, VALID

    def data_load(self):
        data_loc = self.tr_path

        file_name = ['high_data_1', 'high_data_2',
                     'low_data_1', 'low_data_2',
                     'mid_data_1', 'mid_data_2', 'mid_data_3', 'mid_data_4']
        TRAIN = []
        TEST = []

        for i, file in enumerate(file_name):
            print("OF_folder_1 file_{} {}".format(i, file))
            if i == 0:
                tr_df = pd.read_csv(data_loc + file + ".csv", sep=',')
                # te_df = pd.read_csv("./data_split/4.test(ratio 8;2)/" + file + ".csv", sep=',')
                TRAIN.append(tr_df)
                # TEST.append(te_df)
            else:
                tr_df = pd.read_csv(data_loc + file + ".csv", sep=',', index_col=False)
                # te_df = pd.read_csv("./data_split/4.test(ratio 8;2)/" + file + ".csv", sep=',', index_col=False)
                TRAIN.append(tr_df)
                # TEST.append(te_df)

        TRAIN_DATA = pd.concat(TRAIN, axis=0, ignore_index=True)
        # TEST_DATA = pd.concat(TEST, axis=0, ignore_index=True)
        TRAIN_DATA.drop(['Unnamed: 0'], axis=1, inplace=True)
        # TEST_DATA.drop(['Unnamed: 0'], axis=1, inplace=True)
        print(TRAIN_DATA)
        TRAIN_DATA.to_csv(data_loc + "/summed.csv", index=False)
        # TEST_DATA.to_csv("./data_split/4.test(ratio 8;2)/summed.csv", index=False)
        # exit(0)

        # for i, file in enumerate(file_name):
        #     TRAIN = pd.read_csv("./data_split/2.train/" + file + ".csv", sep=',')
        #     TEST = pd.read_csv("./data_split/2.test/" + file + ".csv", sep=',', index_col=False)
        #     print(TRAIN, TEST)
        #     TRAIN = pd.concat([TRAIN, TEST], axis=0)
        #     print(TRAIN)
        #     # drop_feature = ['index', 'group']
        #     # TRAIN = TRAIN.drop(drop_feature, axis=1)
        #
        #     y, x = TRAIN['bpm'], TRAIN.drop('bpm', 1)
        #     train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
        #
        #     TRAIN = pd.concat([train_x, train_y], axis=1)
        #     TEST = pd.concat([test_x, test_y], axis=1)
        #
        #     print(TRAIN, TEST)
        #
        #     TRAIN.to_csv("./(train)" + file + ".csv")
        #     TEST.to_csv("./(test)" + file + ".csv")
        # exit(0)
        return TRAIN_DATA

    """ Low & Mid & High를 분류하고 데이터 샘플 수를 파악해서 데이터 불균형 및 희소 클래스 문제 해결 """

    def LMH_classification(self, TRAIN_DATA):
        train_col = TRAIN_DATA.columns
        high_group_1, high_group_2 = 0, 0
        mid_group_1, mid_group_2, mid_group_3, mid_group_4 = 0, 0, 0, 0
        low_group_1, low_group_2 = 0, 0

        high_arr_2 = []
        high_arr_1 = []
        mid_arr_4 = []
        mid_arr_3 = []
        mid_arr_2 = []
        mid_arr_1 = []
        low_arr_2 = []
        low_arr_1 = []

        for i in range(0, len(TRAIN_DATA['bpm'])):
            if TRAIN_DATA.loc[i, 'bpm'] >= 110:
                high_group_2 += 1
                high_arr_2.append(TRAIN_DATA.loc[i])
            elif 110 > TRAIN_DATA.loc[i, 'bpm'] >= 100:
                high_group_1 += 1
                high_arr_1.append(TRAIN_DATA.loc[i])
            elif 100 > TRAIN_DATA.loc[i, 'bpm'] >= 90:
                mid_group_4 += 1
                mid_arr_4.append(TRAIN_DATA.loc[i])
            elif 90 > TRAIN_DATA.loc[i, 'bpm'] >= 80:
                mid_group_3 += 1
                mid_arr_3.append(TRAIN_DATA.loc[i])
            elif 80 > TRAIN_DATA.loc[i, 'bpm'] >= 70:
                mid_group_2 += 1
                mid_arr_2.append(TRAIN_DATA.loc[i])
            elif 70 > TRAIN_DATA.loc[i, 'bpm'] >= 60:
                mid_group_1 += 1
                mid_arr_1.append(TRAIN_DATA.loc[i])
            elif 60 > TRAIN_DATA.loc[i, 'bpm'] >= 50:
                low_group_2 += 1
                low_arr_2.append(TRAIN_DATA.loc[i])
            else:
                low_group_1 += 1
                low_arr_1.append(TRAIN_DATA.loc[i])

        print('high_group(110 up): {} high_group(100) {}'.format(high_group_2, high_group_1))
        print('mid_group(90):{} mid_group(80):{} mid_group(70):{} mid_group(60):{}'.format(mid_group_4, mid_group_3,
                                                                                           mid_group_2, mid_group_1))
        print('low_group(50):{} low_group(50 down) {}'.format(low_group_2, low_group_1))

        # high_data_2 = pd.DataFrame(high_arr_2, columns=train_col)
        # high_data_1 = pd.DataFrame(high_arr_1, columns=train_col)
        # mid_data_4 = pd.DataFrame(mid_arr_4, columns=train_col)
        # mid_data_3 = pd.DataFrame(mid_arr_3, columns=train_col)
        # mid_data_2 = pd.DataFrame(mid_arr_2, columns=train_col)
        # mid_data_1 = pd.DataFrame(mid_arr_1, columns=train_col)
        # low_data_2 = pd.DataFrame(low_arr_2, columns=train_col)
        # low_data_1 = pd.DataFrame(low_arr_1, columns=train_col)
        #
        # high_data_2.to_csv("./data_split/high_data_2.csv", index=False)
        # high_data_1.to_csv("./data_split/high_data_1.csv", index=False)
        # mid_data_4.to_csv("./data_split/mid_data_4.csv", index=False)
        # mid_data_3.to_csv("./data_split/mid_data_3.csv", index=False)
        # mid_data_2.to_csv("./data_split/mid_data_2.csv", index=False)
        # mid_data_1.to_csv("./data_split/mid_data_1.csv", index=False)
        # low_data_2.to_csv("./data_split/low_data_2.csv", index=False)
        # low_data_1.to_csv("./data_split/low_data_1.csv", index=False)

    def data_labeling(self, TRAIN_DATA, label_name):

        df_HML = pd.DataFrame(columns=['HML'])
        TRAIN_DATA = pd.concat([TRAIN_DATA, df_HML], axis=1)

        for feat in range(0, len(TRAIN_DATA['bpm'])):
            TRAIN_DATA.loc[feat, 'HML'] = label_name
        TRAIN_DATA.drop(['Unnamed: 0'], axis=1, inplace=True)
        print(TRAIN_DATA)
        return TRAIN_DATA

    ## Data Resampling
    def data_resampling(self, TRAIN_DATA):
        """

        :param TRAIN_DATA:
        :param model: resampling model
        :return: train_x(OF), train_y(BPM)
        """

        train_y, train_x = TRAIN_DATA['HML'], TRAIN_DATA.drop('HML', axis=1)
        print(train_x, train_y)
        train_col = train_x.columns

        print("Data Resampling...")
        sp_model = SMOTE(sampling_strategy=mid_strategy, random_state=0)
        # sp_model = RandomUnderSampler(sampling_strategy=sp_strategy_2, random_state=0)
        train_x, train_y = sp_model.fit_resample(train_x, train_y)

        # data type
        print('{} {}'.format(train_x, train_y))

        print(Counter(train_y))

        ## data split
        train_x = pd.DataFrame(train_x, columns=train_col)

        h1 = []
        h2 = []
        h3 = []
        h4 = []

        for j in range(0, len(train_x['bpm'])):
            if train_y.loc[j] == 'M1':
                h1.append(train_x.loc[j])
            elif train_y.loc[j] == 'M2':
                h2.append(train_x.loc[j])
            elif train_y.loc[j] == 'M3':
                h3.append(train_x.loc[j])
            else:
                h4.append(train_x.loc[j])

        high1 = pd.DataFrame(h1, columns=train_col)
        high2 = pd.DataFrame(h2, columns=train_col)
        high3 = pd.DataFrame(h3, columns=train_col)
        high4 = pd.DataFrame(h4, columns=train_col)

        high1.to_csv("./(over)mid_data_1.csv", index=False)
        high2.to_csv("./(over)mid_data_2.csv", index=False)
        high3.to_csv("./(over)mid_data_3.csv", index=False)
        high4.to_csv("./(over)mid_data_4.csv", index=False)

        return train_x


class Training(preprocessing):
    def build_dnn(self, input_data):
        dnn = Dense(6000)(input_data)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(5000)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(4000)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(3000)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(2000)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(1500)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(1000)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(750)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(500)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(300)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(150)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(64)(dnn)
        dnn = LeakyReLU(alpha=0.2)(dnn)
        dnn = Dense(1, activation='linear')(dnn)

        return dnn

    def train_model(self, train, test, valid, file_name):
        # self.LMH_classification(train)
        TRAIN = train
        TEST = test
        VALID = valid
        print("===================train,test,valid=======================")
        print(TRAIN)  # 5979
        print(TEST)
        print(VALID)

        # drop_feature = ['index', 'group']

        # TRAIN = train.drop(drop_feature, axis=1)
        # TEST = test.drop(drop_feature, axis=1)

        train_y, test_y, valid_y = TRAIN['bpm'], TEST['bpm'], VALID['bpm']
        train_x, test_x, valid_x = TRAIN.drop(['bpm'], axis=1), TEST.drop(['bpm'], axis=1), VALID.drop(['bpm'], axis=1)
        x_col = train_x.columns
        print(valid_x, valid_y)

        ## Data Normalization
        train_x, valid_x, test_x = self.data_normalization(train_x, test_x, valid_x, train_y, x_col, file_name)
        print("==================== 3 Normalization ==========================")
        print(train_x, train_y)
        print(valid_x, valid_y)
        print(test_x, test_y)

        train_x = np.array(train_x)
        valid_x = np.array(valid_x)
        test_x = np.array(test_x)
        sample, feature = train_x.shape
        print(train_x.shape)

        input_data = Input(shape=(feature,), name='input_data')
        dnn_output = self.build_dnn(input_data)
        self.model = Model(input_data, dnn_output)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['mae'])
        self.model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = self.model.fit(train_x, train_y, batch_size=64, epochs=self.epoch,
                                 validation_data=[valid_x, valid_y], verbose=1, callbacks=[es])
        # pred_bpm = self.model.predict(test_x)
        # print('1:', pred_bpm)

        if self.model_save:
            self.model.save(self.model_path + file_name + ".h5")
        model = load_model(self.model_path + file_name + ".h5")
        title = file_name + '_DNN Train Loss and Validation Loss'
        self.draw_loss_graph(history, title)

        pred_bpm = model.predict(test_x)
        print('2:', pred_bpm)

        # pred_bpm과 test_y를 mae 지표로 성능 정확도 측정
        mae = mean_absolute_error(test_y, pred_bpm)
        rmse = mean_squared_error(test_y, pred_bpm) ** 0.5
        r2 = r2_score(test_y, pred_bpm)
        print(f'mae:{mae}')
        print(f'rmse:{rmse}')
        print(f'r2:{r2}')

        pred_bpm = pd.DataFrame(pred_bpm, columns=['pred_bpm'])
        pred_true = pd.concat([pred_bpm, test_y], axis=1)

        pred_true.to_csv(self.res_path + file_name + "_bpm_result.csv")

    def draw_loss_graph(self, history, title):
        """

        :param history: model's history
        :param title: plot title
        :return: plot graph show()
        """

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.axis([0, self.epoch, 0, 500])
        plt.title(title)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    """
    resampling : 'over', 'under', 'None'
    save : True, False (Boolean)
    model : 'dnn'
    """

    # over_train_file_list = ['(over)high_data_1', '(over)high_data_2',
    #                         '(over)mid_data_1', '(over)mid_data_2',
    #                         '(over)mid_data_3', '(over)mid_data_4',
    #                         '(over)low_data_1', '(over)low_data_2',
    #                         '(over)summed']

    train_file_list = ['high_data_1', 'high_data_2', 'high_data_3',
                       'mid_data_1', 'mid_data_2', 'mid_data_3', 'mid_data_4', 'mid_data_5',
                       'low_data_1', 'low_data_2', 'low_data_3']

    i = 8

    tr = Training(sampling=False, sampling_method=None, save=True, model_save=True)
    train, test, valid = tr.train_test_load(train_file_list[i], train_file_list[i])
    tr.train_model(train, test, valid, train_file_list[i])

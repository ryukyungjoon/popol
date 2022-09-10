"""
Feature Importance & Label Correlation 으로,
Robust Feature 선택하기.

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm

from collections import Counter

data_path = r"E:\ryu_pythonProject\2. Cardivu-A\1. data_analysis\test_data/"
eyeb_path = r"E:\ryu_pythonProject\2. Cardivu-A\1. data_analysis\test_data\Eye_blinking/"

BL_path = data_path + "Before/left/"
BR_path = data_path + "Before/right/"
AL_path = data_path + "After/left/"
AR_path = data_path + "After/right/"

Before_L = ["[f]김환진_음주전_2021-05-17-181830-0000_M_L",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_L",
            "[f]이동원_음주전_2021-05-17-182348-0000_M_L",
            "[f]김환진_음주전_2021-06-09-183322-0000_M_L",
            "[f]문예성_음주전_2021-06-09-182014-0000_M_L",
            "[f]이동원_음주전_2021-06-09-182703-0000_M_L",
            "[f]신재민_음주전_2021-06-09-184016-0000_M_L"]

Before_R = ["[f]김환진_음주전_2021-05-17-181830-0000_M_R",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_R",
            "[f]이동원_음주전_2021-05-17-182348-0000_M_R",
            "[f]김환진_음주전_2021-06-09-183322-0000_M_R",
            "[f]문예성_음주전_2021-06-09-182014-0000_M_R",
            "[f]이동원_음주전_2021-06-09-182703-0000_M_R",
            "[f]신재민_음주전_2021-06-09-184016-0000_M_R"]

After_L = ["[f]김환진_음주후2_2021-05-17-203642-0000_M_L",
           "[f]문예성_음주후2_2021-05-17-203150-0000_M_L",
           "[f]이동원_음주후2_2021-05-17-202504-0000_M_L",
           "[f]김환진_음주후_2021-06-09-211918-0000_M_L",
           "[f]문예성_음주후_2021-06-09-213245-0000_M_L",
           "[f]이동원_음주후_2021-06-09-212501-0000_M_L",
           "[f]신재민_음주후_2021-06-09-213742-0000_M_L"]

After_R = ["[f]김환진_음주후2_2021-05-17-203642-0000_M_R",
           "[f]문예성_음주후2_2021-05-17-203150-0000_M_R",
           "[f]이동원_음주후2_2021-05-17-202504-0000_M_R",
           "[f]김환진_음주후_2021-06-09-211918-0000_M_R",
           "[f]문예성_음주후_2021-06-09-213245-0000_M_R",
           "[f]이동원_음주후_2021-06-09-212501-0000_M_R",
           "[f]신재민_음주후_2021-06-09-213742-0000_M_R"]


class exp_2:
    def __init__(self):
        print('Data Load')
        train, test = self.data_load(0, 6, eye_blinking=True)
        train, valid = train_test_split(train, test_size=0.2)
        train, test, valid, empty = self.Feature_Remove(df1=train, df2=test, df3=valid)

        train_y, train_x = train['label'], train.drop('label', axis=1)
        test_y, test_x = test['label'], test.drop('label', axis=1)
        valid_y, valid_x = valid['label'], valid.drop('label', axis=1)

        self.Feature_Selection(train_x, train_y, test_x, test_y)

    def data_load(self, start, end, eye_blinking=None):

        BL_1, AL_1, BR_1, AR_1 = None, None, None, None

        BL_2 = pd.DataFrame()
        AL_2 = pd.DataFrame()
        BR_2 = pd.DataFrame()
        AR_2 = pd.DataFrame()

        for i in range(start, end):
            if i == 0:
                BL_1 = pd.read_csv(BL_path + Before_L[i] + ".csv")
                BL_1 = BL_1.drop(index=0, axis=0)
                AL_1 = pd.read_csv(AL_path + After_L[i] + ".csv")
                AL_1 = AL_1.drop(index=0, axis=0)

                BR_1 = pd.read_csv(BR_path + Before_R[i] + ".csv")
                BR_1 = BR_1.drop(index=0, axis=0)
                AR_1 = pd.read_csv(AR_path + After_R[i] + ".csv")
                AR_1 = AR_1.drop(index=0, axis=0)
            else:
                BL_3 = pd.read_csv(BL_path + Before_L[i] + ".csv", index_col=None)
                BL_3 = BL_3.drop(index=0, axis=0)
                BL_2 = BL_2.append(BL_3)
                AL_3 = pd.read_csv(AL_path + After_L[i] + ".csv", index_col=None)
                AL_3 = AL_3.drop(index=0, axis=0)
                AL_2 = AL_2.append(AL_3)

                BR_3 = pd.read_csv(BR_path + Before_R[i] + ".csv", index_col=None)
                BR_3 = BR_3.drop(index=0, axis=0)
                BR_2 = BR_2.append(BR_3)
                AR_3 = pd.read_csv(AR_path + After_R[i] + ".csv", index_col=None)
                AR_3 = AR_3.drop(index=0, axis=0)
                AR_2 = AR_2.append(AR_3)

        test_BL = pd.read_csv(BL_path + Before_L[6] + ".csv")
        test_AL = pd.read_csv(AL_path + After_L[6] + ".csv")
        test_BL = test_BL.assign(label=0)
        test_AL = test_AL.assign(label=1)

        test_BR = pd.read_csv(BR_path + Before_R[6] + ".csv")
        test_AR = pd.read_csv(AR_path + After_R[6] + ".csv")
        test_BR = test_BR.assign(label=0)
        test_AR = test_AR.assign(label=1)

        if eye_blinking:
            BL_1, AL_1, BR_1, AR_1 = self.Feature_Remove(df1=BL_1, df2=AL_1, df3=BR_1, df4=AR_1)
            BL_2, AL_2, BR_2, AR_2 = self.Feature_Remove(df1=BL_2, df2=AL_2, df3=BR_2, df4=AR_2)
            test_BL, test_AL, test_BR, test_AR = self.Feature_Remove(df1=test_BL, df2=test_AL, df3=test_BR, df4=test_AR)

            BL_1 = self.outlier_remove(BL_1, file_name="Before_L1")
            AL_1 = self.outlier_remove(AL_1, file_name="After_L1")
            BR_1 = self.outlier_remove(BR_1, file_name="Before_R1")
            AR_1 = self.outlier_remove(AR_1, file_name="After_R1")

            BL_2 = self.outlier_remove(BL_2, file_name="Before_L2")
            AL_2 = self.outlier_remove(AL_2, file_name="After_L2")
            BR_2 = self.outlier_remove(BR_2, file_name="Before_R2")
            AR_2 = self.outlier_remove(AR_2, file_name="After_R2")

            test_BL = self.outlier_remove(test_BL, file_name="test_BL")
            test_AL = self.outlier_remove(test_AL, file_name="test_AL")
            test_BR = self.outlier_remove(test_BR, file_name="test_BR")
            test_AR = self.outlier_remove(test_AR, file_name="test_AR")
            print('eye blinking remove end..')

        # 왼쪽 눈
        BL = pd.concat([BL_1, BL_2], axis=0, ignore_index=True)
        AL = pd.concat([AL_1, AL_2], axis=0, ignore_index=True)
        BL = BL.assign(label=0)
        AL = AL.assign(label=1)

        train_L = pd.concat([BL, AL], axis=0, ignore_index=True)
        test_L = pd.concat([test_BL, test_AL], axis=0, ignore_index=True)

        # 오른쪽 눈
        BR = pd.concat([BR_1, BR_2], axis=0, ignore_index=True)
        AR = pd.concat([AR_1, AR_2], axis=0, ignore_index=True)
        BR = BR.assign(label=0)
        AR = AR.assign(label=1)

        train_R = pd.concat([BR, AR], axis=0, ignore_index=True)
        test_R = pd.concat([test_BR, test_AR], axis=0, ignore_index=True)

        # Normalization
        train_L, test_L = self.norm(train_L, test_L)
        train_R, test_R = self.norm(train_R, test_R)

        train = pd.concat([train_L, train_R], axis=0, ignore_index=True)
        test = pd.concat([test_L, test_R], axis=0, ignore_index=True)

        return train, test

    def Label_correlation(self, x, y):

        # Label Correlation(Pearson)
        print("Label Correltaion")
        x = pd.DataFrame(x)
        y = pd.Series(y)
        corr = []
        col = x.columns
        for i in range(len(col)):
            cor = y.corr(x.iloc[:, i], method='pearson')
            corr.append(abs(cor))
        corr = pd.DataFrame(corr)
        print(corr)

        return corr

    def norm(self, train, test):
        train_y, train_x = train['label'], train.drop('label', axis=1)
        x_col = train_x.columns
        test_y, test_x = test['label'], test.drop('label', axis=1)
        mms = MinMaxScaler(feature_range=(0, 1))
        mms.fit(train_x, train_y)
        train_x = mms.transform(train_x)
        test_x = mms.transform(test_x)

        train_x = pd.DataFrame(train_x, columns=x_col)
        test_x = pd.DataFrame(test_x, columns=x_col)

        train = pd.concat([train_x, train_y], axis=1)
        test = pd.concat([test_x, test_y], axis=1)

        return train, test

    def Feature_Remove(self, df1=None, df2=None, df3=None, df4=None):
        drop_feature = ['index', 'pupil_size', 'Iris_X', 'Iris_Y', 'diam']
        try:
            df1 = df1.drop(drop_feature, axis=1)
            df2 = df2.drop(drop_feature, axis=1)
            df3 = df3.drop(drop_feature, axis=1)
            df4 = df4.drop(drop_feature, axis=1)
        except KeyError:
            print('해당 컬럼이 없습니다.')
            pass

        return df1, df2, df3, df4

    def Feature_Selection(self, train_x, train_y, test_x, test_y):
        """
        Feature Importance
        1. Random Forest
        2. XGBoost
        3. LightGBM

        Label Correlation
        1. Find Common Feature

        => Importance와 Correlation의 공통분모 찾기.
        """

        # Label Correlation
        corr = self.Label_correlation(train_x, train_y)
        corr.to_csv("./corr.csv")

        # RandomForest
        print("RandomForest")
        rf_model = RandomForestClassifier(n_estimators=100, n_jobs=1)
        rf_model.fit(train_x, train_y)
        pred = rf_model.predict(test_x)
        report = classification_report(test_y, pred)
        print(str(report))
        rf_importance = rf_model.feature_importances_
        rf_importance = pd.DataFrame(rf_importance)
        rf_importance.to_csv("./rf_importance.csv")

        # XGBoost
        print("XGBoost")
        xgb_model = xgb.XGBClassifier(n_estimators=100, n_jobs=1, learning_rate=0.01)
        xgb_model.fit(train_x, train_y)
        pred = xgb_model.predict(test_x)
        report = classification_report(test_y, pred)
        print(str(report))
        xgb_importance = xgb_model.feature_importances_
        xgb_importance = pd.DataFrame(xgb_importance)
        xgb_importance.to_csv("./xgb_importance.csv")

        # LightGBM
        print("LightGBM")
        lgbm_model = lgbm.LGBMClassifier(n_estimators=100, n_jobs=1, learning_rate=0.01)
        lgbm_model.fit(train_x, train_y)
        pred = lgbm_model.predict(test_x)
        report = classification_report(test_y, pred)
        print(str(report))
        lgbm_importance = lgbm_model.feature_importances_
        lgbm_importance = pd.DataFrame(lgbm_importance)
        lgbm_importance.to_csv("./lgbm_importance.csv")

    def outlier_remove(self, data, file_name=None):
        """

        OF_1 : Total Optical Flow에서 1000이상 값부터 이후 100 frame 제거
        제거 후, 제거 전의 OF1과 제거 후의 OF1 각각 알코올 라벨과의 상관관계 확인
        """
        eye_blinking_idx = data[(data['OF_1'] >= 1000)].index
        print(len(eye_blinking_idx))
        ke = 0
        for k, eb_idx in enumerate(eye_blinking_idx):
            ## eye blinking 발생 후, 100frame 제거
            for s in range(100):
                rm_idx = eb_idx + s
                try:
                    data = data.drop(rm_idx, axis=0)
                except KeyError:
                    ke += 1

        print("Key Error count:", ke)
        df = data.reset_index(drop=True)
        print(df)
        col = data.columns
        Outliers_to_drop = self.IQR_outlier(df, 2, col)

        ## outlier row 확인
        df = df.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
        print(df)
        df.to_csv(eyeb_path + file_name + ".csv", index=False)
        return df

    def IQR_outlier(self, df, n, features):
        """

        eye blinking 제거 후, outlier 확인해서 제거
        제거 후, Tree 모델 Learning
        """
        outlier_indices = []
        for col in features:
            Q1 = np.percentile(df[col], 25)
            Q3 = np.percentile(df[col], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
            outlier_indices.extend(outlier_list_col)
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        return multiple_outliers


if __name__ == '__main__':
    exp_2()

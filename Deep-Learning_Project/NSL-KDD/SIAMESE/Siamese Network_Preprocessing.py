## Preprocessing

import numpy as np
import pandas as pd
from dask_ml.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from xgboost import XGBClassifier


class DataTransformer:
    def __init__(self):
        self.fs_model = None
        self.norm = None

    ''' Feature Selection - Wrapper method(Feature Importance) '''
    def FS_Wrapper(self, df_train, df_test, fs_type=None, features=None):
        if fs_type == 'rf':
            self.fs_model = RandomForestClassifier(random_state=42, n_estimators=10, n_jobs=-1)
        elif fs_type == 'xg':
            self.fs_model = XGBClassifier()
        self.fs_model.fit(df_train_num, train_y)
        imp = self.fs_model.feature_importances_

        return df_train, df_test

    ''' Data Normalization '''
    def preprocess_norm(self, df_train, df_test, norm_type=None):
        if norm_type == 'minmax':
            self.norm = MinMaxScaler(feature_range=(1, 255))
        elif norm_type == 'qnt':
            self.norm = QuantileTransformer()

        self.norm.fit(df_train_num)
        df_train_num = self.norm.transform(df_train_num)
        df_test_num = self.norm.transform(df_test_num)

        return x_train, x_test

    ''' Onehot Encoding Object Feature '''
    def toOnehot(self, df_train, df_test, label=None):
        obj_col = list(df_train.select_dtypes('object').columns)
        if label:
            obj_col.remove(label)
        x_train = pd.get_dummies(df_train, columns=obj_col)  # One-Hot Encoding
        x_test = pd.get_dummies(df_test, columns=obj_col)  # One-Hot Encoding
        return x_train, x_test

''' NSL-KDD Dataset Transform '''

class KDD_Transformer(DataTransformer):
    def __init__(self, label='outcome'):
        self.label = label
        self.encoder = LabelEncoder()

    def transform(self, train_path, test_path, preprocess='minmax', ctype='binary', fs_type=None, save_path=None):
        ''' Irrelevant Feature Remove'''
        df_train = df_train.drop('difficulty', axis=1)

        df_test = df_test.drop('difficulty', axis=1)

        ''' Data Type Conversion '''
        x_train, x_test = self.toOnehot(x_train, x_test, label=self.label)
        ''' Data Normalization '''
        self.preprocess_norm(df_train, df_test, norm_type=preprocess)

        if ctype == 'category':
            print('category')
            self.toCategoryAttack(x_train)
            self.toCategoryAttack(x_test)
        elif ctype == 'binary':
            print('binary')
            self.toBinaryAttack(x_train)
            self.toBinaryAttack(x_test)
        return x_train, x_test

    ''' Label Transform '''
    def toBinaryAttack(self, df):
        attack = df.outcome.values
        attack_type = [binary_attack_map[attack[i]] for i in range(len(attack))]
        df.outcome = attack_type
        return df

    ''' Label Transform '''
    def toCategoryAttack(self, df):
        attack = df.outcome.values
        attack_type = [kdd_attack_map[attack[i]] for i in range(len(attack))]
        df.outcome = attack_type
        return df

if __name__ == '__main__':
    train_path = '../dataset/KDDTrain+.txt'
    test_path = '../dataset/KDDTest+.txt'
    kt = KDD_Transformer()
    x_train, x_test = kt.transform(train_path, test_path, preprocess='qnt', ctype='category',
                                   fs_type='None', save_path="../dataset")
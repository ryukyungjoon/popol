from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np

class Semi_balancing:
    def __init__(self):
        pass

    def nsl_sampling(self, x, y, features, sampling_strategy):
        binary_train_up_dic = {
            'normal': 80000,
            'attack': 130000
        }

        train_up_dic = {
            'normal': 67343,
            'DoS': 45927,
            'Probe': 15000,          # 11656 -> 15000
            'R2L': 15000,            # 995 -> 15000
            'U2R': 15000             # 52 -> 15000
        }

        test_dic = {
            'normal': 9711,
            'DoS': 7458,
            'Probe': 2421,
            'R2L': 2887,
            'U2R': 67
        }

        print("Data Resampling...")

        train_sm = SMOTE(kind='regular', sampling_strategy=train_up_dic, random_state=0)
        a, b = train_sm.fit(x, y)

        label = ['outcome']
        sampled_x = pd.DataFrame(a, columns=list(features))
        sampled_y = pd.DataFrame(b, columns=list(label))

        data_resampled = pd.concat([sampled_x, sampled_y], axis=1)

        print(data_resampled)

        print("After OverSampling, the shape of train_x:{}".format(data_resampled.shape))
        print("After OverSampling, the shape of train_x:{} \n".format(data_resampled.shape))

        data_resampled.to_csv('../dataset/nsl-kdd balancing ['+sampling_strategy+'].txt', mode='w', index=False)

        return sampled_x, sampled_y
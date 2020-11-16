import numpy as np

class Cleaning():
    def data_cleaning(data):
        print("Data Cleaning...")
        data1 = data.dropna(axis =0, how='any')  # Remove Null values records
        y, x = data1['Label'], data1.drop('Label', 1)
        df = x.astype('float')  # dataFrame 타입 변환 [object->float64]
        df = df.replace(np.inf, np.nan)  # inf 값을 nan 값으로 replace
        df_max = df.max().max()
        x = df.replace(np.nan, df_max)  # inf 값을 제외한 값 중, 가장 큰 값인 655453030.0을 inf값 대신 넣어준다.

        return x, y

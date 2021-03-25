from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

import pandas as pd

class Norm:
    def __init__(self):
        pass

    def normalizations(self, data, normalization_type, imp, subsample):
        print('Data Normalizing...')
        fe_data = pd.DataFrame(data)
        remain_features = fe_data.head(n=0)

        y, x = fe_data['Label'], fe_data.drop('Label', 1)
        x = x.dropna(axis=0)
        x = x[(x.T != 0).any()]

        if normalization_type == 'mms':
            scaler = MinMaxScaler(feature_range=(0, 255))

        elif normalization_type == 'std':
            scaler = StandardScaler()

        else:
            scaler = QuantileTransformer(n_quantiles=15, subsample=3532583)     # subsample parameter

        scale_x = scaler.fit_transform(x)
        scale_x = pd.DataFrame(scale_x)
        scale_y = pd.DataFrame(y)
        print(scale_x, scale_y)

        scaled_data = pd.concat([scale_x, scale_y], axis=1)
        print(scaled_data)
        scaled_data = pd.DataFrame(scaled_data)
        scaled_data.to_csv("../dataset/Normalization/["+imp+"] " + normalization_type + ".csv", header=list(remain_features), index=False)

        return scaled_data





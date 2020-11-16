from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

import pandas as pd

class Data_Normalization:
    def normalizations(data, normalization_type, subsample):
        print('Data Normalizing...')
        fe_data1 = pd.DataFrame(data)
        # fe_data1 = fe_data1.rename(columns={'Fwd Header Length.1': 'Fwd Header Length'})
        y, x = fe_data1['Label'], fe_data1.drop('Label', 1)
        x.dropna(axis=0)
        x = x[(x.T != 0).any()]
        remain_features = fe_data1.head(n=0)

        if normalization_type == 'mms':
            mms = MinMaxScaler(feature_range=(0, 255))
            print("mms: ")
            mms_data = mms.fit_transform(x)
            mms_data = pd.DataFrame(mms_data)
            y = pd.DataFrame(y)
            norm_set = mms_data[(mms_data.T != 0).any()]  # Remove Zero records
            norm_train = pd.concat([norm_set, y], axis=1)

        if normalization_type == 'std':
            std = StandardScaler()
            print("std: ")
            x_scale = std.fit_transform(x)
            x_scale = pd.DataFrame(x_scale)
            y = pd.DataFrame(y)
            norm_set = x_scale[(x_scale.T != 0).any()]  # Remove Zero records
            norm_train = pd.concat([norm_set, y], axis=1)

        if normalization_type == 'qnt':
            qnt_train = QuantileTransformer(n_quantiles=15, subsample=3532583)
            x_train = qnt_train.fit_transform(x)
            print("qnt: ")
            qnt_train = pd.DataFrame(x_train)
            norm_train = qnt_train[(qnt_train.T != 0).any()]  # Remove Zero records
            norm_train = pd.concat([norm_train, y], axis=1)


        print(norm_train)

        norm_train.to_csv("../dataset/fin_dataset/cicids_origin"+normalization_type+".csv", header=list(remain_features), index=False)

        return norm_train
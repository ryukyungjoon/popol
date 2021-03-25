import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from NSLKDD_Feature_Extraction import Feature_Extraction

from Data_split import Data_split as ds
from Train_model import DNN as dnn
from Drawing import Drawing as dw
from Semi_balancing import Semi_balancing as sb

from Train_model import Train_model as tm

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

class Main:
    if __name__ == "__main__":
        data_loc = "../dataset/NSL-KDD/"
        data_format = ".csv"
        train_data_file = "qnt_KDDTrain_category"
        test_data_file = "qnt_KDDTest_category"
        data_format_txt = ".txt"

        print("Data Loading...")
        train_x, train_y = ds._load_data_txt(data_loc + train_data_file + data_format_txt)
        test_x, test_y = ds._load_data_txt(data_loc + test_data_file + data_format_txt)
        print(Counter(test_y))
        print(Counter(train_y))
        features = train_x.head(0)

        # fe = Feature_Extraction()
        # train_data, remain_features = fe.feature_extraction(train_X, train_Y, features)
        # train_Y, train_X = train_data['outcome'], train_data.drop('outcome', 1)

        # drop_features = Counter(features) - Counter(remain_features)
        # drop_features = list(drop_features)
        # print(features)
        # print(remain_features)
        # print(drop_features)

        train_dic = {
            'normal': 1000,
            'Probe': 1000,
            'DoS': 1000,
        }

        print("Data Resampling...")

        train_sm = RandomUnderSampler(sampling_strategy=train_dic, random_state=0)
        a, b = train_sm.fit_sample(train_x, train_y)

        label = ['outcome']
        train_x = pd.DataFrame(a, columns=list(features))
        train_y = pd.DataFrame(b, columns=list(label))
        train_x = np.array(train_x)
        test_x = np.array(test_x)

        classes_y = np.unique([test_y])
        print(classes_y)
        features = list(features)
        test_x = pd.DataFrame(test_x, columns=features)
        print(test_x)

        # # Test dataset's drop features
        # for i in range(len(drop_features)):
        #     test_X = test_X.drop(drop_features[i], axis=1)
        # print(test_X)

        # test_X = np.array(test_X)

        # Scikit-Learn Model
        # use_model_confusion_matrix, test_Y_pred = tm.train_model(train_X, train_Y, classes_y, test_X, test_Y)
        # print(use_model_confusion_matrix)

        # Deep Neural Network Learning
        use_model_confusion_matrix, history, pred, y_test = dnn.dnn_model(train_x, train_y, test_x, test_y, norm_type='qnt')
        print(pred)

        dw.print_confusion_matrix(use_model_confusion_matrix, classes_y, normalize=False)

        raw_encoded, raw_cat = test_y.factorize()

        y_class = np.unique([raw_encoded])
        acc = accuracy_score(y_test, pred)
        print(acc)
        report = classification_report(y_test, pred, labels=y_class, target_names=classes_y)
        print(str(report))
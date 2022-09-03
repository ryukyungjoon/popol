import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

from sklearn.neural_network import MLPClassifier

from keras.optimizers import Adam
from keras.layers import Dense

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

cic_features = ['Destination Port',
                'Flow Duration',
                'Total Fwd Packets',
                'Total Backward Packets',
                'Total Length of Fwd Packets',
                'Total Length of Bwd Packets',
                'Fwd Packet Length Max',
                'Fwd Packet Length Min',
                'Fwd Packet Length Mean',
                'Fwd Packet Length Std',
                'Bwd Packet Length Max',
                'Bwd Packet Length Min',
                'Bwd Packet Length Mean',
                'Bwd Packet Length Std',
                'Flow Bytes/s',
                'Flow Packets/s',
                'Flow IAT Mean',
                'Flow IAT Std',
                'Flow IAT Max',
                'Flow IAT Min',
                'Fwd IAT Total',
                'Fwd IAT Mean',
                'Fwd IAT Std',
                'Fwd IAT Max',
                'Fwd IAT Min',
                'Bwd IAT Total',
                'Bwd IAT Mean',
                'Bwd IAT Std',
                'Bwd IAT Max',
                'Bwd IAT Min',
                'Fwd PSH Flags',
                'Bwd PSH Flags',
                'Fwd URG Flags',
                'Bwd URG Flags',
                'Fwd Header Length',
                'Bwd Header Length',
                'Fwd Packets/s',
                'Bwd Packets/s',
                'Min Packet Length',
                'Max Packet Length',
                'Packet Length Mean',
                'Packet Length Std',
                'Packet Length Variance',
                'FIN Flag Count',
                'SYN Flag Count',
                'RST Flag Count',
                'PSH Flag Count',
                'ACK Flag Count',
                'URG Flag Count',
                'CWE Flag Count',
                'ECE Flag Count',
                'Down/Up Ratio',
                'Average Packet Size',
                'Avg Fwd Segment Size',
                'Avg Bwd Segment Size',
                'Fwd Header Length',
                'Fwd Avg Bytes/Bulk',
                'Fwd Avg Packets/Bulk',
                'Fwd Avg Bulk Rate',
                'Bwd Avg Bytes/Bulk',
                'Bwd Avg Packets/Bulk',
                'Bwd Avg Bulk Rate',
                'Subflow Fwd Packets',
                'Subflow Fwd Bytes',
                'Subflow Bwd Packets',
                'Subflow Bwd Bytes',
                'Init_Win_bytes_forward',
                'Init_Win_bytes_backward',
                'act_data_pkt_fwd',
                'min_seg_size_forward',
                'Active Mean',
                'Active Std',
                'Active Max',
                'Active Min',
                'Idle Mean',
                'Idle Std',
                'Idle Max',
                'Idle Min',
                'Label']

down_dic = {
            'BENIGN': 250000,
            'Bot': 1956,
            'DDoS': 128025,
            'DoS GoldenEye': 10293,
            'DoS Hulk': 230124,
            'DoS Slowhttptest': 5499,
            'DoS slowloris': 5796,
            'FTP-Patator': 7935,
            'Heartbleed': 11,
            'Infiltration': 36,
            'SSH-Patator': 5897,
            'PortScan': 158804,
            'Web Attack Brute Force': 1507,
            'Web Attack Sql Injection': 21,
            'Web Attack XSS': 652
        }
up_dic = {
            'BENIGN': 250000,
            'Bot': 5000,
            'DDoS': 128025,
            'DoS GoldenEye': 10293,
            'DoS Hulk': 230124,
            'DoS Slowhttptest': 5499,
            'DoS slowloris': 5796,
            'FTP-Patator': 7935,
            'Heartbleed': 5000,
            'Infiltration': 5000,
            'SSH-Patator': 5897,
            'PortScan': 158804,
            'Web Attack Brute Force': 5000,
            'Web Attack Sql Injection': 5000,
            'Web Attack XSS': 5000
        }


class Preprocessing:
    def __init__(self):
        pass

    ''' Data Normalization '''
    def normalization(self, train_x, train_y, test_x, test_y, norm_type):
        if norm_type == 'mms':
            self.norm_tool = MinMaxScaler(feature_range=(0, 255))
        elif norm_type == 'std':
            self.norm_tool = StandardScaler()
        elif norm_type == 'qnt':
            self.norm_tool = QuantileTransformer(n_quantiles=15, subsample=3532583)
        else:
            print("None Normalization Type")
            exit(0)
        self.norm_tool.fit(train_x, train_y)
        norm_trainx, norm_trainy = self.norm_tool.transform(train_x, train_y)
        norm_testx, norm_testy = self.norm_tool.transform(test_x, test_y)

        return norm_trainx, norm_trainy, norm_testx, norm_testy

    ''' Feature Selection '''
    def feature_selection(self, train_x, train_y, test_x, test_y, fs_type):
        ''' data type에 따른 분류 (number, object)'''
        num_col = train_x.select_dtypes('number').columns
        obj_col = train_x.select_dtypes('object').columns

        train_num, train_obj = train_x[num_col], train_x[obj_col]
        test_num, test_obj = test_x[num_col], test_x[obj_col]

        ''' No.0 is HYBRID FEATURE SELECTION '''
        if fs_type == 0:
            """ WRAPPER METHOD """
            fs_model = RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1)
            fs_model.fit(train_num, train_y)
            imp = fs_model.feature_importances_
            indices = np.argsort(imp)[::-1]

            remain_idx = []
            for k in range(len(imp)):
                if imp[indices[k]] >= 0.001:
                    remain_idx.append(indices[k])

            train_num = train_num[remain_idx]
            test_num = test_num[remain_idx]

            """ FILTER METHOD """
            fs_model = train_num.rank().corr(method="pearson")
            corr_table = pd.DataFrame(fs_model)
            columns = np.full((corr_table.shape[0],), True, dtype=bool)

            for i in range(corr_table.shape[0]):
                for j in range(i + 1, corr_table.shape[0]):
                    if corr_table.iloc[i, j] >= 0.95:
                        if columns[j]:
                            columns[j] = False

            train_num = train_num.columns[columns]
            test_num = test_num.columns[columns]

            if obj_col:
                train_x = pd.concat([train_num, train_obj], axis=1)
                test_x = pd.concat([test_num, test_obj], axis=1)
            else:
                train_x, test_x = train_num, test_num

            return train_x, test_x

        elif fs_type == 1:
            ''' No.1 is WRAPPER FEATURE SELECTION '''
            return 0
        elif fs_type == 2:
            ''' No.2 is FILTER FEATURE SELECTION '''
            return 0
        else:
            print('No Selected FEATURE SELECTION TYPE')
            return 0

''' CICIDS 2018 Dataset Preprocessing '''
class CIC_Transform(Preprocessing):
    def __init__(self, data, label='Label', norm_type=None, fs_type=None):
        self.y_label = label
        self.data = data
        self.norm_type = norm_type
        self.fs_type = fs_type
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

''' Data Clearning '''
    def cic_cleaning(self):
        print("Data Cleaning...")
        data1 = self.data.dropna(axis=0, how='any')
        y, x = data1[self.y_label], data1.drop(self.y_label, 1)
        df = x.astype('float')
        df = df.replace(np.inf, np.nan)
        df_max = df.max().max()
        x = df.replace(np.nan, df_max)

        return x, y

''' Train & Test Split '''
    def cic_split(self):
        print("[train_set] <=> [test_set]")

        train_set, test_set = train_test_split(self.data, test_size=0.3)

        self.train_y, self.train_x = train_set['Label'], train_set.drop('Label', 1)
        self.test_y, self.test_x = test_set['Label'], test_set.drop('Label', 1)

''' Imbalance Data  '''
    def balancing(self):
        rus = RandomUnderSampler(sampling_strategy=down_dic, random_state=0)
        sm = SMOTE(kind='regular', sampling_strategy=up_dic, random_state=0)

        x_resampled, y_resampled = rus.fit_sample(self.train_x, self.train_y)
        x_resampled_1, y_resampled_1 = sm.fit_sample(x_resampled, y_resampled)
        self.train_x = pd.DataFrame(x_resampled_1)
        self.train_y = pd.DataFrame(y_resampled_1)

        print("After OverSampling, the shape of data:{}\n".format(self.train_x.shape, self.train_y.shape))

        return self.train_x, self.train_y

    def data_transform(self, save=None):
        x, y = self.cic_cleaning()
        classes_y = np.unique([y])
        self.cic_split()
        train_x, train_y, test_x, test_y = self.normalization(self.train_x, self.train_y,
                                                              self.test_x, self.test_y, norm_type=self.norm_type)
        train_x, test_x = self.feature_selection(train_x, train_y, test_x, test_y, self.fs_type)
        train_x, train_y = self.balancing()

        if save:
            data.to_csv('save_loc')

class Traninng:
    def __init__(self, train_x, train_y, test_x, test_y, batch_size, epoch, model_name):
        self.model = None
        self.batch_size = batch_size
        self.learning_rate = 0.01
        self.optimizer = Adam(lr=self.learning_rate)
        self.epochs = epoch

        if model_name == 'dnn':
            self.dnn(train_x, train_y, test_x, test_y)
        if model_name == 'mlp':
            self.mlp(train_x, train_y, test_x, test_y, mini_batch_size=batch_size)

    def dnn(self, train_x, train_y, test_x, test_y):
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=(78, 1), name="Input_Layer"))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(15, activation='softmax'))

    def mlp(self, train_x, train_y, test_x, test_y, mini_batch_size):
        self.model = MLPClassifier(hidden_layer_sizes=(1000, 500, 100), max_iter=100, random_state=42)
        batch_size = len(train_y)
        total_epoch = int(batch_size / mini_batch_size)
        current_batch = 0
        start = time.time()
        for i in range(1, total_epoch):
            end_batch = i * mini_batch_size
            self.model.partial_fit(train_x[current_batch:end_batch], train_y[current_batch:end_batch], classes=classes_y)
            current_batch = end_batch
        self.model.partial_fit(train_x[current_batch:batch_size], train_y[current_batch:batch_size], classes=classes_y)
        end = time.time()


if __name__ == "__main__":
    data_loc = "../dataset/fin_dataset/nonmoon/"
    data_format = ".csv"
    data_file = "11. merge&delete_set"

    """
    Data Load
    """
    data = pd.read_csv(data_loc + data_file + data_format, sep=',', dtype='unicode')

    """
    Data Preprocessing
    """
    cic_preprocessing = CIC_Transform(data, label='Label', norm_type='mms', fs_type='hb')
    cic_preprocessing.data_transform()

    """
    Training
    """
    train = Training(train_x, trina_y, test_x, test_y, batch_size, epoch, model_name)

    raw_encoded, raw_cat = data_Label.factorize()
    print(raw_encoded, raw_cat)
    print(y_test)

    y_class = np.unique([re])
    print(y_class)

    """
    print_confusion_matrix & classification_report
    """
    dw.print_confusion_matrix(use_model_confusion_matrix, classes_y)
    report = classification_report(y_test, pred, labels=y_class, target_names=classes_y)
    print(str(report))

import pandas as pd
import numpy as np

from Data_cleaning import Cleaning as cl
from Feature_Extraction import Feature_Extraction
from Data_Normalization import Data_Normalization as dn
from Train_model import Train_model as tm
from Data_split import Data_split as ds
from Train_model import DNN as dnn
from Drawing import Drawing as dw

from sklearn.metrics import classification_report

class Main:
    if __name__ == "__main__":
        features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

        data_loc = "../dataset/fin_dataset/nonmoon/"
        data_format = ".csv"
        data_file = "11. merge&delete_set"

        print("Data Loading...")

        data = pd.read_csv(data_loc + data_file + data_format, sep=',', dtype='unicode')
        data, data_Label = cl.data_cleaning(data)
        classes_y = np.unique(data_Label)

        rc, re = data_Label.factorize()
        print(rc, np.unique([re]))
        fe = Feature_Extraction()
        fe = fe.feature_extraction_1(data, data_Label, features)

        print(fe)
        rows, columns = fe.shape
        print(rows)
        norm_data = dn.normalizations(fe, normalization_type='qnt', subsample=rows)

        norm_data = norm_data.dropna(axis=0)
        print(norm_data)
        train_X, train_Y, test_X, test_Y = ds.ae_split(norm_data)
        use_model_confusion_matrix, history, pred, y_test = dnn.dnn_model(train_X, train_Y, test_X, test_Y, norm_type='qnt')

        raw_encoded, raw_cat = data_Label.factorize()
        print(raw_encoded, raw_cat)
        print(y_test)


        y_class = np.unique([re])
        print(y_class)
        dw.print_confusion_matrix(use_model_confusion_matrix, classes_y)

        report = classification_report(y_test, pred, labels=y_class, target_names=classes_y)
        print(str(report))

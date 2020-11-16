from sklearn.metrics import classification_report

from model import LSTM as lstm
from result import print_result
from balancing import Semi_balancing

import pandas as pd
import numpy as np

data_path = "../dataset/"

nsl_tr_data = "nsl-kdd_balancing [qnt_up]"
nsl_te_data = "qnt_KDDTest_category"
nsl_data_format = ".txt"

train_data = pd.read_csv(data_path+nsl_tr_data+nsl_data_format, sep=',', dtype='unicode')
test_data  = pd.read_csv(data_path+nsl_te_data+nsl_data_format, sep=',', dtype='unicode')

train_y, train_x = train_data['outcome'], train_data.drop('outcome', 1)
test_y,  test_x  = test_data['outcome'], test_data.drop('outcome', 1)

# sb = Semi_balancing()
# train_x, train_y = sb.nsl_sampling(train_x, train_y, train_x.head(0), 'qnt')

train_x = np.array(train_x)
test_x  = np.array(test_x)

tr_samples, tr_feature_num = train_x.shape
te_samples, te_feature_num = test_x.shape

train_x = np.reshape(train_x, (tr_samples, 1, tr_feature_num))         # (n, 1, 78)
test_x = np.reshape(test_x, (te_samples, 1, te_feature_num))           # (n, 1, 78)
print(train_x.shape, test_x.shape)

rnn = RNN()
cm, history, pred, y_test = lstm.main(train_x, train_y, test_x, test_y)

re, rc = test_y.factorize()

class_names = np.unique([re])
classes_y = np.unique([test_y])
print(class_names)

report = classification_report(y_test, pred, labels=class_names, target_names=classes_y)
print(str(report))

pr = print_result()
pr.print_confusion_matrix(cm, classes_y)       # (혼동행렬, 클래스 이름)
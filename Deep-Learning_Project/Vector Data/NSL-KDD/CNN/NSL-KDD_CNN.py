from imblearn.under_sampling import RandomUnderSampler
from keras.layers import Dense, Input, Dropout, Lambda, Conv1D, MaxPooling1D, Flatten, Activation
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import to_categorical

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
import pandas as pd
import numpy.random as random

from Drawing import Drawing as dw

class CNN:
    def __init__(self, train_x, test_x, train_y, test_y):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y


        self.input_shape = self.train_x.shape
        self.num_classes = 5
        self.batch_size = 10

        ## String 2 Float
        l_encoder = LabelEncoder()
        self.y_train = l_encoder.fit_transform(train_y)
        self.y_test = l_encoder.fit_transform(test_y)

        ## One-Hot
        self.Onehot_train_Y2 = to_categorical(self.y_train, num_classes=5)
        self.Onehot_test_Y2 = to_categorical(self.y_test, num_classes=5)
        print('Original Data : {}'.format(train_y))
        print('Original Data : {}'.format(test_y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(self.Onehot_train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(self.Onehot_test_Y2))

        # train_x에서 train_y와 같은 값을 갖는 데이터끼리 그룹화함.
        self.train_groups = [train_x[np.where(train_y == i)[0]] for i in np.unique(train_y)]
        self.test_groups = [test_x[np.where(test_y == i)[0]] for i in np.unique(train_y)]

        print('train groups:', [x.shape[0] for x in self.train_groups])  # train 데이터의 class 별 데이터 개수
        print('test groups:', [x.shape[0] for x in self.test_groups])  # test 데이터의 class 별 데이터 개수

        self.x = [x.shape[0] for x in self.train_groups]
        self.z = [x.shape[0] for x in self.test_groups]

        # Feature Vector Network(CNN) 구성
        convnet = Sequential([
            Conv1D(filters=256, kernel_size=3, input_shape=(122, 1)),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=128, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=64, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=32, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(5, activation='softmax')
        ])

        # Siamese Neural Network 구성
        self.input_data = Input((122, 1), name='input_data')

        self.conv_ids = convnet(self.input_data)
        self.conv_ids = Model(inputs=self.input_data, outputs=self.conv_ids)

        adam = optimizers.Adam(lr=0.0001)

        self.conv_ids.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        self.conv_ids.summary()

    def main(self):
        loss_history = self.conv_ids.fit(self.train_x, self.Onehot_train_Y2,
                                         batch_size=32, epochs=100, verbose=1,
                                         validation_split=0.2)
        loss = loss_history.history['loss']
        val_loss = loss_history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Convolutional Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        classes_y = np.unique(self.test_y)
        print(classes_y)

        # Model Evaluate
        performance_test = self.conv_ids.evaluate(self.test_x, self.Onehot_test_Y2)
        print('Test Loss and Accuracy ->', performance_test)

        pred = self.conv_ids.predict(self.test_x)
        print(pred)
        pred = np.argmax(pred, axis=1)
        cm = confusion_matrix(np.argmax(self.Onehot_test_Y2, axis=1), pred)
        print(cm)

        dw.print_confusion_matrix(cm, classes_y, normalize=False)

        raw_encoded, raw_cat = self.test_y.factorize()

        y_class = np.unique([raw_encoded])
        acc = accuracy_score(self.y_test, pred)
        print(acc)
        report = classification_report(self.y_test, pred, labels=y_class, target_names=classes_y)
        print(str(report))


if __name__=='__main__':

    data_path = "../dataset/NSL-KDD/"

    nsl_tr_data = "qnt_KDDTrain_category"
    nsl_te_data = "qnt_KDDTest_category"
    nsl_data_format = ".txt"

    train_data = pd.read_csv(data_path + nsl_tr_data + nsl_data_format, sep=',', dtype='unicode')
    test_data = pd.read_csv(data_path + nsl_te_data + nsl_data_format, sep=',', dtype='unicode')

    train_y, train_x = train_data['outcome'], train_data.drop('outcome', 1)
    test_y, test_x = test_data['outcome'], test_data.drop('outcome', 1)
    # features = train_x.head(0)
    # train_dic = {
    #     'normal': 1000,
    #     'Probe': 1000,
    #     'DoS': 1000,
    # }
    #
    # print("Data Resampling...")
    #
    # train_sm = RandomUnderSampler(sampling_strategy=train_dic, random_state=0)
    # a, b = train_sm.fit_sample(train_x, train_y)
    #
    # label = ['outcome']
    # train_x = pd.DataFrame(a, columns=list(features))
    # train_y = pd.DataFrame(b, columns=list(label))
    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    siam = CNN(train_x, test_x, train_y, test_y)
    siam.main()
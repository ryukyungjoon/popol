import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt

## 3D convolution Layer
import os
import keras
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, MaxPool3D, \
    GlobalAveragePooling3D
from keras.utils import to_categorical
import numpy as np

data_path = r"E:\ryu_pythonProject\2. Cardivu-A\1. data_analysis\test_data/"

BL_path = data_path + "Before/left/"
BR_path = data_path + "Before/right/"
AL_path = data_path + "After/left/"
AR_path = data_path + "After/right/"

Before_L = ["[f]김환진_음주전_2021-05-17-181830-0000_M_L",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_L",
            "[f]이동원_음주전_2021-05-17-182348-0000_M_L",
            "[f]김환진_음주전_2021-06-09-183322-0000_M_L",
            "[f]문예성_음주전_2021-06-09-182014-0000_M_L",
            "[f]이동원_음주전_2021-06-09-182703-0000_M_L",
            "[f]신재민_음주전_2021-06-09-184016-0000_M_L"]

Before_R = ["[f]김환진_음주전_2021-05-17-181830-0000_M_R",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_R",
            "[f]이동원_음주전_2021-05-17-182348-0000_M_R",
            "[f]김환진_음주전_2021-06-09-183322-0000_M_R",
            "[f]문예성_음주전_2021-06-09-182014-0000_M_R",
            "[f]이동원_음주전_2021-06-09-182703-0000_M_R",
            "[f]신재민_음주전_2021-06-09-184016-0000_M_R"]

After_L = ["[f]김환진_음주후2_2021-05-17-203642-0000_M_L",
           "[f]문예성_음주후2_2021-05-17-203150-0000_M_L",
           "[f]이동원_음주후2_2021-05-17-202504-0000_M_L",
           "[f]김환진_음주후_2021-06-09-211918-0000_M_L",
           "[f]문예성_음주후_2021-06-09-213245-0000_M_L",
           "[f]이동원_음주후_2021-06-09-212501-0000_M_L",
           "[f]신재민_음주후_2021-06-09-213742-0000_M_L"]

After_R = ["[f]김환진_음주후2_2021-05-17-203642-0000_M_R",
           "[f]문예성_음주후2_2021-05-17-203150-0000_M_R",
           "[f]이동원_음주후2_2021-05-17-202504-0000_M_R",
           "[f]김환진_음주후_2021-06-09-211918-0000_M_R",
           "[f]문예성_음주후_2021-06-09-213245-0000_M_R",
           "[f]이동원_음주후_2021-06-09-212501-0000_M_R",
           "[f]신재민_음주후_2021-06-09-213742-0000_M_R"]


class exp_1:
    def __init__(self):
        train, test = self.data_load(0, 3)
        valid, test = self.data_load(3, 6)
        print(train)
        print(valid)
        print(test)
        train, test, valid = self.FS(train, test, valid)

        train_y, train_x = train['label'], train.drop('label', axis=1)
        test_y, test_x = test['label'], test.drop('label', axis=1)
        valid_y, valid_x = valid['label'], valid.drop('label', axis=1)

        self.autoencoder_classifier(train_x, train_y, test_x, test_y, valid_x, valid_y)

    def data_load(self, start, end):
        BL_1, AL_1, BR_1, AR_1 = None, None, None, None
        BL_2 = pd.DataFrame()
        AL_2 = pd.DataFrame()
        BR_2 = pd.DataFrame()
        AR_2 = pd.DataFrame()

        for i in range(start, end):
            if i == 0:
                BL_1 = pd.read_csv(BL_path + Before_L[i] + ".csv")
                BL_1 = BL_1.drop(index=0, axis=0)
                AL_1 = pd.read_csv(AL_path + After_L[i] + ".csv")
                AL_1 = AL_1.drop(index=0, axis=0)

                BR_1 = pd.read_csv(BR_path + Before_R[i] + ".csv")
                BR_1 = BR_1.drop(index=0, axis=0)
                AR_1 = pd.read_csv(AR_path + After_R[i] + ".csv")
                AR_1 = AR_1.drop(index=0, axis=0)
            else:
                BL_3 = pd.read_csv(BL_path + Before_L[i] + ".csv", index_col=None)
                BL_3 = BL_3.drop(index=0, axis=0)
                BL_2 = BL_2.append(BL_3)
                AL_3 = pd.read_csv(AL_path + After_L[i] + ".csv", index_col=None)
                AL_3 = AL_3.drop(index=0, axis=0)
                AL_2 = AL_2.append(AL_3)

                BR_3 = pd.read_csv(BR_path + Before_R[i] + ".csv", index_col=None)
                BR_3 = BR_3.drop(index=0, axis=0)
                BR_2 = BR_2.append(BR_3)
                AR_3 = pd.read_csv(AR_path + After_R[i] + ".csv", index_col=None)
                AR_3 = AR_3.drop(index=0, axis=0)
                AR_2 = AR_2.append(AR_3)

        # 왼쪽 눈
        BL = pd.concat([BL_1, BL_2], axis=0, ignore_index=True)
        AL = pd.concat([AL_1, AL_2], axis=0, ignore_index=True)
        BL = BL.assign(label=0)
        AL = AL.assign(label=1)
        train_L = pd.concat([BL, AL], axis=0, ignore_index=True)
        test_BL = pd.read_csv(BL_path + Before_L[6] + ".csv")
        test_AL = pd.read_csv(AL_path + After_L[6] + ".csv")
        test_BL = test_BL.assign(label=0)
        test_AL = test_AL.assign(label=1)
        test_L = pd.concat([test_BL, test_AL], axis=0, ignore_index=True)

        # 오른쪽 눈
        BR = pd.concat([BR_1, BR_2], axis=0, ignore_index=True)
        AR = pd.concat([AR_1, AR_2], axis=0, ignore_index=True)
        BR = BR.assign(label=0)
        AR = AR.assign(label=1)
        train_R = pd.concat([BR, AR], axis=0, ignore_index=True)
        test_BR = pd.read_csv(BR_path + Before_R[6] + ".csv")
        test_AR = pd.read_csv(AR_path + After_R[6] + ".csv")
        test_BR = test_BR.assign(label=0)
        test_AR = test_AR.assign(label=1)
        test_R = pd.concat([test_BR, test_AR], axis=0, ignore_index=True)

        train_L, test_L = self.norm(train_L, test_L)
        train_R, test_R = self.norm(train_R, test_R)

        train = pd.concat([train_L, train_R], axis=0, ignore_index=False)
        test = pd.concat([test_L, test_R], axis=0, ignore_index=False)

        return train, test

    def norm(self, train, test):
        train_y, train_x = train['label'], train.drop('label', axis=1)
        x_col = train_x.columns
        test_y, test_x = test['label'], test.drop('label', axis=1)
        mms = MinMaxScaler(feature_range=(0, 1))
        mms.fit(train_x, train_y)
        train_x = mms.transform(train_x)
        test_x = mms.transform(test_x)

        train_x = pd.DataFrame(train_x, columns=x_col)
        test_x = pd.DataFrame(test_x, columns=x_col)

        train = pd.concat([train_x, train_y], axis=1)
        test = pd.concat([test_x, test_y], axis=1)

        return train, test

    def FS(self, train, test, valid):
        drop_feature = ['index', 'pupil_size', 'Iris_X', 'Iris_Y', 'diam']
        train = train.drop(drop_feature, axis=1)
        test = test.drop(drop_feature, axis=1)
        valid = valid.drop(drop_feature, axis=1)

        return train, test, valid

    def encoder(self, input_data):
        enc = Dense(64, activation='relu')(input_data)
        enc = Dense(32, activation='relu')(enc)
        enc = Dense(32, activation='relu')(enc)
        enc = Dense(16, activation='relu')(enc)
        latent_dim = Dense(16)(enc)
        encoder_model = Model(input_data, latent_dim, name='encoder')
        return latent_dim, encoder_model

    def decoder(self, latent_dim):
        dec = Dense(16, activation='relu')(latent_dim)
        dec = Dense(32, activation='relu')(dec)
        dec = Dense(32, activation='relu')(dec)
        dec = Dense(64, activation='relu')(dec)
        output = Dense(241)(dec)
        return output

    def autoencoder(self, input_data):
        enc_output, enc = self.encoder(input_data)
        ae = self.decoder(enc_output)
        model = Model(inputs=input_data, outputs=ae, name='Autoencoder')
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        model.summary()
        return model, enc_output

    def draw_loss_graph(self, history, title):
        """

        :param history: model's history
        :param title: plot title
        :return: plot graph show()
        """

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.axis([0, 200, 0, 1])
        plt.title(title)
        plt.legend()
        plt.show()

    def classifier(self, input_data, enc_output):
        cls = Dense(10, activation='relu')(enc_output)
        cls = Dense(2, activation='softmax')(cls)
        classifier_model = Model(inputs=input_data, outputs=cls, name='classifier_model')
        return classifier_model

    def autoencoder_classifier(self, train_x, train_y, test_x, test_y, valid_x, valid_y):
        # One-hot Encoding
        train_y = to_categorical(train_y, num_classes=2)
        te_y = to_categorical(test_y, num_classes=2)
        valid_y = to_categorical(valid_y, num_classes=2)

        input_data = Input(shape=(241,))
        model, enc_output = self.autoencoder(input_data)
        es = EarlyStopping(monitor='val_loss', patience=15)
        history = model.fit(train_x, train_x, batch_size=16, verbose=1, epochs=100, validation_data=[valid_x, valid_x],
                            callbacks=[es])
        # draw_loss_graph(history, title="cardivu-A Autoencoder")
        cls_model = self.classifier(input_data, enc_output)
        for i, layer in enumerate(cls_model.layers[0:6]):
            layer.set_weights(model.layers[i].get_weights())
            layer.trainable = False

        cls_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        cls_model.summary()
        cls_model.fit(train_x, train_y, batch_size=16, verbose=1, epochs=100, validation_data=[valid_x, valid_y],
                      callbacks=[es])

        for i, layer in enumerate(cls_model.layers[0:6]):
            layer.trainable = True

        cls_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        cls_model.summary()
        history = cls_model.fit(train_x, train_y, batch_size=16, verbose=1, epochs=100, validation_data=[valid_x, valid_y],
                                callbacks=[es])
        self.draw_loss_graph(history, title="cardivu-A Autoencoder cls")
        pred = cls_model.predict(test_x)
        print(pred)
        pred = np.argmax(pred, axis=1)
        print(pred, test_y)

        cm = confusion_matrix(test_y, pred)
        report = classification_report(test_y, pred)
        print(cm)
        print(str(report))


if __name__ == '__main__':
    exp_1()

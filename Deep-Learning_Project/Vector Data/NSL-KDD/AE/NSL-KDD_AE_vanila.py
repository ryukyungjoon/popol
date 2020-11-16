from Data_cleaning import Cleaning as cl
from Data_split import Data_split as ds
from Data_Normalization_AE import Data_Normalization_AE as dn
from Drawing import Drawing as dw

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import random

random.seed(0)

encoding_h = [1000, 500, 250, 125, 80]
encoding_out = 61
decoding_h = [80, 125, 250, 500, 1000]
decoding_output = 122
class Autoencoder_Test():
    def __init__(self):
        self.input_dim = 122
        self.latent_dim = 61
        self.num_classes = 5
        self.batch_size = 5000
        self.epochs = 100

        input_img = Input(shape=(self.input_dim))
        latent_img = Input(shape=(self.latent_dim))

        encode_output = self.build_encoder(input_img)
        decode_output = self.build_decoder(latent_img)

        # autoencoder 모델 구성
        self.autoencoder_model = Model(input_img, self.build_decoder(self.build_encoder(input_img)))
        self.autoencoder_model.compile(loss='mean_squared_error', optimizer='adam')
        self.autoencoder_model.summary()

        self.full_model = Model(input_img, self.build_fc(encode_output))
        self.full_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
        self.full_model.summary()
        # for i, layer in enumerate(self.full_model.layers[0:7]):                     # Weights Transfer & Freeze , 7 is encoder layers num
        #     layer.set_weights(self.autoencoder_model.layers[i].get_weights())
        #     layer.trainable = False
        # self.full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        # self.full_model.summary()

    def build_encoder(self, input_img):
        enc_1 = Dense(encoding_h[0], activation='relu', name='encoder-1')(input_img)
        enc_2 = Dense(encoding_h[1], activation='relu', name='encoder-2')(enc_1)
        enc_3 = Dense(encoding_h[2], activation='relu', name='encoder-3')(enc_2)
        enc_4 = Dense(encoding_h[3], activation='relu', name='encoder-4')(enc_3)
        enc_5 = Dense(encoding_h[4], activation='relu', name='encoder-5')(enc_4)
        enc_output = Dense(encoding_out, name='encoder-output')(enc_5)

        return enc_output

    def build_decoder(self, latent_img):
        dec_1 = Dense(decoding_h[0], activation='relu', name='decoder-1')(latent_img)
        dec_2 = Dense(decoding_h[1], activation='relu', name='decoder-2')(dec_1)
        dec_3 = Dense(decoding_h[2], activation='relu', name='decoder-3')(dec_2)
        dec_4 = Dense(decoding_h[3], activation='relu', name='decoder-4')(dec_3)
        dec_5 = Dense(decoding_h[4], activation='relu', name='decoder-5')(dec_4)
        dec_output = Dense(decoding_output, name='decoder-output')(dec_5)

        return dec_output

    def build_fc(self, enco):
        den = Dense(self.input_dim, activation='relu')(enco)
        out = Dense(self.num_classes, activation='softmax')(den)
        return out

    def main(self):
        data_loc = "../dataset/NSL-KDD/"
        train_data_file = "minmax_KDDTrain"
        test_data_file ="minmax_KDDTest"
        data_format_txt = ".txt"

        print("Data Loading...")
        # data = pd.read_csv(data_loc + data_file + data_format, sep=',', dtype='unicode')
        train_X, train_Y = ds._load_data_txt(data_loc+train_data_file+data_format_txt)
        test_X, test_Y = ds._load_data_txt(data_loc+test_data_file+data_format_txt)
        # train_X, test_X, train_Y, test_Y = dn.normalizations(train_X, test_X, train_Y, test_Y, 'qnt')
        print(train_X.iloc[[1]].dtypes)

        train_X = np.array(train_X)
        test_X = np.array(test_X)

        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


        l_encoder = LabelEncoder()
        y_train = l_encoder.fit_transform(train_Y)
        y_test = l_encoder.fit_transform(test_Y)
        print(np.unique([train_Y]))
        print(train_Y)
        Onehot_train_Y2 = to_categorical(y_train, num_classes=5)
        Onehot_test_Y2 = to_categorical(y_test, num_classes=5)
        print('Original Data : {}'.format(train_Y))
        print('Original Data : {}'.format(test_Y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(Onehot_train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(Onehot_test_Y2))

        autoencoder_train = self.autoencoder_model.fit(train_X, train_X, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Autoencoder Training and validation loss')
        plt.legend()
        plt.show()

        print('== Full Model Start ==')

        classify_train = self.full_model.fit(train_X, Onehot_train_Y2, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)
        dw.loss_acc_graph(classify_train)

        print('======= test evaluation ========')

        test_eval = self.full_model.evaluate(test_X, Onehot_test_Y2, verbose=1)
        print('Test loss and Accuracy :', test_eval)
        pred = self.full_model.predict(test_X)
        pred = np.argmax(np.round(pred), axis=1)
        Onehot_test_Y = np.argmax(np.round(Onehot_test_Y2), axis=1)
        print('prediction : ', pred)
        print('test_Y :', Onehot_test_Y)

        cm = confusion_matrix(Onehot_test_Y, pred)
        print(cm)
        classes_y = np.unique([train_Y])  # y_train은 원핫인코딩 전 라벨인코더 단계를 거친 값
        print(classes_y)

        raw_encoded, raw_cat = test_Y.factorize()
        print(raw_encoded, raw_cat)
        print(y_test.shape, pred.shape)
        y_class = np.unique([raw_encoded])

        dw.print_confusion_matrix(cm, class_names=classes_y)
        cre = classification_report(y_test, pred, labels=y_class, target_names=classes_y)
        print(str(cre))

if __name__ == '__main__':
    ae_test = Autoencoder_Test()
    ae_test.main()
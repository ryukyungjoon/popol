from Data_cleaning import Cleaning as cl
from Data_split import Data_split as ds
from Data_Normalization_AE import Data_Normalization_AE as dn
from Drawing import Drawing as dw

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from tensorflow.keras.datasets import mnist

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 5

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

encoding_h = [1000, 500, 250, 125, 60]
encoding_out = 30
decoding_h = [60, 125, 250, 500, 1000]
decoding_output = 784
class Autoencoder_Test():
    def __init__(self):
        self.input_dim = 784
        self.latent_dim = 30
        self.num_classes = 10
        self.batch_size = 128
        self.epochs = 1

        input_img = Input(shape=(self.input_dim))
        latent_img = Input(shape=(self.latent_dim))

        encode_output = self.build_encoder(input_img)
        decode_output = self.build_decoder(latent_img)

        # autoencoder 모델 구성
        self.autoencoder_model = Model(input_img, self.build_decoder(self.build_encoder(input_img)))
        self.autoencoder_model.compile(loss='mean_squared_error', optimizer='adam')
        self.autoencoder_model.summary()

        # Full Freeze 모델 구성
        self.full_model = Model(input_img, self.build_fc(encode_output))
        self.full_model.summary()

        for i, layer in enumerate(self.full_model.layers[0:7]):                     # Weights Transfer & Freeze
            layer.set_weights(self.autoencoder_model.layers[i].get_weights())
            layer.trainable = False

        self.full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.full_model.summary()

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
        (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
        y_train = np.unique(train_Y)

        train_Y = to_categorical(train_Y)
        test_Y = to_categorical(test_Y)

        a, b, c = train_X.shape
        a1, b1, c1 = test_X.shape

        train_X = train_X.reshape(-1, b * c)
        test_X = test_X.reshape(-1, b1 * c1)

        train_X = train_X / 255.0
        test_X = test_X / 255.0

        autoencoder_train = self.autoencoder_model.fit(train_X, train_X, epochs=self.epochs, shuffle=True, batch_size=self.batch_size, validation_split=0.2)

        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('MNIST Autoencoder Training and validation loss')
        plt.legend()
        plt.show()

        print('== Full Model Start ==')
        print(train_X.shape, train_Y.shape)

        classify_train = self.full_model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)

        accuracy = classify_train.history['acc']
        val_accuracy = classify_train.history['val_acc']
        loss = classify_train.history['loss']
        val_loss = classify_train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('MNIST Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('MNIST Full Model Training and validation loss')
        plt.legend()
        plt.show()

        # UnFreeze model
        for layer in self.full_model.layers[0:7]:
            layer.trainable = True
        self.full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.full_model.summary()

        import time
        start = time.time()
        classify_train = self.full_model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)
        end = time.time()
        print(f'full_model Training Time{end-start}')

        # Save Weights & Model
        self.full_model.save_weights('MNIST_full_model_weights')
        full_model_json = self.full_model.to_json()
        with open('MNIST_full_model.json', 'w') as json_file:
            json_file.write(full_model_json)

        accuracy = classify_train.history['acc']
        val_accuracy = classify_train.history['val_acc']
        loss = classify_train.history['loss']
        val_loss = classify_train.history['val_loss']
        epochs = range(len(accuracy))

        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('MNIST Training and validation accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('MNIST Full Model Training and validation loss')
        plt.legend()
        plt.show()

        print('======= test evaluation ========')

        test_eval = self.full_model.evaluate(test_X, test_Y, batch_size=128, verbose=1)
        print('Test loss and Accuracy :', test_eval)
        pred = self.full_model.predict(test_X)

        pred = np.argmax(np.round(pred), axis=1)
        test_Y = np.argmax(np.round(test_Y), axis=1)
        print('prediction : ', pred)
        print('test_Y :', test_Y)

        cm = confusion_matrix(test_Y, pred)
        print(cm)
        classes_y = np.unique(y_train)
        classes_y = list(classes_y)
        print(classes_y)

        dw.print_confusion_matrix(cm, classes_y, normalize=True)
        report = classification_report(test_Y, pred, labels=classes_y, target_names=classes_y)
        print(str(report))

if __name__ == '__main__':
    ae_test = Autoencoder_Test()
    ae_test.main()
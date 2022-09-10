import warnings

warnings.filterwarnings('ignore')
## 3D convolution Layer
import os
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # then these two lines force keras to use your CPUsports-1m dataset
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, MaxPool3D, \
    GlobalAveragePooling3D, Conv2D
from keras.utils import to_categorical
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

path = r"C:\Users\SD_loacl\Desktop\CardiVu-A/"


class conv3d:
    def __init__(self):
        Before_x, B_label, After_X, A_label = self.data_load()
        print(np.shape(Before_x), np.shape(After_X))
        print(np.shape(B_label), np.shape(A_label))
        x = np.append(Before_x, After_X, axis=0)
        y = np.append(B_label, A_label, axis=0)
        y = to_categorical(y, num_classes=2)
        x = np.reshape(x, (-1, 16553, 180, 180, 3))
        print(np.shape(x))
        print(y)

        model = self.build_conv3d()
        history = model.fit(x, y, batch_size=1, epochs=2, validation_split=0.2,
                            verbose=1)
        self.draw_loss_graph(history, title='Conv3D Net')
        ## test code



    def data_load(self):
        Before_x = []
        B_label = []
        file_list = [_ for _ in os.listdir(path) if _.endswith(r".mp4")]
        non_alcohol = ["[Timesync]20210730-141115_L",
                       "[Timesync]20210730-141115_R",
                       "[Timesync]20210730-180302_L",
                       "[Timesync]20210730-180302_R",
                       "[Timesync]20210802-112054_L",
                       "[Timesync]20210802-112054_R",
                       "[Timesync]20210802-154358_L",
                       "[Timesync]20210802-154358_R"]

        for i in range(len(non_alcohol)):
            cap = cv2.VideoCapture(path + non_alcohol[i] + ".mp4")
            sample = []
            sl = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                sample.append(frame)
                sl = len(sample)
                if sl == 16553:
                    Before_x.append(sample)
                    B_label.append([0])
                    # print(np.shape(B_label))
                    sample = []

            print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        Before_x = np.array(Before_x)
        B_label = np.array(B_label, dtype='int32')

        After_X = []
        A_label = []
        file_list = [_ for _ in os.listdir(path) if _.endswith(r".mp4")]
        alcohol = ["[Timesync]20210730-190351_L",
                   "[Timesync]20210730-190351_R",
                   "[Timesync]20210804-192506_L",
                   "[Timesync]20210804-192506_R"]

        for i in range(len(alcohol)):
            cap = cv2.VideoCapture(path + alcohol[i] + ".mp4")
            sample = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                sample.append(frame)
                sl = len(sample)
                if sl == 16553:
                    After_X.append(sample)
                    A_label.append([1])
                    print(np.shape(A_label))
                    sample = []

            print('alcohol Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        After_X = np.array(After_X)
        A_label = np.array(A_label, dtype='int32')

        return Before_x, B_label, After_X, A_label

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

    def build_conv2d(self):
        input_layer = Input(shape=(90, 90, 3))

        conv = Conv2D

    def build_conv3d(self):
        ## input layer
        input_layer = Input(shape=(16553, 180, 180, 3))

        ## convolutional layers
        conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

        ## add max pooling to obtain the most imformatic features
        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

        conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
        conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
        pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

        ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pooling_layer2 = BatchNormalization()(pooling_layer2)
        flatten_layer = Flatten()(pooling_layer2)

        ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
        ## add dropouts to avoid overfitting / perform regularization
        dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=2, activation='softmax')(dense_layer2)

        ## define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        return model


if __name__ == '__main__':
    conv3d()

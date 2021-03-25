from keras.layers import Input, Lambda, Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam

import numpy as np
import pandas as pd

class SiameseNet:
    def __init__(self):
        left_input = Input((122, 1, ))
        right_input = Input((122, 1, ))

        Feature_net = Sequential([
            Conv1D(filters=5, kernel_size=3, activation='relu', input_shape=(122, 1, )),
            MaxPooling1D(),
            Conv1D(filters=5, kernel_size=3, activation='relu'),
            MaxPooling1D(),
            Conv1D(filters=7, kernel_size=2, activation='relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(18),
            Activation('sigmoid')
        ])
        encoded_l = Feature_net(left_input)
        encoded_r = Feature_net(right_input)

        L1_layer = Lambda(lambda tensor:K.abs(tensor[0]-tensor[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

        optimizer = Adam(lr=0.001, decay=2.5e-4)
        siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

if __name__ == '__main__':
    SiameseNet()

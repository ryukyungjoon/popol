from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Lambda
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import os
import numpy as np

np.random.seed(1337)

class LSTM:

    def __init__(self):
        self.input_dim = 122
        self.num_classes = 5
        self.batch_size = 64
        self.epoch = 100
        self.hidden_node = 80
        self.time_step = 1

        self.use_model = Sequential()
        self.use_model.add(LSTM(self.hidden_node, input_shape=(self.time_step, self.input_dim)))
        self.use_model.add(Dense(self.num_classes, activation='softmax'))

        adam = optimizers.Adam(learning_rate=0.01)
        self.use_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
        self.use_model.summary()

    def main(self, train_x, train_y, test_x, test_y):
        l_encoder = LabelEncoder()
        l_encoder.fit(train_y)
        y_train = l_encoder.transform(train_y)
        y_test = l_encoder.transform(test_y)
        Onehot_train_Y2 = to_categorical(y_train, num_classes=self.num_classes)
        Onehot_test_Y2 = to_categorical(y_test, num_classes=self.num_classes)
        print('Original Data : {}'.format(train_y))
        print('Original Data : {}'.format(test_y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(Onehot_train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(Onehot_test_Y2))

        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='min')

        history = self.use_model.fit(train_x, Onehot_train_Y2, epochs=self.epoch,
                                     batch_size=self.batch_size, verbose=1, validation_split=0.1, callbacks=[early_stop])

        pred = self.use_model.predict(test_x)
        pred = np.argmax(pred, axis=1)
        cm = confusion_matrix(np.argmax(Onehot_test_Y2, axis=1), pred)
        print(cm)

        # Model Save
        MODEL_PATH = '../dataset/h5 file/'
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        self.use_model.save_weights(MODEL_PATH + 'nsl_balancing_RNN[].h5')

        return cm, history, pred, y_test
from keras.layers import Conv1D, Dense, Input, LeakyReLU, concatenate, BatchNormalization, Activation, Lambda, MaxPooling1D, \
    Flatten, Dropout
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
import pandas as pd
import numpy.random as random

import random

from collections import Counter

class CNNSiameseNetwork:
    def __init__(self, train_x, train_y, test_x, test_y, n_epochs=15, batch_size=16, latent_dim=32):
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

        self.loss_results = [[], []]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_classes = 5
        self.step_per_epoch = 250

        self.latent_dim = latent_dim
        self.input_dim = (self.train_x.shape[1], 1)

        adam = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.5)
        self.optimizer = adam

        # train_x에서 train_y와 같은 값을 갖는 데이터끼리 그룹화함.
        self.train_groups = [train_x[np.where(train_y == i)[0]] for i in np.unique(train_y)]
        self.test_groups = [test_x[np.where(test_y == i)[0]] for i in np.unique(train_y)]

        self.encoder = LabelEncoder()
        self.encoder.fit(self.train_y)
        self.categorical_train_y = self.encoder.transform(self.train_y)
        self.categorical_test_y = self.encoder.transform(self.test_y)
        self.onehot_train_y = to_categorical(self.categorical_train_y, num_classes=self.num_classes)
        self.onehot_test_y = to_categorical(self.categorical_test_y, num_classes=self.num_classes)

        self.left_input = Input(shape=self.input_dim, name='left_input')
        self.right_input = Input(shape=self.input_dim, name='right_input')

        self.l_encoder = self.convnet(self.left_input)
        self.r_encoder = self.convnet(self.right_input)

        # merge_layer = concatenate([left_feat, right_feat])
        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([self.l_encoder, self.r_encoder])
        merge_layer = Dense(1, activation='sigmoid')(distance)

        self.siamese_net = Model([self.left_input, self.right_input], merge_layer)

        self.siamese_net.compile(loss=self.euclidean_distance_loss,
                                 optimizer=self.optimizer,
                                 metrics=['acc'],
                                 loss_weights=[1.0])

        self.siam_cls = Model(self.left_input, self.build_classifier(self.convnet(self.left_input)), name='siam_cls')

    def convnet(self, input_data):
        conv_filter = [128, 64, 32]

        conv = Conv1D(filters=conv_filter[0], kernel_size=3, input_shape=self.input_dim)(input_data)
        conv = LeakyReLU(alpha=0.1)(conv)
        conv = Dropout(0.5)
        conv = Conv1D(filters=conv_filter[1], kernel_size=3)(conv)
        conv = LeakyReLU(alpha=0.1)(conv)
        conv = Dropout(0.5)
        conv = Conv1D(filters=conv_filter[2], kernel_size=3)(conv)
        conv = LeakyReLU(alpha=0.1)(conv)
        conv = Dropout(0.5)
        # conv = Conv1D(filters=conv_filter[3], kernel_size=3)(conv)
        # conv = LeakyReLU(alpha=0.1)(conv)
        # conv = Conv1D(filters=conv_filter[4], kernel_size=3, name='l_Feature_layer')(conv)
        # conv = LeakyReLU(alpha=0.1)(conv)
        conv = Flatten()(conv)
        conv = Lambda(lambda x: K.l2_normalize(x, axis=-1))(conv)
        conv = Dense(1)(conv)
        conv = Activation('sigmoid')(conv)
        return conv

    # def r_convnet(self, input_data):
    #     conv_filter = [128, 64, 32]
    #
    #     conv = Conv1D(filters=conv_filter[0], kernel_size=3, input_shape=self.input_dim)(input_data)
    #     conv = LeakyReLU(alpha=0.1)(conv)
    #     conv = Conv1D(filters=conv_filter[1], kernel_size=3)(conv)
    #     conv = LeakyReLU(alpha=0.1)(conv)
    #     conv = Conv1D(filters=conv_filter[2], kernel_size=3)(conv)
    #     conv = LeakyReLU(alpha=0.1)(conv)
    #     # conv = Conv1D(filters=conv_filter[3], kernel_size=3)(conv)
    #     # conv = LeakyReLU(alpha=0.1)(conv)
    #     # conv = Conv1D(filters=conv_filter[4], kernel_size=3, name='r_Feature_layer')(conv)
    #     # conv = LeakyReLU(alpha=0.1)(conv)
    #     conv = Flatten()(conv)
    #     conv = Lambda(lambda x: K.l2_normalize(x, axis=-1))(conv)
    #     conv = Dense(1)(conv)
    #     conv = Activation('sigmoid')(conv)
    #     return conv

    def build_classifier(self, input_data):
        cls = Dense(self.train_x.shape[0], activation='relu')(input_data)
        cls = Dense(self.num_classes, activation='softmax')(cls)
        return cls

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))

    def contrastive_loss(self, y_true, y_pred):
        margin = 2
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def gen_random_batch(self, in_groups, batch_halfsize=1):
        out_a, out_b, out_score = [], [], []
        all_groups = list(range(len(in_groups)))

        for class_idx in range(5):
            class_idx = [class_idx]
            for match_group in [True, False]:
                # size 만큼 그룹 ID를 랜덤으로 뽑아냄 (normal :4, Dos:0, ...,)
                group_idx = np.random.choice(class_idx,
                                             size=batch_halfsize)  ## group_idx = [class_idx]*batch_halfsize 5개씩
                # class 0~4까지 순차적으로 5개씩 샘플이 뽑힌다.
                out_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
                if match_group:
                    b_group_idx = group_idx
                    out_score += [1] * batch_halfsize
                else:
                    non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
                    b_group_idx = non_group_idx
                    out_score += [0] * batch_halfsize
                out_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

        return np.stack(out_a, 0), np.stack(out_b, 0), np.stack(out_score, 0)

    def siam_gen(self, in_groups, batch_size=2):
        while True:
            pv_a, pv_b, pv_sim = self.gen_random_batch(in_groups, batch_size // 2)
            yield [pv_a, pv_b], pv_sim

    def main(self):
        valid_a, valid_b, valid_sim = self.gen_random_batch(self.train_groups, 5)

        '''
        ======================
        Siamese model Training
        ======================
        '''
        history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 10),
                                                 steps_per_epoch=self.step_per_epoch,
                                                 validation_data=([valid_a, valid_b], valid_sim),
                                                 epochs=self.n_epochs,
                                                 verbose=True)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Full Siamese Model Training and Validation loss')
        plt.legend()
        plt.show()

        '''
        ==================================
        Classifier model Tranining(Freeze)
        ==================================
        '''
        for i, layer in enumerate(self.siam_cls.layers[0:10]):
            layer.set_weights(self.siamese_net.layers[i * 2].get_weights())
            layer.trainable = False
        self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.siam_cls.summary()

        self.siam_cls.fit(self.train_x, self.onehot_train_y,
                          epochs=self.n_epochs,
                          batch_size=64,
                          verbose=True,
                          validation_split=0.2)

        '''
        ==================================
        Classifier model Tranining(UnFreeze)
        ==================================
        '''

        for i, layer in enumerate(self.siam_cls.layers[0:10]):
            layer.trainable = True
        self.siam_cls.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.siam_cls.summary()

        self.siam_cls.fit(self.train_x, self.onehot_train_y,
                          epochs=self.n_epochs,
                          batch_size=64,
                          verbose=True,
                          validation_split=0.2)

        '''
        ==================
        Full model Testing
        ==================
        '''

        eval = self.siam_cls.evaluate(self.test_x, self.onehot_test_y)
        print(eval)
        pred = self.siam_cls.predict(test_x)
        pred = np.argmax(pred, axis=1)

        cm = confusion_matrix(np.argmax(self.onehot_test_y, axis=1), pred)
        print(cm)
        classes_y = np.unique([test_y])  # y_train은 원핫인코딩 전 라벨인코더 단계를 거친 값
        print(classes_y)
        re, rc = self.train_y.factorize()
        class_names = np.unique([re])

        print(class_names)

        report = classification_report(self.categorical_test_y, pred, labels=class_names, target_names=classes_y)
        print(str(report))


if __name__ == '__main__':
    data_loc = "../dataset/"

    train_data_file = "qnt_KDDTrain_category"
    test_data_file = "qnt_KDDTest_category"
    data_format_txt = ".txt"

    print("Data Loading...")
    train_data = pd.read_csv(data_loc + train_data_file + data_format_txt, sep=',', dtype='unicode')
    test_data = pd.read_csv(data_loc + test_data_file + data_format_txt, sep=',', dtype='unicode')
    print(train_data)

    train_y, train_x = train_data['outcome'], train_data.drop('outcome', 1)
    test_y, test_x = test_data['outcome'], test_data.drop('outcome', 1)

    print(Counter(train_y))

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    print("=========== CNN Siameses Network init ============")
    siam = CNNSiameseNetwork(train_x, train_y, test_x, test_y)
    siam.main()

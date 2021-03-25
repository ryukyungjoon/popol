from keras.layers import Conv2D, Dense, Input, concatenate, Activation, Lambda, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

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
import os
import cv2

from keras.preprocessing.image import load_img

from collections import Counter

random.seed(1)

class CNNSiameseNetwork:
    def __init__(self, train_x, train_y, test_x, test_y, n_epochs=5, batch_size=16, latent_dim=32):
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_classes = 5
        self.step_per_epoch = 2000

        self.latent_dim = latent_dim
        self.input_dim = (self.train_x[0].shape[0], self.train_x[0].shape[1], 3)

        adam = optimizers.Adam(lr=0.00006)
        self.optimizer = adam

        # train_x에서 train_y와 같은 값을 갖는 데이터끼리 그룹화함.
        self.train_groups = [train_x[np.where(train_y == i)[0]] for i in np.unique(train_y)]
        self.test_groups = [test_x[np.where(test_y == i)[0]] for i in np.unique(train_y)]

        encoder = LabelEncoder()
        encoder.fit(self.train_y)
        self.categorical_train_y = encoder.transform(self.train_y)
        self.categorical_test_y = encoder.transform(self.test_y)
        self.onehot_train_y = to_categorical(self.categorical_train_y, num_classes=self.num_classes)
        self.onehot_test_y = to_categorical(self.categorical_test_y, num_classes=self.num_classes)

        print(Counter(self.categorical_train_y))

        left_input = Input(shape=(64, 64, 3), name='left_input')
        right_input = Input(shape=(64, 64, 3), name='right_input')

        convnet = Sequential([
            Conv2D(64, (3, 3), input_shape=(64, 64, 3)),
            Activation('relu'),
            MaxPooling2D(),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            Flatten(),
            Dense(18),
            Activation('sigmoid')
        ])

        left_feat = convnet(left_input)
        right_feat = convnet(right_input)

        # self.compare = Model(inputs=[left_input, right_input], outputs=[left_feat, right_feat])
        # self.compare.compile(loss=self.contrastive_loss, optimizer=self.optimizer)

        merge_layer = concatenate([left_feat, right_feat])
        merge_layer = Dense(1, activation='sigmoid')(merge_layer)

        self.siamese_net = Model([left_input, right_input], merge_layer)
        self.siamese_net.summary()
        self.siamese_net.compile(optimizer=self.optimizer, loss="binary_crossentropy",
                                 metrics=['acc'])
        # encoded_l = convnet(left_input)
        # encoded_r = convnet(right_input)
        #
        # # Getting the L1 Distance between the 2 encodings
        # L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))
        #
        # # Add the distance function to the network
        # L1_distance = L1_layer([encoded_l, encoded_r])
        #
        # prediction = Dense(1, activation='sigmoid')(L1_distance)
        # self.siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        #
        # optimizer = optimizers.Adam(0.001, decay=2.5e-4)
        # self.siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        '''
        ===========================
        classifier 모델 구현
        ===========================
        '''

        self.siam_cls = Model(left_input, self.build_classifier(convnet(left_input)), name='siam_cls')
        self.siam_cls.summary()
        for i, layer in enumerate(self.siam_cls.layers[0:2]):
            print(i, layer)
            layer.set_weights(self.siamese_net.layers[i*2].get_weights())
            # layer.trainable = False
        self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.siam_cls.summary()

    def build_classifier(self, input_data):
        classifier = Dense(122, activation='relu')(input_data)
        # classifier = Dense(1024, activation='relu')(classifier)
        # classifier = Dense(512, activation='relu')(classifier)
        # classifier = Dense(256, activation='relu')(classifier)
        classifier = Dense(128, activation='relu')(classifier)
        classifier = Dense(64, activation='relu')(classifier)
        classifier = Dense(32, activation='relu')(classifier)
        output = Dense(self.num_classes, activation='softmax')(classifier)
        return output

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def gen_random_batch(self, in_groups, batch_halfsize=8):
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

        # Call-back Early Stopping!
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        print('===============TRAINING===============')
        history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 10),
                                                 steps_per_epoch=self.step_per_epoch,
                                                 validation_data=([valid_a, valid_b], valid_sim),
                                                 epochs=self.n_epochs,
                                                 verbose=True,
                                                 callbacks=[cb_early_stopping])
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Training and Validation loss')
        plt.legend()
        plt.show()

        '''
        ===========================
        Model Save
        ===========================
        '''
        model_json = self.siamese_net.to_json()
        with open('./model_save/siamese_net(conv2d).json', 'w') as json_file:
            json_file.write(model_json)
        self.siamese_net.save_weights('./model_save/siamese_net(conv2d).h5')

        ## Classifier Training ##
        print(self.onehot_train_y)
        history = self.siam_cls.fit(self.train_x, self.onehot_train_y,
                                    batch_size=128, epochs=5, verbose=1,
                                    validation_split=0.2, callbacks=[cb_early_stopping])
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(len(loss))
        # plt.plot(epochs, loss, 'bo', label='Training loss')
        # plt.plot(epochs, val_loss, 'b', label='Validation loss')
        # plt.title('Freeze Siamese Training and Validation loss')
        # plt.legend()
        # plt.show()

        # for layer in self.siam_cls.layers[0:2]:
        #     layer.trainable = True
        # self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # self.siam_cls.summary()

        ## Classifier Training ##
        # history = self.siam_cls.fit(self.train_x, self.onehot_train_y,
        #                             batch_size=128, epochs=10, verbose=1,
        #                             validation_split=0.2, callbacks=[cb_early_stopping])
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(len(loss))
        # plt.plot(epochs, loss, 'bo', label='Training loss')
        # plt.plot(epochs, val_loss, 'b', label='Validation loss')
        # plt.title('Full Siamese Model Training and Validation loss')
        # plt.legend()
        # plt.show()

        eval = self.siam_cls.evaluate(self.test_x, self.onehot_test_y)
        print(eval)
        pred = self.siam_cls.predict(self.test_x)
        pred = np.argmax(pred, axis=1)
        print(pred)
        cm = confusion_matrix(np.argmax(self.onehot_test_y, axis=1), pred)
        print(cm)

        classes_y = np.unique(self.train_y)  # y_train은 원핫인코딩 전 라벨인코더 단계를 거친 값
        print(classes_y)
        re, rc = pd.factorize(self.train_y)
        print(re, rc)
        class_names = np.unique([re])

        print(class_names)

        report = classification_report(self.categorical_test_y, pred, labels=class_names, target_names=classes_y)
        print(str(report))

if __name__ == '__main__':
    train_data_loc = "../dataset/NSL-KDD Imagedata/train/picture/"
    test_data_loc = "../dataset/NSL-KDD Imagedata/test/picture/"
    attack_type = ['normal', 'r2l', 'u2r', 'dos', 'probe']
    # label = ['normal', 'R2L', 'U2R', 'DoS', 'Probe']

    train_x, train_label = [], []

    print('Data Loading...')
    for idex, categorie in enumerate(attack_type):
        image_dir = train_data_loc + categorie + '/'
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                print(image_dir + filename)
                img = cv2.imread(image_dir + filename)
                img = cv2.resize(img, None, fx=64 / img.shape[1], fy=64 / img.shape[0])  # img.shape[2] 는 싸이즈 2배가 된다.
                train_x.append(img / 255)  # 픽셀 0 ~ 255가 들어가기 때문에 256로 나누어서 0 ~ 1 사이로 표현
                train_label.append(attack_type[idex])

    train_x, train_label = np.array(train_x), np.array(train_label)

    test_x, test_label = [], []
    for idex, categorie in enumerate(attack_type):
        image_dir = test_data_loc + categorie + '/'
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                print(image_dir + filename)
                img = cv2.imread(image_dir + filename)
                img = cv2.resize(img, None, fx=64 / img.shape[1],
                                 fy=64 / img.shape[0])  # img.shape[2] 는 싸이즈 2배가 된다.
                test_x.append(img / 255)  # 픽셀 0 ~ 255가 들어가기 때문에 256로 나누어서 0 ~ 1 사이로 표현
                test_label.append(attack_type[idex])

    test_x, test_label = np.array(test_x), np.array(test_label)

    siam = CNNSiameseNetwork(train_x, train_label, test_x, test_label)
    siam.main()
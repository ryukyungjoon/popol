from keras.layers import Dense, Input, BatchNormalization, Activation, Lambda, concatenate
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
import pandas as pd
import numpy.random as random


class SiameseNeuralNetwork:
    def __init__(self, train_x, test_x, train_y, test_y):
        self.input_shape = 122
        self.num_classes = 5
        self.batch_size = 10

        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        encoder = LabelEncoder()

        encoder.fit(self.train_y)
        self.categorical_train_y = encoder.transform(self.train_y)
        self.categorical_test_y = encoder.transform(self.test_y)

        self.onehot_train_y = to_categorical(self.categorical_train_y, num_classes=5)
        self.onehot_test_y = to_categorical(self.categorical_test_y, num_classes=5)

        print(self.categorical_train_y, self.categorical_test_y)

        self.train_groups = [train_x[np.where(train_y == i)[0]] for i in
                             np.unique(train_y)]  # train_x에서 train_y와 같은 값을 갖는 데이터끼리 그룹화함.
        self.test_groups = [test_x[np.where(test_y == i)[0]] for i in np.unique(train_y)]

        print('train groups:', [x.shape[0] for x in self.train_groups])  # train 데이터의 class 별 데이터 개수
        print('test groups:', [x.shape[0] for x in self.test_groups])  # test 데이터의 class 별 데이터 개수

        self.x = [x.shape[0] for x in self.train_groups]
        self.z = [x.shape[0] for x in self.test_groups]

        input_data = Input(shape=(122,), name='Feature_Input')
        self.feature_model = self.FeatureNet_Input(input_data)
        self.feature_model.summary()

        left_input = Input(shape=(122,))
        right_input = Input(shape=(122,))

        # Feature model 구성
        left_feat = self.feature_model(left_input)
        right_feat = self.feature_model(right_input)
        L1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        euclidean_distance = L1_distance_layer([left_feat, right_feat])

        adam = optimizers.Adam(lr=0.0006)

        prediction = Dense(1, activation='sigmoid')(euclidean_distance)

        # siamese network model 구성
        self.siamese_net = Model([left_input, right_input], prediction)
        self.siamese_net.compile(optimizer=adam, loss='binary_crossentropy')
        self.siamese_net.summary()

        # Freeze model 구성
        self.siam_IDS = self.Classification(prediction)
        self.siam_IDS.summary()

        # siamese network model의 weights 가져와서 test 데이터 Classifier 훈련
        for l1, l2 in zip(self.siam_IDS.layers[0:4], self.siamese_net[0:4]):
            l1.set_weights(l2.get_weights())
            l1.trainable = False

        self.siam_IDS.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
        self.siam_IDS.summary()

    def Classification(self, prediction):
        c_layer = Dense(self.input_shape, activation='relu')(prediction)
        c_layer = Dense(64, activation='relu')(c_layer)
        c_layer = Dense(32, activation='relu')(c_layer)
        c_layer = Dense(16, activation='relu')(c_layer)
        c_layer = Dense(self.num_classes, activation='softmax')(c_layer)
        c_model = Model(inputs=[prediction], outputs=[c_layer], name='ClassificationModel')
        return c_model

    def FeatureNet_Input(self, input_data):
        n_layer = Dense(1024, activation='linear')(input_data)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Dense(512, activation='linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Dense(256, activation='linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Dense(128, activation='linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Dense(64, activation='linear')(n_layer)
        feature_model = Model(inputs=[input_data], outputs=[n_layer], name='FeatureGenerationModel')

        return feature_model

    def gen_random_batch(self, in_groups, batch_halfsize=1, mode=None):
        out_a, out_b, out_score, a_label, b_label = [], [], [], [], []
        all_groups = list(range(len(in_groups)))

        for match_group in [True, False]:
            if mode == 'test':
                group_idx = [0] * batch_halfsize
                group_idx = np.asarray(group_idx)
                out_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in
                          group_idx]  # group_idx의 개수만큼 group_idx 클래스 샘플이 뽑힌다.
                # out_a는 정답값
                a_label += [[c_idx] for c_idx in group_idx]
                if match_group:
                    b_group_idx = group_idx
                    out_score += [0] * batch_halfsize

                else:
                    non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in
                                     group_idx]
                    b_group_idx = non_group_idx
                    out_score += [1] * batch_halfsize

                out_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in
                          b_group_idx]
                # out_b는 테스트 값
                b_label += [[c_idx] for c_idx in b_group_idx]

            else:
                group_idx = np.random.choice(all_groups,
                                             size=batch_halfsize)  # size 만큼 그룹 ID를 랜덤으로 뽑아냄 (normal :4, Dos:0, ...,)
                out_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in
                          group_idx]  # group_idx의 개수만큼 샘플이 뽑힌다.
                a_label += [[c_idx] for c_idx in group_idx]
                if match_group:
                    b_group_idx = group_idx
                    out_score += [0] * batch_halfsize

                else:
                    non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
                    b_group_idx = non_group_idx
                    out_score += [1] * batch_halfsize

                out_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in
                          b_group_idx]
                b_label += [[c_idx] for c_idx in b_group_idx]
        return np.stack(out_a, 0), np.stack(out_b, 0), np.stack(out_score, 0), np.stack(a_label, 0), np.stack(b_label, 0)

    def siam_gen(self, in_groups, batch_size=2, mode=None):
        while True:
            if mode == 'test':
                pv_a, pv_b, pv_sim, a_label, b_label = self.gen_random_batch(in_groups, batch_size // 2, mode=mode)
                # print('pv_a:{}, pv_b:{}, same(0) or not(1):{}'.format(a_label, b_label, pv_sim))
                yield [pv_a, pv_b], pv_sim
            else:
                pv_a, pv_b, pv_sim, a_label, b_label = self.gen_random_batch(in_groups, batch_size // 2)
                yield [pv_a, pv_b], pv_sim

    def main(self):
        valid_a, valid_b, valid_sim, a_label, b_label = self.gen_random_batch(self.train_groups, 5)

        print('===============TRAINING===============')
        loss_history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 10),
                                                      steps_per_epoch=1,
                                                      validation_data=([valid_a, valid_b], valid_sim),
                                                      epochs=1,
                                                      verbose=True)
        self.siamese_net.fit_generator()

        loss = loss_history.history['loss']
        val_loss = loss_history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        print('============== siam_IDS ==============')
        loss_history = self.siam_IDS.fit_generator(self.siam_gen(self.train_groups, 32),
                                                   steps_per_epoch=1,
                                                   validation_data=([valid_a, valid_b], valid_sim),
                                                   epochs=1,
                                                   verbose=True
                                                   )
        loss = loss_history.history['loss']
        val_loss = loss_history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        for i, layer in enumerate(self.siam_IDS.layers[0:5]):
            layer.trainable = True
        self.siam_IDS.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.siam_IDS.summary()

        loss_history = self.siam_IDS.fit(train_x, self.onehot_train_y, batch_size=32, epochs=10, verbose=1)

        loss = loss_history.history['loss']
        val_loss = loss_history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        pv_a, pv_b, pv_sim, a_label, b_label = self.gen_random_batch(self.test_groups, len(self.test_y) // 2)
        print("test data {}, support data {}, True or False {}".format(pv_a, pv_b, pv_sim))

        print("-- Evaluate --")
        score = self.siam_IDS.evaluate(test_x, test_y)
        print(score)

        print("%s: %.2f%%" % (self.siam_IDS.metrics_names[2], score[2] * 100))

        print("-- Predict --")
        output = self.siam_IDS.predict_generator(self.siam_gen(self.test_groups, self.test_y), steps=1,
                                                    verbose=False)  # test sample 5개 뽑아서 10번 테스트

        print("output : {}".format(output))
        pred = np.round(output)
        print("pred : {}".format(pred))  # Prediction이 0이면 UnSimilarity, 1이면 Similarity
        cm = confusion_matrix(pv_sim, pred)
        print(cm)


if __name__ == '__main__':
    data_path = "../dataset/"

    nsl_tr_data = "qnt_KDDTrain_category"
    nsl_te_data = "qnt_KDDTest_category"
    nsl_data_format = ".txt"

    train_data = pd.read_csv(data_path + nsl_tr_data + nsl_data_format, sep=',', dtype='unicode')
    test_data = pd.read_csv(data_path + nsl_te_data + nsl_data_format, sep=',', dtype='unicode')

    train_y, train_x = train_data['outcome'], train_data.drop('outcome', 1)
    test_y, test_x = test_data['outcome'], test_data.drop('outcome', 1)

    # sb = Semi_balancing()
    # train_x, train_y = sb.nsl_sampling(train_x, train_y, train_x.head(0), 'qnt')

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_x = train_x.reshape(-1, 122)
    test_x = test_x.reshape(-1, 122)

    siam = SiameseNeuralNetwork(train_x, test_x, train_y, test_y)
    siam.main()
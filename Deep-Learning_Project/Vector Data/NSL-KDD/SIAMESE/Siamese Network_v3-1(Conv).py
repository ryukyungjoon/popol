from keras.layers import Dense, Input, Dropout, Lambda, Conv1D, MaxPooling1D, Flatten, Activation, Concatenate
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

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

        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        self.num_classes = 5
        self.batch_size = 10

        encoder = LabelEncoder()

        encoder.fit(self.train_y)
        self.categorical_train_y = encoder.transform(self.train_y)
        self.categorical_test_y = encoder.transform(self.test_y)
        self.onehot_train_y = to_categorical(self.categorical_train_y, num_classes=self.num_classes)
        self.onehot_test_y = to_categorical(self.categorical_test_y, num_classes=self.num_classes)

        print(self.categorical_train_y, self.categorical_test_y)

        # train_x에서 train_y와 같은 값을 갖는 데이터끼리 그룹화함.
        self.train_groups = [train_x[np.where(train_y == i)[0]] for i in np.unique(train_y)]
        self.test_groups = [test_x[np.where(test_y == i)[0]] for i in np.unique(train_y)]

        print('train groups:', [x.shape[0] for x in self.train_groups])  # train 데이터의 class 별 데이터 개수
        print('test groups:', [x.shape[0] for x in self.test_groups])  # test 데이터의 class 별 데이터 개수

        self.x = [x.shape[0] for x in self.train_groups]
        self.z = [x.shape[0] for x in self.test_groups]

        # Feature Vector Network(CNN) 구성
        convnet = Sequential([
            Conv1D(filters=64, kernel_size=3, input_shape=(122, 1)),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=64, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=128, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=256, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(32, activation='sigmoid')
        ])
        convnet.summary()


        # Siamese Neural Network 구성
        self.left_input = Input((122, 1), name='Left_input')
        self.right_input = Input((122, 1), name='Right_input')

        left_feat = convnet(self.left_input)
        right_feat = convnet(self.right_input)

        adam = optimizers.Adam(lr=0.0006)

        def euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        def contrastive_loss(y_true, y_pred):
            margin = 1
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

        self.distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([left_feat, right_feat])

        self.siamese_net = Model(inputs=[self.left_input, self.right_input], outputs=self.distance)
        self.siamese_net.summary()
        self.siamese_net.compile(optimizer=adam, loss=contrastive_loss, metrics=['acc'])

    def show_model_output(self, nb_examples=2):
        for class_idx in range(5):
            pv_a, pv_b, pv_sim, a_label, b_label = self.gen_random_batch(self.test_groups,
                                                                         (self.z[class_idx] // 10) * 5)
            print("test data {}, support data {}, True or False {}".format(pv_a, pv_b, pv_sim))

            print("-- Evaluate --")
            score = self.siamese_net.evaluate_generator(self.siam_gen(self.test_groups, 10),
                                                        steps=self.z[class_idx] // 10,
                                                        verbose=False)  # test sample 5개 뽑아서 10번 테스트
            print(score)

            print("%s: %.2f%%" % (self.siamese_net.metrics_names[2], score[2] * 100))

            print("-- Predict --")
            output = self.siamese_net.predict_generator(self.siam_gen(self.test_groups, 10),
                                                        steps=self.z[class_idx] // 10,
                                                        verbose=False)  # test sample 5개 뽑아서 10번 테스트

            pred = np.round(output)

            pred = np.array(pred)
            print("pred : {}".format(pred))  # Prediction이 0이면 UnSimilarity, 1이면 Similarity
            cm = confusion_matrix(pv_sim, pred)
            print(cm)

            precision = precision_score(pv_sim, pred, average='binary')
            recall = recall_score(pv_sim, pred, average='binary')

            print('precision score {}\nrecall {}'.format(precision, recall))

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
                    out_score += [1] * batch_halfsize

                else:
                    non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in
                                     group_idx]
                    b_group_idx = non_group_idx
                    out_score += [0] * batch_halfsize

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

                out_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]
                b_label += [[c_idx] for c_idx in b_group_idx]

        out_a = np.array(out_a)
        out_b = np.array(out_b)

        return np.stack(out_a, 0), np.stack(out_b, 0), np.stack(out_score, 0), np.stack(a_label, 0), np.stack(b_label,
                                                                                                              0)

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
        valid_a, valid_b, valid_sim, label_a, label_b = self.gen_random_batch(self.train_groups, 25)
        print('===============TRAINING===============')
        history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 50),
                                                 steps_per_epoch=250,
                                                 validation_data=([valid_a, valid_b], valid_sim),
                                                 epochs=100,
                                                 verbose=True)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        self.show_model_output()


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

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    siam = SiameseNeuralNetwork(train_x, test_x, train_y, test_y)
    siam.main()

from keras.layers import Dense, Input, concatenate, BatchNormalization, Activation, Lambda
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
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

        print(self.categorical_train_y, self.categorical_test_y)

        self.train_groups = [train_x[np.where(train_y == i)[0]] for i in np.unique(train_y)]  # train_x에서 train_y와 같은 값을 갖는 데이터끼리 그룹화함.
        self.test_groups = [test_x[np.where(test_y == i)[0]] for i in np.unique(train_y)]

        arr = np.empty((122,), dtype=float)
        print(arr)
        print(np.shape(self.test_groups[0]))
        for i in range(5):
            for j in range(10):
                self.test_10 = np.append(arr, self.test_groups[i][j], axis=0)
        self.test_10_y = []

        print('train groups:', [x.shape[0] for x in self.train_groups])  # train 데이터의 class 별 데이터 개수
        print('test groups:', [x.shape[0] for x in self.test_groups])  # test 데이터의 class 별 데이터 개수

        input_data = Input(shape=(122,), name='Feature_Input')
        self.feature_model = self.FeatureNet_Input(input_data)
        self.feature_model.summary()

        self.left_input = Input(shape=(122,), name='Left_input')
        self.right_input = Input(shape=(122,), name='Right_input')

        left_feat = self.feature_model(self.left_input)
        right_feat = self.feature_model(self.right_input)

        adam = optimizers.Adam(lr=0.0006)

        self.siamese_net = self.merge_Features(left_feat, right_feat)

        self.siamese_net.summary()
        self.siamese_net.compile(optimizer=adam, loss=self.contrastive_loss, metrics=['mae', 'acc'])

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

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

    def merge_Features(self, left_feat, right_feat):
        combined_features = concatenate([left_feat, right_feat], name='merge_features')
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(64, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(32, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(16, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(8, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(5, activation='softmax')(combined_features)
        similarity_model = Model(inputs=[self.left_input, self.right_input], outputs=[combined_features], name='Similarity_Model')

        return similarity_model

    def gen_random_batch(self, in_groups, batch_halfsize=1):
        out_a, out_b, out_score = [], [], []
        all_groups = list(range(len(in_groups)))
        for match_group in [True, False]:
            group_idx = np.random.choice(all_groups, size=batch_halfsize)   # size 만큼 그룹 ID를 랜덤으로 뽑아냄 (normal :1, Dos:1, ...,)
            out_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))]for c_idx in group_idx]   # group_idx의 개수만큼 샘플이 뽑힌다.
            # print('label : {}'.format(group_idx))
            if match_group:
                b_group_idx = group_idx
                out_score += [1]*batch_halfsize
                label_a = group_idx
            else:
                non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
                b_group_idx = non_group_idx
                out_score += [0]*batch_halfsize
                label_b = non_group_idx

            out_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]
        return np.stack(out_a, 0), np.stack(out_b, 0), np.stack(out_score, 0)

    def show_model_output(self, nb_examples=2):
        pv_a, pv_b, pv_sim = self.gen_random_batch(self.test_groups, nb_examples)
        print("test data {}, support data {}, True or False {}".format(pv_a, pv_b, pv_sim))

        print("-- Evaluate --")
        score = self.siamese_net.evaluate_generator(self.siam_gen(self.test_groups, 10), steps=1, verbose=True)      # test sample 5개 뽑아서 10번 테스트
        val = self.siamese_net.evaluate([pv_a, pv_b], pv_sim)
        print(val)

        print("%s: %.2f%%" % (self.siamese_net.metrics_names[2], score[2] * 100))

        print("-- Predict --")
        output = self.siamese_net.predict_generator(self.siam_gen(self.test_groups, 10), steps=1, verbose=True)      # test sample 5개 뽑아서 10번 테스트
        predict = self.siamese_net.predict([pv_a, pv_b])
        print(predict)
        print('================================================')

        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(output)})
        print("output : {}".format(output))
        pred = np.round(output)
        print("pred : {}".format(pred))         # Prediction이 0이면 UnSimilarity, 1이면, Similarity
        cm = confusion_matrix(pv_sim, pred)
        print(cm)

    def siam_gen(self, in_groups, batch_size=2):
        while True:
            pv_a, pv_b, pv_sim = self.gen_random_batch(in_groups, batch_size//2)
            yield [pv_a, pv_b], pv_sim

    def main(self):
        valid_a, valid_b, valid_sim = self.gen_random_batch(self.train_groups, 10)
        loss_history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 10),
                                                      steps_per_epoch=500,
                                                      validation_data=([valid_a, valid_b], valid_sim),
                                                      epochs=1,
                                                      verbose=True)

        loss = loss_history.history['loss']
        val_loss = loss_history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        self.show_model_output(5)

if __name__=='__main__':

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
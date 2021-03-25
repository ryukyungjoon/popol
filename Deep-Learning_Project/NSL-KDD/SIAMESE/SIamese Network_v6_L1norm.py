from imblearn.under_sampling import RandomUnderSampler
from keras.layers import Conv1D, Dense, Input, LeakyReLU, concatenate, BatchNormalization, Activation, Lambda, MaxPooling1D, \
    Flatten
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers as reg

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

from collections import Counter


class CNNSiameseNetwork:
    def __init__(self, train_x, train_y, test_x, test_y, n_epochs=100, batch_size=16, latent_dim=32):
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

        self.loss_results = [[], []]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_classes = 5
        self.step_per_epoch = 2000

        self.latent_dim = latent_dim
        self.input_dim = (122, 1)

        adam = optimizers.Adam(lr=0.0002)
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

        left_input = Input(shape=self.input_dim, name='left_input')
        right_input = Input(shape=self.input_dim, name='right_input')

        left_feat = self.convnet(left_input)
        right_feat = self.convnet(right_input)

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([left_feat, right_feat])
        merge_layer = Dense(1, kernel_regularizer=regularizers.l1(0.01), activation='sigmoid')(distance)
        # merge_layer = Dense(1, activation='sigmoid')(merge_layer)

        self.siamese_net = Model([left_input, right_input], merge_layer)
        self.siamese_net.summary()
        self.siamese_net.compile(optimizer=self.optimizer, loss="binary_crossentropy",
                                 metrics=['acc'])

        '''
        ===========================
        classifier 모델 구현 
        ===========================
        '''

        self.siam_cls = Model(left_input, self.build_classifier(self.convnet(left_input)), name='siam_cls')
        self.siam_cls.summary()

        for i, layer in enumerate(self.siam_cls.layers[0:]):
            print(i, layer)


    def convnet(self, input_data):
        conv_filter = [512, 256, 128, 64, 32]

        conv = Conv1D(filters=conv_filter[0], kernel_size=3, input_shape=(122, 1))(input_data)
        conv = LeakyReLU(alpha=0.2)(conv)
        conv = Conv1D(filters=conv_filter[1], kernel_size=3)(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
        conv = Conv1D(filters=conv_filter[2], kernel_size=3)(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
        conv = Conv1D(filters=conv_filter[3], kernel_size=3)(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
        conv = Conv1D(filters=conv_filter[4], kernel_size=3)(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
        conv = MaxPooling1D()(conv)
        conv = Flatten()(conv)

        return conv

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))

    def build_classifier(self, input_data):
        classifier = Dense(122, activation='relu')(input_data)
        # classifier = Dense(1024, activation='relu')(classifier)
        # classifier = Dense(512, activation='relu')(classifier)
        # classifier = Dense(256, activation='relu')(classifier)
        # classifier = Dense(128, activation='relu')(classifier)
        # classifier = Dense(64, activation='relu')(classifier)
        # classifier = Dense(32, activation='relu')(classifier)
        output = Dense(self.num_classes, activation='softmax')(classifier)
        return output

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def gen_random_batch(self, in_groups, batch_halfsize=1):
        out_a, out_b, out_score = [], [], []
        all_groups = list(range(len(in_groups)))

        # for class_idx in range(5):
        #     class_idx = [class_idx]
        for match_group in [True, False]:
            # size 만큼 그룹 ID를 랜덤으로 뽑아냄 (normal :4, Dos:0, ...,)
            group_idx = np.random.choice(all_groups, size=batch_halfsize) ## group_idx = [class_idx]*batch_halfsize 5개씩
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
        valid_a, valid_b, valid_sim = self.gen_random_batch(self.train_groups, 5)       # 5-shot

        # Call-back Early Stopping!
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        print('===============TRAINING===============')

        history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 20),
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

        for i, layer in enumerate(self.siam_cls.layers[0:11]):
            print(i, layer)
            layer.set_weights(self.siamese_net.layers[i*2].get_weights())
            layer.trainable = False
        self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.siam_cls.summary()

        '''
        ===========================
        Model Save
        ===========================
        '''
        # model_json = self.siamese_net.to_json()
        # with open('./model_save/siamese_net(3).json', 'w') as json_file:
        #     json_file.write(model_json)
        # self.siamese_net.save_weights('./model_save/siamese_net(3).h5')

        ## Classifier Training ##
        print(self.onehot_train_y)
        history = self.siam_cls.fit(self.train_x, self.onehot_train_y,
                                    batch_size=32, epochs=100, verbose=1,
                                    validation_split=0.2, callbacks=[cb_early_stopping])
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Freeze Siamese Training and Validation loss')
        plt.legend()
        plt.show()

        for layer in self.siam_cls.layers[0:10]:
            layer.trainable = True
        self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.siam_cls.summary()

        ## Classifier Training ##
        history = self.siam_cls.fit(self.train_x, self.onehot_train_y,
                                    batch_size=32, epochs=100, verbose=1,
                                    validation_split=0.2, callbacks=[cb_early_stopping])
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Full Siamese Model Training and Validation loss')
        plt.legend()
        plt.show()

        eval = self.siam_cls.evaluate(self.test_x, self.onehot_test_y)
        print(eval)
        pred = self.siam_cls.predict(self.test_x)
        pred = np.argmax(pred, axis=1)

        cm = confusion_matrix(np.argmax(self.onehot_test_y, axis=1), pred)
        print(cm)
        classes_y = np.unique([self.test_y])  # y_train은 원핫인코딩 전 라벨인코더 단계를 거친 값
        print(classes_y)
        re, rc = pd.factorize(self.categorical_test_y)
        class_names = np.unique([re])

        print(class_names)

        report = classification_report(self.categorical_test_y, pred, labels=class_names, target_names=classes_y)
        print(str(report))


if __name__ == '__main__':
    data_loc = "../../dataset/NSL-KDD/"

    train_data_file = "qnt_KDDTrain_category"
    test_data_file = "qnt_KDDTest_category"
    data_format_txt = ".txt"

    print("Data Loading...")
    train_data = pd.read_csv(data_loc + train_data_file + data_format_txt, sep=',', dtype='unicode')
    test_data = pd.read_csv(data_loc + test_data_file + data_format_txt, sep=',', dtype='unicode')
    print(Counter(train_data['outcome']))

    # del_label = ['Probe', 'DoS']
    # for i in range(0, len(del_label)):
    #     del_idx = train_data[train_data['outcome'] == del_label[i]].index
    #     train_data = train_data.drop(del_idx)
    #     del_idx = test_data[test_data['outcome'] == del_label[i]].index
    #     test_data = test_data.drop(del_idx)

    train_y, train_x = train_data['outcome'], train_data.drop('outcome', 1)
    test_y, test_x = test_data['outcome'], test_data.drop('outcome', 1)
    features = train_x.head(0)
    train_dic = {
        'normal': 1000,
        'Probe': 1000,
        'DoS': 1000,
    }

    print("Data Resampling...")

    train_sm = RandomUnderSampler(sampling_strategy=train_dic, random_state=0)
    a, b = train_sm.fit_sample(train_x, train_y)

    label = ['outcome']
    train_x = pd.DataFrame(a, columns=list(features))
    train_y = pd.DataFrame(b, columns=list(label))

    # re_train = pd.concat([train_x, train_y], axis=1)
    # print(re_train)
    # re_train.to_csv('./balancing_nslkdd.csv', index=False, header=True)
    # exit(0)

    print(train_x)
    print(test_x)

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    print("=========== CNN Siameses Network init ============")
    siam = CNNSiameseNetwork(train_x, train_y, test_x, test_y)
    siam.main()

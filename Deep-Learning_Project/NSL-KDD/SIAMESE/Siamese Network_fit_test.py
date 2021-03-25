from keras.layers import Dense, Input, Dropout, Lambda, Conv1D, MaxPooling1D, Flatten, Activation, BatchNormalization, concatenate
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import to_categorical

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

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

        self.input_shape = self.train_x.shape
        self.num_classes = 10
        self.batch_size = 10

        adam = optimizers.Adam(lr=0.0006)
        self.optimizer = adam

        self.encoder = LabelEncoder()

        self.encoder.fit(self.train_y)
        self.categorical_train_y = self.encoder.transform(self.train_y)
        self.categorical_test_y = self.encoder.transform(self.test_y)
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
            Conv1D(filters=32, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Conv1D(filters=32, kernel_size=3),
            Activation('relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(32, activation='sigmoid')
        ])

        # Siamese Neural Network 구성
        left_input = Input((784, 1), name='Left_input')
        right_input = Input((784, 1), name='Right_input')

        left_feat = convnet(left_input)
        right_feat = convnet(right_input)
        merge_layer = concatenate([left_feat, right_feat], name='merge_layer')
        merge_layer = Dense(1, activation='sigmoid')(merge_layer)

        self.siamese_net = Model(inputs=[left_input, right_input], outputs=merge_layer)
        self.siamese_net.summary()
        self.siamese_net.compile(optimizer=adam, loss=self.contrastive_loss, metrics=['mae', 'acc'])

        output = self.build_classifier(merge_layer)
        self.siam_cls = Model([left_input, right_input], output)
        self.siam_cls.summary()
        self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    def build_classifier(self, merge_layer):
        classifier = Dense(784, activation='relu')(merge_layer)
        output = Dense(self.num_classes, activation='softmax')(classifier)

        return output

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


    def main(self):
        # First let's separate the dataset from 1 matrix to a list of matricies
        sample_size = 50
        image_list = np.split(self.train_x[:sample_size], sample_size)
        label_list = np.split(self.train_y[:sample_size], sample_size)

        left_input = []
        right_input = []
        targets = []

        # Number of pairs per image
        pairs = 10
        # Let's create the new dataset to train on
        for i in range(len(label_list)):
            for _ in range(pairs):
                compare_to = i
                while compare_to == i:  # Make sure it's not comparing to itself
                    compare_to = random.randint(0, sample_size-1)
                left_input.append(image_list[i])
                right_input.append(image_list[compare_to])
                if label_list[i] == label_list[compare_to]:  # They are the same
                    targets.append(1.)
                else:  # Not the same
                    targets.append(0.)

        left_input = np.squeeze(np.array(left_input))
        right_input = np.squeeze(np.array(right_input))
        targets = np.squeeze(np.array(targets))

        iceimage = self.train_x[101]
        test_left = []
        test_right = []
        test_targets = []

        for i in range(self.train_y.shape[0] - sample_size):
            test_left.append(iceimage)
            test_right.append(self.train_x[i + sample_size])
            test_targets.append(self.train_y[i + sample_size])

        test_left = np.squeeze(np.array(test_left))
        test_right = np.squeeze(np.array(test_right))
        test_targets = np.squeeze(np.array(test_targets))

        test_targets = self.encoder.transform(test_targets)

        # Convert input shape
        left_input = np.reshape(left_input, (left_input.shape[0], left_input.shape[1], 1))
        right_input = np.reshape(right_input, (right_input.shape[0], right_input.shape[1], 1))
        test_left = np.reshape(test_left, (test_left.shape[0], test_left.shape[1], 1))
        test_right = np.reshape(test_right, (test_right.shape[0], test_right.shape[1], 1))



        print('===============TRAINING===============')
        loss_history = self.siamese_net.fit([left_input, right_input], targets,
                                            steps_per_epoch=250,
                                            epochs=10,
                                            verbose=1,
                                            validation_steps=250,
                                            validation_split=0.2)

        loss = loss_history.history['loss']
        val_loss = loss_history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Siamese Neural Network Training and Validation loss')
        plt.legend()
        plt.show()

        model_json = self.siamese_net.to_json()
        with open('./model_save/siamese_net(fit).json', 'w') as json_file:
            json_file.write(model_json)
        self.siamese_net.save_weights('./model_save/siamese_net(fit).h5')

        for i, layer in enumerate(self.siam_cls.layers[0:4]):
            layer.set_weights(self.siamese_net.layers[i].get_weights())
            layer.trainable = False
        self.siam_cls.summary()
        self.siam_cls.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])

        self.siam_cls.fit(self.train_x, self.onehot_train_y)

        for i, layer in enumerate(self.siam_cls.layers[0:4]):
            layer.trainable = True
        self.siam_cls.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.siam_cls.fit(train_x, train_y)

        pred = self.siamese_net.predict(self.test_x)
        pred = np.argmax(pred, axis=1)
        print(pred)
        cm = confusion_matrix(test_targets, pred)
        print(cm)

if __name__ == '__main__':
    # data_path = "../dataset/"
    #
    # nsl_tr_data = "qnt_KDDTrain_category"
    # nsl_te_data = "qnt_KDDTest_category"
    # nsl_data_format = ".txt"
    #
    # train_data = pd.read_csv(data_path + nsl_tr_data + nsl_data_format, sep=',', dtype='unicode')
    # test_data = pd.read_csv(data_path + nsl_te_data + nsl_data_format, sep=',', dtype='unicode')
    #
    # train_y, train_x = train_data['outcome'], train_data.drop('outcome', 1)
    # test_y, test_x = test_data['outcome'], test_data.drop('outcome', 1)
    #
    # # sb = Semi_balancing()
    # # train_x, train_y = sb.nsl_sampling(train_x, train_y, train_x.head(0), 'qnt')
    #
    # train_x = np.array(train_x)
    # test_x = np.array(test_x)
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    a, b, c = train_x.shape
    a1, b1, c1 = test_x.shape

    train_x = train_x.reshape(-1, b * c, 1)
    test_x = test_x.reshape(-1, b1 * c1, 1)

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    siam = SiameseNeuralNetwork(train_x, test_x, train_y, test_y)
    siam.main()

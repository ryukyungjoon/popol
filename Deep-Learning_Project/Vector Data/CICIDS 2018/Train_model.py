from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from keras import layers, models, Input

from tensorflow.keras.utils import to_categorical

from keras.callbacks import EarlyStopping

import numpy as np

import time

from matplotlib import pyplot as plt

class Train_model:
    def train_model(train_X, train_Y, classes_y, test_X, test_Y):
        print('model training...')
        print('MLP Classifier')
        use_model = MLPClassifier(hidden_layer_sizes=(1000, 500, 100), max_iter=100, random_state=42)

        print('string type [train_Y]:', train_Y)

        mini_batch_size = 128
        batch_size = len(train_Y)
        total_epoch = int(batch_size / mini_batch_size)
        current_batch = 0
        start = time.time()
        for i in range(1, total_epoch):
            end_batch = i * mini_batch_size
            use_model.partial_fit(train_X[current_batch:end_batch], train_Y[current_batch:end_batch], classes=classes_y)
            current_batch = end_batch
        use_model.partial_fit(train_X[current_batch:batch_size], train_Y[current_batch:batch_size], classes=classes_y)
        end = time.time()
        print(f"MLP training time{end-start}s")

        test_Y_pred = use_model.predict(test_X)
        print(test_Y_pred)
        print('Accuracy Performance :', metrics.accuracy_score(test_Y, test_Y_pred))
        use_model_confusion_matrix = confusion_matrix(test_Y, test_Y_pred)

        return use_model_confusion_matrix, test_Y_pred

Nin = 122
Nh_l = [1000, 500, 100] # 히든 레이어 개수
number_of_class = 5     # 분류 클래스 개수
Nout = number_of_class  # 출력 노드의 개수

class DNN(models.Sequential):

    def __init__(self, Nin, Nh_l, Nout):        # 모델 구조 정의
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def dnn_model(train_X, train_Y, test_X, test_Y, norm_type):
        print('DNN Classifier')

        print('model training...')

        train_x = np.array(train_X)
        test_X = np.array(test_X)

        print(train_x.shape)        # (samples, 49)
        print(test_X.shape)         # (samples, 49)

        a, b = train_x.shape        # 2D shape를 가진다.(samples_num, features_num)
        a1, b1 = test_X.shape

        train_x = train_x.reshape(-1, b)
        test_X = test_X.reshape(-1, b1)

        train_x = np.array(train_x)
        test_X = np.array(test_X)

        ## String 2 Float
        l_encoder = LabelEncoder()
        y_train = l_encoder.fit_transform(train_Y)
        y_test = l_encoder.fit_transform(test_Y)

        ## One-Hot
        Onehot_train_Y2 = to_categorical(y_train, num_classes=Nout)
        Onehot_test_Y2 = to_categorical(y_test, num_classes=Nout)
        print('Original Data : {}'.format(train_Y))
        print('Original Data : {}'.format(test_Y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(Onehot_train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(Onehot_test_Y2))

        # Call-back Early Stopping!
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=15)

        # Model Instance 호출
        use_model = DNN(Nin, Nh_l, Nout)

        # Learning
        training_start = time.time()
        history = use_model.fit(train_x, Onehot_train_Y2, epochs=100, batch_size=100,
                                validation_split=0.1, verbose=1)
        training_end = time.time()
        print(f"DNN Training Time : {training_end-training_start}")

        # Model Evaluate
        performance_test = use_model.evaluate(test_X, Onehot_test_Y2)
        print('Test Loss and Accuracy ->', performance_test)

        pred = use_model.predict(test_X)
        print(pred)
        pred = np.argmax(pred, axis=1)
        cm = confusion_matrix(np.argmax(Onehot_test_Y2, axis=1), pred)
        print(cm)

        return cm, history, pred, y_test
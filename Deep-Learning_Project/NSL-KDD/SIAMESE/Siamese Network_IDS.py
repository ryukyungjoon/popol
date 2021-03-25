## Training

class CNNSiameseNetwork:
    def __init__(self, train_x, train_y, test_x, test_y, n_epochs=100, batch_size=16, latent_dim=32):
        self.train_x, self.train_y = train_x, train_y

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

        self.siamese_net = Model([left_input, right_input], merge_layer)
        self.siamese_net.summary()
        self.siamese_net.compile(optimizer=self.optimizer, loss="binary_crossentropy",
                                 metrics=['acc'])

        self.siam_cls = Model(left_input, self.build_classifier(self.convnet(left_input)), name='siam_cls')
        self.siam_cls.summary()


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


def build_classifier(self, input_data):
    classifier = Dense(122, activation='relu')(input_data)
    output = Dense(self.num_classes, activation='softmax')(classifier)
    return output


def euclidean_distance(self, vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(self, shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def model_save(self, model_json, save_file_loc, save_file_name):
    with open(save_file_loc + save_file_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    self.siamese_net.save_weights(save_file_loc + save_file_name + '.h5')


def draw_loss_graph(self, history, title):
    loss = history.history['loss']


val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title(title)
plt.legend()
plt.show()


def gen_random_batch(self, in_groups, batch_halfsize=1):
    out_a, out_b, out_score = [], [], []


    all_groups = list(range(len(in_groups)))

    for match_group in [True, False]:
        # size 만큼 그룹 ID를 랜덤으로 뽑아냄 (normal :4, Dos:0, ...,)
        group_idx = np.random.choice(all_groups, size=batch_halfsize)  ## group_idx = [class_idx]*batch_halfsize 5개씩
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

        def run(self):
            # 5-shot
            valid_a, valid_b, valid_sim = self.gen_random_batch(self.train_groups, 5)

            # Call-back Early Stopping!
            cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

            ''' SiameseNeuralNetwork(SNN) Training '''
            history = self.siamese_net.fit_generator(self.siam_gen(self.train_groups, 20),
                                                     steps_per_epoch=self.step_per_epoch,
                                                     validation_data=([valid_a, valid_b], valid_sim),
                                                     epochs=self.n_epochs,
                                                     verbose=True,
                                                     callbacks=[cb_early_stopping])
            ''' loss function graph '''
            self.draw_loss_graph(history, title='Siamese Training and Validation loss')

            ''' Save Model '''
            model_json = self.siamese_net.to_json()
            save_file_loc = './model_save/'
            save_file_name = 'siamese_net'
            self.model_save(model_json, save_file_loc, save_file_name)

            ''' Transfer Weights & Freeze  '''
            for i, layer in enumerate(self.siam_cls.layers[0:11]):
                layer.set_weights(self.siamese_net.layers[i * 2].get_weights())
                layer.trainable = False
                self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
                self.siam_cls.summary()

            ''' Classifier Training '''
            history = self.siam_cls.fit(self.train_x, self.onehot_train_y,
                                        batch_size=32, epochs=100, verbose=1,
                                        validation_split=0.2, callbacks=[cb_early_stopping])
            self.draw_loss_graph(history, title='Freeze Siamese Training and Validation loss')

            ''' UnFreeze '''
            for layer in self.siam_cls.layers[0:10]:
                layer.trainable = True
            self.siam_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
            self.siam_cls.summary()
            
            ''' Full Model Training '''
            history = self.siam_cls.fit(self.train_x, self.onehot_train_y,
                                        batch_size=32, epochs=100, verbose=1,
                                        validation_split=0.2, callbacks=[cb_early_stopping])

            self.draw_loss_graph(history, title='Full Siamese Model Training and Validation loss')

            ''' Model Test '''
            pred = self.siam_cls.predict(self.test_x)
            pred = np.argmax(pred, axis=1)
            confusion_matrix(np.argmax(self.onehot_test_y, axis=1), pred)
            classes_y = np.unique([self.test_y])
            re, rc = pd.factorize(self.categorical_test_y)
            class_names = np.unique([re])
            report = classification_report(self.categorical_test_y, pred, labels=class_names, target_names=classes_y)
            print(str(report))

if __main__ == '__main__':
    data_loc = "../../dataset/NSL-KDD/"
    train_data_file = "qnt_KDDTrain_category"
    test_data_file = "qnt_KDDTest_category"
    data_format_txt = ".txt"

    ''' Data UnderSampling '''
    train_dic = {
        'normal': 1000,
        'Probe': 1000,
        'DoS': 1000,
    }

    train_sm = RandomUnderSampler(sampling_strategy=train_dic, random_state=0)
    a, b = train_sm.fit_sample(train_x, train_y)

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    siam = CNNSiameseNetwork(train_x, train_y, test_x, test_y)
    siam.run()
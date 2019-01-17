import tensorflow as tf
tf.set_random_seed(1)

from Utils.files import *
from Utils.Normalization import *
from GridMatrix.Grid import *

import keras
import keras.backend as K
from keras import layers
from keras.models import Model
from keras.layers import Input, Lambda

from random import shuffle

class TripletLossLayer(layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def create_base_network(input_shape):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize'))

    return model

def build_model(input_shape):
    anchor_input_layer = Input(shape=input_shape)
    positive_input_layer = Input(shape=input_shape)
    negative_input_layer = Input(shape=input_shape)

    base_network = create_base_network(input_shape)

    anchor_embedding = base_network(anchor_input_layer)
    positive_embedding = base_network(positive_input_layer)
    negative_embedding = base_network(negative_input_layer)

    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([anchor_embedding, positive_embedding, negative_embedding])
    model = Model([anchor_input_layer, positive_input_layer, negative_input_layer], triplet_loss_layer)
    model.compile(loss=None, optimizer='adam')

    embedding_input = Input(shape=input_shape)
    embedding_layer = base_network(embedding_input)
    embedding_model = Model([embedding_input], embedding_layer)

    return model, embedding_model

class DATA():
    def __init__(self, dataset, m, n):
        x_trains, y_trains, x_tests, y_tests = get_ucr_train_test_datasets(dataset)
        x_trains = feature_scaling_datasets(x_trains)
        x_tests = feature_scaling_datasets(x_tests)

        g = Grid(m, n)
        x_train = g.dataset2Matrices(x_trains)
        x_test = g.dataset2Matrices(x_tests)

        img_rows, img_cols = x_train.shape[1:]
        class_set = set(y_trains)
        self.class_set = class_set
        num_classes = len(class_set)
        min_class = min(class_set)

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1,
                                      img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1,
                                    img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0],
                                      img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows,
                                    img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #x_train /= 255
        #x_test /= 255

        self.input_shape = input_shape
        self.num_classes = num_classes

        train_datasets = []
        for i in range(len(x_train)):
            train_datasets.append((x_train[i], y_trains[i]))

        test_datasets = []
        for i in range(len(x_test)):
            test_datasets.append((x_test[i], y_tests[i]))

        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

        class_wise_train_datasets = dict()
        for data in train_datasets:
            if data[1] in class_wise_train_datasets.keys():
                class_wise_train_datasets[data[1]].append(data[0])
            else:
                class_wise_train_datasets[data[1]] = [data[0]]

        self.class_wise_train_datasets = class_wise_train_datasets

def embedding_ED(x, y):
    return np.linalg.norm(x-y)

def triplet_generator(n, class_set, class_wise_train_datasets, embedding_model):
    while True:
        a_batch = []
        p_batch = []
        n_batch = []

        for c in class_set:
            data_cnt = len(class_wise_train_datasets[c])
            x = [i for i in range(data_cnt)]
            shuffle(x)

            # generate anchor
            anchors = []
            for i in range(min(n, data_cnt)):
                anchor_embedding = embedding_model.predict(np.expand_dims(class_wise_train_datasets[c][i], axis=0))
                anchors.append((class_wise_train_datasets[c][i], anchor_embedding))

            negatives = []
            for other in class_set:
                if other == c:
                    continue
                data_cnt = len(class_wise_train_datasets[other])
                x = [i for i in range(data_cnt)]
                shuffle(x)

                for i in range(min(n, data_cnt)):
                    negatives_embedding = embedding_model.predict(np.expand_dims(class_wise_train_datasets[other][i], axis=0))
                    negatives.append((class_wise_train_datasets[other][i], negatives_embedding))

        for i in range(len(anchors)):
            for j in range(len(anchors)):
                if i != j:
                    anc = anchors[i]
                    pos = anchors[j]
                    anc_pos_dist = embedding_ED(anc[1], pos[1])
                    for neg in negatives:
                        anc_neg_dist = embedding_ED(anc[1], neg[1])
                        if anc_neg_dist <= anc_pos_dist+0.2:
                            a_batch.append(anc[0])
                            p_batch.append(pos[0])
                            n_batch.append(neg[0])
        yield [a_batch, p_batch, n_batch], None

if __name__ == '__main__':
    dataset = "synthetic_control"
    m, n = 56, 56
    data = DATA(dataset, m, n)

    model, embedding_model = build_model(data.input_shape)
    #model.summary()
    #embedding_model.summary()

    #f = embedding_model.predict(data.x_train)

    generator = triplet_generator(40, data.class_set, data.class_wise_train_datasets, embedding_model)
    model.compile(loss=None, optimizer='adam')

    for i in range(10):
        d = next(generator)
        if len(d[0][0]) > 0:
            model.fit(d[0], d[1], batch_size=2000, epochs=5)
        else:
            break

    train_embeddings = [x[0] for x in data.train_datasets]
    train_embeddings = embedding_model.predict(np.array(train_embeddings))

    test_embeddings = [x[0] for x in data.test_datasets]
    test_embeddings = embedding_model.predict(np.array(test_embeddings))

    #print(test_embeddings)

    predict_cnt = 0
    error_cnt = 0
    for q_idx in range(len(test_embeddings)):
        q_label = data.test_datasets[q_idx][1]
        q_embed = test_embeddings[q_idx]

        min_dist = 2
        predict_label = -1
        for b_idx in range(len(train_embeddings)):
            b_embed = train_embeddings[b_idx]
            dist = embedding_ED(q_embed, b_embed)
            if dist < min_dist:
                min_dist = dist
                predict_label = data.train_datasets[b_idx][1]

        predict_cnt += 1
        if predict_label != q_label:
            error_cnt += 1

    print('error_rate :', error_cnt/predict_cnt)



'''
[References]
http://krasserm.github.io/2018/02/07/deep-face-recognition/
https://github.com/krasserm/face-recognition
'''
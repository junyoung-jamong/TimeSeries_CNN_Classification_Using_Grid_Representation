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

        print('class_set :', class_set)
        print('img_rows :', img_rows, 'img_columns :', img_cols)

        num_classes = len(class_set)
        min_class = min(class_set)

        y_trains = [y-min_class for y in y_trains]
        y_tests = [y-min_class for y in y_tests]

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

        y_train = keras.utils.to_categorical(y_trains, num_classes)
        y_test = keras.utils.to_categorical(y_tests, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

'''
def triplet_generator():
    while True:
        a_batch = np.random.rand(4, 28, 28, 1)
        p_batch = np.random.rand(4, 28, 28, 1)
        n_batch = np.random.rand(4, 28, 28, 1)
        yield [a_batch, p_batch, n_batch], None
'''

if __name__ == '__main__':
    dataset = "CBF"
    m, n = 28, 28
    data = DATA(dataset, m, n)

    model, embedding_model = build_model(data.input_shape)
    model.summary()
    embedding_model.summary()

    f = embedding_model.predict(data.x_train)
    print(f)

    #generator = triplet_generator()
    #model.compile(loss=None, optimizer='adam')
    #model.fit_generator(generator, epochs=10, steps_per_epoch=100)

'''
[References]
http://krasserm.github.io/2018/02/07/deep-face-recognition/
https://github.com/krasserm/face-recognition
'''
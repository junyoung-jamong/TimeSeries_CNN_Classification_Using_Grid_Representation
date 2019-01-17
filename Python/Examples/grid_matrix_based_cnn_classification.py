from Utils.files import *
from Utils.Normalization import *
from GridMatrix.Grid import *
from GridMatrix.Validation import *

import keras
from keras import models, layers
from keras import backend

class CNN(models.Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.add(layers.MaxPool2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.add(layers.MaxPool2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer='rmsprop',
                     metrics=['accuracy'])

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

        if backend.image_data_format() == 'channels_first':
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

if __name__ == '__main__':
    batch_size = 5
    epochs = 100

    dataset = "Gun_Point"

    data = DATA(dataset, 28, 28)
    model = CNN(data.input_shape, data.num_classes)
    model.summary()

    history = model.fit(data.x_train, data.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_split=0.1)

    score = model.evaluate(data.x_test, data.y_test)
    print()
    print('Test loss:', score[0])
    print('Test accuracy : ', score[1])
    error_rate = round(1 - score[1], 3)

    print('error rate of :', error_rate)

    # plotting results
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

from Utils.files import *
from Utils.Normalization import *
from GridMatrix.Grid import *
from GridMatrix.Validation import *
import matplotlib.pylab as plt

if __name__ == '__main__':
    x_trains, y_trains, x_tests, y_tests = get_example_train_test_datasets()
    x_trains = feature_scaling_datasets(x_trains)
    x_tests = feature_scaling_datasets(x_tests)

    # create grid with 15rows and 30columns
    g = Grid(15, 30)

    # find best parameters(m, n) with train error rate
    #g.train(x_trains, y_trains)

    # transpose time series to grid-matrix
    x_matrices_train = g.dataset2Matrices(x_trains)
    x_matrices_test = g.dataset2Matrices(x_tests)

    # conducts 1-nn classification and gets test error rate
    error_rate = one_nn_classification(x_matrices_train, y_trains, x_matrices_test, y_tests)
    print('1-NN classification Test Error Rate :', error_rate)

    # visualize time series representation sample
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(x_tests[0])
    ax2.imshow(x_matrices_test[0], interpolation='nearest', cmap='gray', aspect='auto')
    plt.show()
import sys
from GridMatrix.Similarity import *

def one_nn_classification(x_matrices_train, y_trains, x_matrices_test, y_tests):
    query_cnt = 0
    error_cnt = 0

    for q_idx in range(len(x_matrices_test)):
        query = x_matrices_test[q_idx]
        predict_label = -1
        min_dist = sys.float_info.max

        for b_idx in range(len(x_matrices_train)):
            base = x_matrices_train[b_idx]
            sim = GMED(base, query)
            if sim < min_dist:
                min_dist = sim
                predict_label = y_trains[b_idx]

        query_cnt += 1
        if predict_label != y_tests[q_idx]:
            error_cnt += 1

    return error_cnt / query_cnt

def loocv(train_matrices, y_trains):
    query_cnt = 0
    error_cnt = 0

    n = len(train_matrices)

    for q_idx in range(n):
        query = train_matrices[q_idx]
        predict_label = -1
        min_dist = sys.float_info.max

        for b_idx in range(n):
            if q_idx == b_idx:
                continue

            base = train_matrices[b_idx]
            sim = GMED(base, query)
            if sim < min_dist:
                min_dist = sim
                predict_label = y_trains[b_idx]

        query_cnt += 1
        if predict_label != y_trains[q_idx]:
            error_cnt += 1

    return error_cnt/query_cnt


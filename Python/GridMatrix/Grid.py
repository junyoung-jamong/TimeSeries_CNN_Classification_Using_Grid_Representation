import sys
import numpy as np
from GridMatrix.Validation import loocv

class Grid:
    def __init__(self, m=5, n=5):
        self.m = m
        self.n = n

    #find the best parameters(m, n) for matrix representation using train datasets
    def train(self, x_trains, y_trains):
        best_m, best_n = self.step1(x_trains, y_trains)
        best_m, best_n = self.step2(x_trains, y_trains, best_m, best_n)
        self.m = best_m
        self.n = best_n

    def step1(self, x_trains, y_trains):
        min_error_rate = sys.float_info.max

        best_m = 5
        best_n = 5

        for m in range(5, 40, 5):
            for n in range(5, 35, 5):
                self.m = m
                self.n = n
                train_matrices = self.dataset2Matrices(x_trains)
                error_rate = loocv(train_matrices, y_trains)

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_m = m
                best_n = n

        return best_m, best_n

    def step2(self, x_trains, y_trains, center_m, center_n):
        min_error_rate = sys.float_info.max

        for m in range(center_m-4, center_m+5):
            for n in range(center_n-4, center_n+5):
                self.m = m
                self.n = n
                train_matrices = self.dataset2Matrices(x_trains)
                error_rate = loocv(train_matrices, y_trains)

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_m = m
                best_n = n
                self.x_matrices_train = train_matrices

        return best_m, best_n


    def dataset2Matrices(self, ts_set):
        matrices = []
        for ts in ts_set:
            matrices.append(self.ts2Matrix(ts))

        return matrices

    def ts2Matrix(self, ts):
        matrix = np.zeros((self.m, self.n))
        T = len(ts)

        height = 1.0/self.m
        width = T/self.n

        for idx in range(T):
            i = int((1-ts[idx])/height)
            if i == self.m:
                i -= 1

            t = idx+1
            j = t/width
            if int(j) == round(j, 7):
                j = int(j)-1
            else:
                j = int(j)

            matrix[i][j] += 1
        return matrix

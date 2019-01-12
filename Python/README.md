# TimeSeries CNN Classification Using Grid Representation
This project aims to apply the CNN algorithm - which has achieved great results in image processing - by treating the time-series data as an image.
Grid-based representation algorithms are used to represent time-series data as images. 

Representation
----------------------
To represent the time series as image data, we partition the time series using the m x n grid structure.
[Grid representation](https://link.springer.com/article/10.1007/s10115-018-1264-0) is a compression technique that transforms a time series into a matrix format, while maintaining the point distribution of the original time series.

In the following, a detailed algorithm for transforming time-series into the grid is presented with python code.
```
def ts2Matrix(self, ts):
    matrix = np.zeros((self.m, self.n))
    T = len(ts)

    height = 1.0/self.m  # cell's height of grid 
    width = T/self.n  # cell's width of grid

    for idx in range(T):
        i = int((1-ts[idx])/height)
        if i == self.m:
            i -= 1

        t = idx+1
        j = t/width
        if int(j) == round(j, 7):  # If the point is at the cell boundary
            j = int(j)-1
        else:
            j = int(j)

        matrix[i][j] += 1
    return matrix
```

if m=5, n=7

>input: [0.11, 0.22, 0.44, 0.56, 0.78, 0.11, 0.22, 0.00, 0.44, 0.67, 0.22, 0.00, 1.00, 0.44]
>
>output: [[0 0 0 0 0 0 1], [0 0 1 0 1 0 0], [0 2 0 0 1 0 1], [1 0 0 1 0 1 0], [1 0 1 1 0 1 0]]


#### Grid representation example
![representation_sample](./assets/img/Grid_representation_of_sample_from_CBF_dataset.png)
Grid representation of sample from CBF(UCR archive) dataset. The partition
matrix is 15 × 30

Example
----------------------
 * [Grid representation classification](./Examples/grid_matrix_sample.py) reproduced the algorithm and experiment of "Similarity measures for time series data classification using
grid representation and matrix distance(2018)"

Credits
----------------------
 Grid matrix representation of time-series borrows algorithms from research paper. Below a complete list of credits can be found.
 
 * Yangqing Ye, et al., [Similarity measures for time series data classification using
grid representation and matrix distance](https://link.springer.com/article/10.1007/s10115-018-1264-0), 2018.
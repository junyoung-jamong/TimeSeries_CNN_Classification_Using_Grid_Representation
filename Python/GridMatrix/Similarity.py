import math
import numpy as np

def GMED(gm1, gm2):
    sum = 0

    m = len(gm1)
    n = len(gm1[0])

    for i in range(m):
        for j in range(n):
            d = gm1[i][j] - gm2[i][j]
            sum += d*d
    return math.sqrt(sum)

def GMDTW(gm1, gm2):
    len1 = len(gm1)
    len2 = len(gm2)
    
    dist_matrix = np.zeros(len1, len2)

    for i in range(len1):
        for j in range(len2):
            d = ED(gm1[i], gm2[j])
            if i == 0 and j == 0:
                dist_matrix[i][j] = d
            elif i == 0:
                dist_matrix[i][j] = d+dist_matrix[i][j-1]
            elif j == 0:
                dist_matrix[i][j] = d+dist_matrix[i-1][j]
            else:
                dist_matrix[i][j] = d+min(dist_matrix[i-1][j-1], dist_matrix[i-1][j], dist_matrix[i][j-1])

    return dist_matrix[len1-1][len2-1]

def ED(a, b):
    sum = 0
    for i in range(len(a)):
        d = a[i] - b[i]
        sum += d * d
    return math.sqrt(sum)

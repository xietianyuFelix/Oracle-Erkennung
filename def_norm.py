import numpy as np


def norm(matrix):
    columnMax = matrix.max(axis=0)
    columnMin = matrix.min(axis=0)
    mat = (matrix - columnMin) / (columnMax - columnMin)
    return mat


# x = np.array([[1000, 10, 0.5],
#               [765, 5, 0.35],
#               [800, 7, 0.09]])

# print(norm(x))
# #
# # l1 = [1000, 765, 800]
# # a = np.array(l1)
# #
# # print(norm(a))
# #
# # print(type(a))
# print(norm(x).shape)


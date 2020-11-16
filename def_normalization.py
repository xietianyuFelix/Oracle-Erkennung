import numpy as np
import os
import def_norm as nor
import gc

'''
a test data + train data => normalization
'''


def normalizator(testSetMat, dataSetMat):
    testData = np.reshape(testSetMat, (1, -1))
    normedMat = nor.norm(np.vstack((testData, dataSetMat)))
    return normedMat[0, :], np.delete(normedMat, 0, axis=0)


# mat1 = np.array([1, 10, 100, 1000])
# mat2 = np.array([[2, 20, 200, 2000],
#                  [3, 30, 300, 3000],
#                  [4, 40, 400, 4000],
#                  [5, 50, 500, 5000]])
# # m = np.mean(mat2, axis=0)
# # print(m)
#
# #
# a, b = normalizator(mat1, mat2)
# print(a)
# print(b)
# print(a.shape)
# print(b.shape)
# print(type(a))
# aa = np.reshape(a, (1, -1))
# print(aa)
# print(aa.shape)

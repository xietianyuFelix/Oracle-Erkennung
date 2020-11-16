import numpy as np

'''
"def giveValueOfComponentOf_" is called for catching horizontal component
"def giveValueOfComponentOfI" is called for catching vertical component
"def giveValueOfComponentOfSlash" is called for catching slash component
"def giveValueOfComponentOfBackslash" is called for catching backslash component

input image: black character white background (size must odd number)
'''


def convolution(k, data):
    n, m = data.shape
    img_new = []
    for i in range(n - 2):
        line = []
        for j in range(m - 2):
            a = data[i:i + 3, j:j + 3]
            mul_a_k = np.multiply(k, a)
            sum_a_k = np.sum(mul_a_k)
            if sum_a_k >= 3 * 255:
                line.append(255.0)
            elif sum_a_k == 2 * 255:
                line.append(127.0)
            else:
                line.append(0.0)
        img_new.append(line)
    return np.array(img_new)


# # for alone point and start point
# def convolution2(k, data):
#     n, m = data.shape
#     img_new = []
#     for i in range(n - 2):
#         line = []
#         for j in range(m - 2):
#             a = data[i:i + 3, j:j + 3]
#             mul_a_k = np.multiply(k, a)
#             sum_a_k = np.sum(mul_a_k)
#             if sum_a_k >= 2 * 255:
#                 line.append(255.0)
#             elif sum_a_k == 255:
#                 line.append(127.0)
#             else:
#                 line.append(0.0)
#         img_new.append(line)
#     return np.array(img_new)


# capture horizontal, return value
def giveValueOfComponentOf_(matrix):
    matrix_inv = 255 - matrix
    filFor_ = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]])
    value = np.sum(convolution(filFor_, matrix_inv))
    return value/255


# capture horizontal, return mat
def giveMatrixOfComponentOf_(matrix):
    matrix_inv = 255 - matrix
    filFor_ = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]])
    resMatrix = convolution(filFor_, matrix_inv)
    return resMatrix


# capture vertical, return value
def giveValueOfComponentOfI(matrix):
    matrix_inv = 255 - matrix
    filForI = np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]])
    value = np.sum(convolution(filForI, matrix_inv))
    return value/255


# capture vertical, return mat
def giveMatrixOfComponentOfI(matrix):
    matrix_inv = 255 - matrix
    filForI = np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]])
    resMatrix = convolution(filForI, matrix_inv)
    return resMatrix


# capture slash, return value
def giveValueOfComponentOfSlash(matrix):
    matrix_inv = 255 - matrix
    filForSlash = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]])
    value = np.sum(convolution(filForSlash, matrix_inv))
    return value/255


# capture slash, return mat
def giveMatrixOfComponentOfSlash(matrix):
    matrix_inv = 255 - matrix
    filForSlash = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]])
    resMatrix = convolution(filForSlash, matrix_inv)
    return resMatrix


# capture backslash, return value
def giveValueOfComponentOfBacklash(matrix):
    matrix_inv = 255 - matrix
    filForBacklash = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    value = np.sum(convolution(filForBacklash, matrix_inv))
    return value/255


# capture backslash, return mat
def giveMatrixOfComponentOfBackSlash(matrix):
    matrix_inv = 255 - matrix
    filForBacklash = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    resMatrix = convolution(filForBacklash, matrix_inv)
    return resMatrix



